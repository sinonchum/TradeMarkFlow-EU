"""
TradeMarkFlow-EU — LanceDB Tool 定义

为 Agno Agent 提供 LanceDB 混合检索能力。
封装为标准的 Agno Tool 函数，Agent 通过 Tool Calling 自动调用。
"""

from __future__ import annotations

import json
from typing import Optional

import lancedb
import numpy as np

from app.models.schema import DB_PATH, TABLE_NAME, get_db_connection


def _get_table(db_path: str = DB_PATH) -> lancedb.table.Table:
    """获取 LanceDB 表引用"""
    db = get_db_connection(db_path)
    return db.open_table(TABLE_NAME)


def hybrid_search_trademarks(
    query_text: str,
    query_vector: list[float] | None = None,
    nice_classes: list[int] | None = None,
    status_filter: str = "",
    country_filter: str = "",
    filing_date_after: str = "",
    limit: int = 20,
) -> str:
    """
    在 LanceDB 商标库中执行混合检索。

    支持三种检索模式：
    1. 有 query_vector → 向量语义检索（cosine similarity）
    2. 有 query_text 无向量 → SQL 文本匹配（LIKE）
    3. 两者都有 → 先向量检索再 SQL 过滤

    Args:
        query_text: 搜索关键词，如 "SolarFlux"
        query_vector: 商标名称的嵌入向量（可选，1024维 float list）
        nice_classes: 尼斯分类过滤，如 [9, 35]
        status_filter: 状态过滤，如 "registered"
        country_filter: 申请人国家，如 "FR"
        filing_date_after: 申请日起始，如 "2020-01-01"
        limit: 返回结果数量上限

    Returns:
        JSON 字符串，包含匹配的商标列表
    """
    try:
        table = _get_table()

        # 构建查询
        if query_vector and len(query_vector) > 0:
            # ── 向量语义检索 ──
            lance_query = table.search(
                query_vector,
                vector_column_name="text_vector",
            ).metric("cosine").limit(limit)
        elif query_text:
            # ── SQL 文本匹配（无向量时回退） ──
            lance_query = table.search().limit(limit)
        else:
            # ── 纯过滤 ──
            lance_query = table.search().limit(limit)

        # ── SQL 元数据过滤 ──
        if nice_classes:
            # LanceDB list 包含查询
            for nc in nice_classes:
                lance_query = lance_query.where(f"array_contains(nice_classes, {nc})")

        if status_filter:
            lance_query = lance_query.where(f"status = '{status_filter}'")

        if country_filter:
            lance_query = lance_query.where(f"applicant_country = '{country_filter}'")

        if filing_date_after:
            lance_query = lance_query.where(f"filing_date >= '{filing_date_after}'")

        if query_text and not query_vector:
            # SQL LIKE 模糊匹配
            lance_query = lance_query.where(f"mark_name LIKE '%{query_text}%'")

        # 执行查询
        results = lance_query.to_list()

        # 清理向量字段（不返回给 Agent，节省 token）
        for r in results:
            r.pop("text_vector", None)
            r.pop("image_vector", None)
            # 计算相似度分数（如果有的话）
            if "_distance" in r:
                r["similarity_score"] = round(1.0 - r["_distance"], 4)  # cosine distance → similarity
                del r["_distance"]

        return json.dumps({
            "success": True,
            "count": len(results),
            "results": results,
        }, ensure_ascii=False, default=str)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "results": [],
        })


def sql_query_trademarks(
    where_clause: str,
    limit: int = 50,
) -> str:
    """
    直接执行 SQL 过滤查询（不涉及向量检索）。

    适用于：
    - 按申请号精确查找
    - 按申请人名称搜索
    - 按日期范围统计
    - 按尼斯分类组合过滤

    Args:
        where_clause: SQL WHERE 子句，如 "applicant_name LIKE '%Google%'"
        limit: 返回上限

    Returns:
        JSON 字符串
    """
    try:
        table = _get_table()
        results = table.search().where(where_clause).limit(limit).to_list()

        for r in results:
            r.pop("text_vector", None)
            r.pop("image_vector", None)

        return json.dumps({
            "success": True,
            "count": len(results),
            "results": results,
        }, ensure_ascii=False, default=str)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
        })


def get_trademark_by_application_number(application_number: str) -> str:
    """
    通过申请号精确查询单条商标记录。

    Args:
        application_number: EUIPO 申请号

    Returns:
        JSON 字符串，包含完整商标记录
    """
    try:
        table = _get_table()
        results = (
            table.search()
            .where(f"application_number = '{application_number}'")
            .limit(1)
            .to_list()
        )

        if results:
            r = results[0]
            r.pop("text_vector", None)
            r.pop("image_vector", None)
            return json.dumps({"success": True, "record": r}, ensure_ascii=False, default=str)
        else:
            return json.dumps({"success": False, "error": f"未找到申请号: {application_number}"})

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def get_database_stats() -> str:
    """获取数据库统计信息"""
    try:
        table = _get_table()
        count = table.count_rows()

        # 统计各状态数量
        statuses = table.search().to_pandas()
        status_counts = statuses["status"].value_counts().to_dict() if count > 0 else {}

        return json.dumps({
            "success": True,
            "total_records": count,
            "status_distribution": status_counts,
            "table_name": TABLE_NAME,
        }, ensure_ascii=False, default=str)

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
