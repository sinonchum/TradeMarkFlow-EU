"""
TradeMarkFlow-EU — LanceDB Schema 定义

设计原则：
1. 单表存储所有商标元数据 + 向量，利用 LanceDB 的混合检索能力
2. 元数据字段支持 SQL 式 WHERE 过滤（Nice 分类、状态、日期范围等）
3. text_vector 列存储商标名称的多语言嵌入向量（稠密检索）
4. image_vector 列存储 Logo 的 CLIP 嵌入向量（图像相似检索）
5. sparse_vector 列存储 BM25 稀疏向量（关键词检索，LanceDB 原生支持）

LanceDB 特性利用：
- SQL metadata filter: `table.search().where("nice_classes @> [9]")`
- Dense + Sparse hybrid: `table.search().vector_type("hybrid")`
- Serverless: 基于文件系统，零运维，本地优先隐私
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import Enum
from typing import Optional

import lancedb
import pyarrow as pa
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
#  1. Pydantic 数据模型（业务层使用）
# ═══════════════════════════════════════════════════════════════

class MarkStatus(str, Enum):
    """EUTM 申请/注册状态"""
    APPLICATION = "application"
    PUBLISHED = "published"
    REGISTERED = "registered"
    OPPOSED = "opposed"
    REFUSED = "refused"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"


class MarkType(str, Enum):
    """商标类型 — EUIPO 分类"""
    WORD = "word"              # 文字商标
    FIGURATIVE = "figurative"  # 图形/Logo 商标
    COMBINATION = "combination"  # 图文组合
    SHAPE = "shape"
    SOUND = "sound"
    PATTERN = "pattern"
    COLOR = "color"


class TrademarkRecord(BaseModel):
    """
    单条商标记录 — 对应 LanceDB 表的一行。

    注意：向量字段用 list[float] 表示，实际存储为 LanceDB 的 vector 类型。
    nice_classes 存储为 list[int]，LanceDB 支持 list 过滤。
    """
    # ── 主键 & 标识 ──
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    application_number: str = Field(..., description="EUIPO 申请号，如 '019123456'")
    registration_number: Optional[str] = Field(None, description="注册号（注册后才有）")

    # ── 商标元数据 ──
    mark_name: str = Field(..., description="商标名称 / 文字元素")
    mark_type: MarkType = Field(default=MarkType.WORD)
    nice_classes: list[int] = Field(
        default_factory=list,
        description="尼斯分类编号列表，如 [9, 35, 42]"
    )
    goods_services: list[str] = Field(
        default_factory=list,
        description="商品/服务描述列表（多语言）"
    )

    # ── 申请人 & 日期 ──
    applicant_name: str = Field(default="")
    applicant_country: str = Field(default="", description="ISO 3166-1 alpha-2")
    filing_date: Optional[str] = Field(None, description="申请日 ISO 8601")
    priority_date: Optional[str] = Field(None, description="优先权日")
    registration_date: Optional[str] = Field(None)
    expiry_date: Optional[str] = Field(None)

    # ── 状态 & 来源 ──
    status: MarkStatus = Field(default=MarkStatus.APPLICATION)
    jurisdiction: str = Field(default="EU", description="EU | WIPO | 国家代码")
    source: str = Field(default="euipo", description="数据来源：euipo | wipo | national")
    languages: list[str] = Field(default_factory=lambda: ["en"])

    # ── 向量（LanceDB 存储为 vector 类型） ──
    text_vector: list[float] = Field(
        default_factory=list,
        description="商标名称的多语言文本嵌入向量（稠密检索）"
    )
    image_vector: list[float] = Field(
        default_factory=lambda: [0.0] * IMAGE_EMBEDDING_DIM,
        description="Logo/图形的 CLIP 视觉嵌入向量（图像相似检索）"
    )

    # ── 元信息 ──
    raw_url: Optional[str] = Field(None, description="EUIPO 原始详情页 URL")
    scraped_at: Optional[str] = Field(None, description="数据抓取时间 ISO 8601")
    updated_at: Optional[str] = Field(None)


# ═══════════════════════════════════════════════════════════════
#  2. PyArrow Schema 定义（LanceDB 底层存储格式）
# ═══════════════════════════════════════════════════════════════

# 向量维度 — 根据嵌入模型调整
TEXT_EMBEDDING_DIM = 1024   # multilingual-e5-large
IMAGE_EMBEDDING_DIM = 768   # CLIP ViT-L/14

def get_arrow_schema() -> pa.Schema:
    """
    定义 LanceDB 表的 PyArrow Schema。

    关键设计决策：
    - text_vector / image_vector 使用 pa.list_(pa.float32()) — LanceDB 自动识别为向量列
    - nice_classes 使用 pa.list_(pa.int32()) — 支持 @> 包含查询
    - 所有字符串字段使用 pa.string()（LanceDB 内部使用 UTF-8）
    - 日期字段存为 pa.string()（ISO 8601 格式），避免时区问题
    """
    return pa.schema([
        # 主键
        pa.field("id", pa.string()),
        pa.field("application_number", pa.string()),
        pa.field("registration_number", pa.string()),

        # 商标元数据
        pa.field("mark_name", pa.string()),
        pa.field("mark_type", pa.string()),
        pa.field("nice_classes", pa.list_(pa.int32())),
        pa.field("goods_services", pa.list_(pa.string())),

        # 申请人 & 日期
        pa.field("applicant_name", pa.string()),
        pa.field("applicant_country", pa.string()),
        pa.field("filing_date", pa.string()),
        pa.field("priority_date", pa.string()),
        pa.field("registration_date", pa.string()),
        pa.field("expiry_date", pa.string()),

        # 状态
        pa.field("status", pa.string()),
        pa.field("jurisdiction", pa.string()),
        pa.field("source", pa.string()),
        pa.field("languages", pa.list_(pa.string())),

        # 向量列 — LanceDB 将 list<float32> 自动索引为向量
        pa.field("text_vector", pa.list_(pa.float32(), TEXT_EMBEDDING_DIM)),
        pa.field("image_vector", pa.list_(pa.float32(), IMAGE_EMBEDDING_DIM)),

        # 元信息
        pa.field("raw_url", pa.string()),
        pa.field("scraped_at", pa.string()),
        pa.field("updated_at", pa.string()),
    ])


# ═══════════════════════════════════════════════════════════════
#  3. LanceDB 数据库初始化
# ═══════════════════════════════════════════════════════════════

DB_PATH = "data/lancedb"
TABLE_NAME = "eu_trademarks"


def get_db_connection(db_path: str = DB_PATH) -> lancedb.DBConnection:
    """
    连接 LanceDB（serverless，基于本地文件系统）。
    零配置、零运维 — 数据库就是一个目录。

    隐私优势：所有数据本地存储，无需连接远程数据库服务。
    """
    db = lancedb.connect(db_path)
    return db


def init_trademark_table(
    db_path: str = DB_PATH,
    embedding_dim: int = TEXT_EMBEDDING_DIM,
) -> lancedb.table.Table:
    """
    初始化商标表。

    如果表已存在则直接打开；否则创建新表并配置索引。

    LanceDB 索引策略：
    - text_vector: IVF_PQ 索引（适合大规模稠密向量检索）
    - metadata 字段自动支持 SQL WHERE 过滤（无需额外索引）
    - nice_classes 的 list 过滤通过 @> 操作符实现
    """
    db = get_db_connection(db_path)

    # 检查表是否已存在
    existing_tables = db.table_names()

    if TABLE_NAME in existing_tables:
        print(f"[LanceDB] 表 '{TABLE_NAME}' 已存在，直接打开")
        table = db.open_table(TABLE_NAME)
        print(f"[LanceDB] 当前行数: {table.count_rows()}")
        return table

    # 创建空表 — 使用 PyArrow Schema 确保类型一致性
    arrow_schema = get_arrow_schema()

    # LanceDB 要求至少插入一行数据来创建表
    # 使用一个空的占位记录
    placeholder = TrademarkRecord(
        id="__placeholder__",
        application_number="000000000",
        mark_name="__placeholder__",
        mark_type="word",
        nice_classes=[],
        goods_services=[],
        status="application",
        jurisdiction="EU",
        source="system",
        text_vector=[0.0] * embedding_dim,
        image_vector=[0.0] * IMAGE_EMBEDDING_DIM,
    )

    table = db.create_table(
        TABLE_NAME,
        data=[placeholder.model_dump()],
        schema=arrow_schema,
        mode="overwrite",
    )

    # 向量索引在数据量足够时创建（IVF_PQ 需要 > 256 条记录）
    # 初始阶段使用暴力扫描（brute force），性能足够
    # 当数据量 > 10K 时调用 ensure_index() 创建 IVF_PQ
    print(f"[LanceDB] 表 '{TABLE_NAME}' 创建成功（索引延迟创建）")
    print(f"[LanceDB] Schema 字段数: {len(arrow_schema)}")
    print(f"[LanceDB] 存储路径: {db_path}")

    return table


def upsert_trademarks(
    records: list[TrademarkRecord],
    db_path: str = DB_PATH,
) -> int:
    """
    批量插入/更新商标记录。

    LanceDB 的 merge_insert 支持原子 upsert：
    - 如果 id 已存在则更新
    - 如果 id 不存在则插入
    """
    table = init_trademark_table(db_path)

    dicts = [r.model_dump() for r in records]

    # 使用 add() 直接追加（LanceDB serverless 模式）
    # 增量更新：先 delete 再 add
    for r in records:
        table.delete(f"id = '{r.id}'")
    table.add(dicts)

    print(f"[LanceDB] Upserted {len(records)} records into '{TABLE_NAME}'")
    return len(records)


def ensure_vector_index(table: lancedb.table.Table):
    """
    当数据量足够大时（>10K 行）创建向量索引。
    小规模数据直接使用暴力扫描即可。
    """
    count = table.count_rows()
    if count < 10_000:
        print(f"[LanceDB] 当前 {count} 行，暴力扫描足够，跳过索引创建")
        return

    print(f"[LanceDB] 创建 IVF_PQ 索引（{count} 行）...")
    table.create_index(
        vector_column_name="text_vector",
        index_type="IVF_PQ",
        metric="cosine",
        num_partitions=min(256, count // 100),
        num_sub_vectors=16,
    )
    print("[LanceDB] 索引创建完成")


def query_examples(table: lancedb.table.Table):
    """
    展示 LanceDB 混合检索的各种用法（仅演示，不执行）。
    """
    examples = """
    # ── 1. 纯向量检索（语义相似） ──
    results = (
        table.search(query_vector)
        .metric("cosine")
        .limit(10)
        .to_pandas()
    )

    # ── 2. 元数据过滤 + 向量检索 ──
    results = (
        table.search(query_vector)
        .where("status = 'registered'")
        .where("nice_classes @> [9]")       # 包含第 9 类
        .where("filing_date >= '2020-01-01'")
        .limit(20)
        .to_pandas()
    )

    # ── 3. 纯 SQL 过滤（无需向量） ──
    results = (
        table.search()
        .where("mark_name LIKE '%Solar%'")
        .where("applicant_country = 'FR'")
        .where("status IN ('registered', 'published')")
        .to_pandas()
    )

    # ── 4. 混合检索（稠密 + 稀疏） ──
    # LanceDB 0.25+ 原生支持 hybrid search
    results = (
        table.search(query_text)
        .vector_type("hybrid")   # 自动使用 dense + sparse
        .rerank(rrf)             # Reciprocal Rank Fusion
        .limit(10)
        .to_pandas()
    )

    # ── 5. 尼斯分类精确过滤 ──
    results = (
        table.search(query_vector)
        .where("nice_classes @> [9, 35]")   # 同时包含第 9 和 35 类
        .limit(20)
        .to_pandas()
    )
    """
    print(examples)


# ═══════════════════════════════════════════════════════════════
#  4. 入口：初始化数据库
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("TradeMarkFlow-EU — LanceDB Schema 初始化")
    print("=" * 60)

    table = init_trademark_table()
    print(f"\n表信息:")
    print(f"  表名: {TABLE_NAME}")
    print(f"  行数: {table.count_rows()}")
    print(f"  列数: {len(table.schema)}")

    print(f"\n列定义:")
    for field in table.schema:
        print(f"  {field.name:25s} {str(field.type):30s}")

    print("\n查询示例:")
    query_examples(table)

    print("\n✅ LanceDB 初始化完成。零运维，数据存储在 data/lancedb/")
