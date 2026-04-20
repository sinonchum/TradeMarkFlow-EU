"""
TradeMarkFlow-EU — FastAPI 入口

轻量级 REST API，为前端/CLI 提供服务。
核心 Agent 逻辑由 Agno 处理，FastAPI 只做路由和编排。
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.models.schema import (
    TrademarkRecord,
    init_trademark_table,
    upsert_trademarks,
    get_db_connection,
    DB_PATH,
)


# ─── Request/Response Models ────────────────────────────────────

class ClearanceRequest(BaseModel):
    mark_name: str
    business_description: str
    use_local_llm: bool = False

class ClearanceResponse(BaseModel):
    mark_name: str
    report: str
    similar_marks: list[dict] = []
    risk_level: str = "unknown"

class OARequest(BaseModel):
    application_number: str = ""
    mark_name: str = ""
    oa_markdown: str  # Marker 转换后的审查意见

class OAResponse(BaseModel):
    application_number: str
    response_draft: str
    objections_found: list[str] = []
    citations: list[str] = []

class IngestRequest(BaseModel):
    urls: list[str]

class IngestResponse(BaseModel):
    total: int
    success: int
    failed: int


# ─── Lifecycle ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化 LanceDB 表
    table = init_trademark_table()
    print(f"[API] LanceDB 表已就绪: {table.count_rows()} 行")
    yield
    print("[API] Shutdown")


# ─── App ────────────────────────────────────────────────────────

app = FastAPI(
    title="TradeMarkFlow-EU",
    description="Privacy-first, serverless EUTM clearance & filing assistant",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ──────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "TradeMarkFlow-EU",
        "stack": "Agno + LanceDB + Crawl4AI + Scrapling + Marker + LightRAG",
    }


@app.post("/api/v1/clearance", response_model=ClearanceResponse)
async def clearance_search(req: ClearanceRequest):
    """
    商标清查接口 — 调用 Clearance Agent 执行完整检索流程。

    流程：
    1. LanceDB 混合检索相似商标
    2. 尼斯分类推荐
    3. 音似/形似/义似三维分析
    4. 绝对理由风险评估
    5. 输出结构化报告
    """
    try:
        from app.agents.agents import run_clearance
        report = run_clearance(
            mark_name=req.mark_name,
            business_description=req.business_description,
            use_local=req.use_local_llm,
        )
        return ClearanceResponse(
            mark_name=req.mark_name,
            report=report,
            risk_level="pending_analysis",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/oa/respond", response_model=OAResponse)
async def oa_respond(req: OARequest):
    """
    审查意见处理接口 — 调用 OA Response Agent 生成答辩草稿。

    流程：
    1. 接收 Marker 解析的 Markdown 审查意见
    2. Agent 识别每项异议（Art. 7 / Art. 8）
    3. 检索 LightRAG 法律知识库
    4. 起草专业法律答辩
    """
    try:
        from app.agents.agents import run_oa_response
        draft = run_oa_response(
            oa_markdown=req.oa_markdown,
            mark_name=req.mark_name,
            application_number=req.application_number,
        )
        return OAResponse(
            application_number=req.application_number,
            response_draft=draft,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest_trademarks(req: IngestRequest):
    """
    数据采集接口 — 批量抓取 EUIPO 商标详情页并入库。
    """
    try:
        from app.pipeline.ingest_euipo import ingest_batch
        success_count = await ingest_batch(req.urls)
        return IngestResponse(
            total=len(req.urls),
            success=success_count,
            failed=len(req.urls) - success_count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/trademarks/stats")
async def trademark_stats():
    """数据库统计"""
    try:
        from app.tools.lancedb_tools import get_database_stats
        return json.loads(get_database_stats())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/trademarks/search")
async def search_trademarks(
    query: str,
    nice_classes: list[int] | None = None,
    status: str = "",
    limit: int = 20,
):
    """商标检索接口"""
    try:
        from app.tools.lancedb_tools import hybrid_search_trademarks
        result = hybrid_search_trademarks(
            query_text=query,
            nice_classes=nice_classes or [],
            status_filter=status,
            limit=limit,
        )
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Dev entrypoint ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
