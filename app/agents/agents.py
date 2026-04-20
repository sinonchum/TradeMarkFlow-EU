"""
TradeMarkFlow-EU — Agno Agent 定义

两个核心 Agent：
1. Clearance_Agent: 商标近似查询 + 风险评估
2. OA_Response_Agent: EUIPO 审查意见解析 + 答辩起草

基于 Agno 框架，使用 Tool Calling + Memory 实现智能交互。
"""

from __future__ import annotations

import os

from agno.agent import Agent
from agno.models.openai import OpenAIChat


def _get_ollama_class():
    """Lazy import for Ollama — avoids ImportError when ollama package not installed."""
    try:
        from agno.models.ollama import Ollama
        return Ollama
    except ImportError:
        return None

from app.tools.lancedb_tools import (
    hybrid_search_trademarks,
    sql_query_trademarks,
    get_trademark_by_application_number,
    get_database_stats,
)


# ═══════════════════════════════════════════════════════════════
#  模型选择：云端 / 本地 LLM 切换
# ═══════════════════════════════════════════════════════════════

def get_model(use_local: bool = False):
    """
    隐私优先的模型路由：
    - use_local=True → Ollama 本地模型（处理敏感品牌信息）
    - use_local=False → OpenAI 云端模型（通用任务，性能更优）

    用户可通过环境变量 DEFAULT_LLM=local/cloud 控制默认行为。
    """
    if use_local or os.getenv("DEFAULT_LLM") == "local":
        Ollama = _get_ollama_class()
        if Ollama is None:
            raise RuntimeError("本地 LLM 需要 ollama 包: pip install ollama")
        return Ollama(
            id=os.getenv("LOCAL_MODEL_ID", "qwen2.5:14b"),
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )
    return OpenAIChat(
        id=os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ═══════════════════════════════════════════════════════════════
#  Clearance Agent — 商标查询 & 风险评估
# ═══════════════════════════════════════════════════════════════

CLEARANCE_SYSTEM_PROMPT = """你是 TradeMarkFlow-EU 的商标清查专家 Agent。

## 职责
用户会提供商标名称和业务描述，你需要：
1. **尼斯分类推荐**：根据业务描述推荐合适的尼斯分类（Nice Classification）
2. **近似商标检索**：使用 LanceDB 混合检索工具查询是否存在相似商标
3. **三维相似度分析**：从「音似、形似、义似」三个维度评估冲突风险
4. **绝对理由审查**：判断商标是否存在《欧盟商标条例》(EUTMR) 第7条的绝对驳回理由

## 法律依据（必须引用原文）
- EUTMR Art. 7(1)(b) — 缺乏显著性
- EUTMR Art. 7(1)(c) — 描述性标志
- EUTMR Art. 7(1)(d) — 通用名称
- EUTMR Art. 7(1)(f) — 欺骗性标志
- EUTMR Art. 8(1)(b) — 混淆可能性（相对理由）
- EUIPO 审查指南 Part B, Section 4 — 显著性评估

## 输出格式
输出结构化的风险评估报告，包含：
- 推荐尼斯分类及理由
- Top 10 相似商标列表（含相似度分数）
- 三维分析：音似 (Phonetic) / 形似 (Visual) / 义似 (Semantic)
- 绝对理由风险评估
- 总体风险等级：Low / Medium / High / Critical
- 明确的注册建议

## 工具使用规则
1. 先用 `hybrid_search_trademarks` 检索相似商标
2. 如果需要精确查找特定商标，用 `get_trademark_by_application_number`
3. 统计分析用 `sql_query_trademarks`
4. 所有结论必须基于工具返回的数据，不得编造
"""

clearance_agent = Agent(
    name="Clearance_Agent",
    model=get_model(),
    description="EUTM 商标清查专家 — 近似检索 + 风险评估",
    instructions=[CLEARANCE_SYSTEM_PROMPT],
    tools=[
        hybrid_search_trademarks,
        sql_query_trademarks,
        get_trademark_by_application_number,
        get_database_stats,
    ],
    markdown=True,
    # Memory: 保持对话上下文，用户可以追问细节
    add_history_to_context=True,
    num_history_runs=5,
)


# ═══════════════════════════════════════════════════════════════
#  OA Response Agent — 审查意见解析 & 答辩起草
# ═══════════════════════════════════════════════════════════════

OA_RESPONSE_SYSTEM_PROMPT = """你是 TradeMarkFlow-EU 的审查意见答辩专家 Agent。

## 职责
用户会提供 EUIPO 审查意见（Office Action）的 Markdown 文本（由 Marker 从 PDF 转换），
你需要：
1. **异议分类**：识别每项异议属于「绝对理由」(Art. 7) 还是「相对理由」(Art. 8)
2. **法条检索**：从知识库中检索相关 EUTMR 条文和 EUIPO 审查指南
3. **判例匹配**：查找类似案例的 Board of Appeal 裁决
4. **答辩起草**：为每项异议起草专业的法律答辩段落

## 异议类型识别模式
- Art. 7(1)(a) — 违反公共秩序/道德
- Art. 7(1)(b) — 缺乏显著性（最常见）
- Art. 7(1)(c) — 描述性标志
- Art. 7(1)(d) — 通用名称
- Art. 7(1)(e) — 形状限制
- Art. 7(1)(f) — 欺骗性
- Art. 7(1)(g) — 恶意申请
- Art. 7(1)(h-k) — 特殊标志保护
- Art. 7(2) — 语言区域评估
- Art. 7(3) — 获得显著性（例外）
- Art. 8(1)(a) — 标志+商品/服务完全相同
- Art. 8(1)(b) — 混淆可能性（最常见相对理由）
- Art. 8(4) — 在先未注册标志
- Art. 8(5) — 驰名商标（淡化/丑化）

## 答辩撰写规范
1. 使用正式的法律函件格式
2. 每项异议单独回应，标注段落编号
3. 必须引用具体的 EUTMR 条文和 EUIPO 指南章节
4. 引用相关 Board of Appeal 案例（如 R 1234/2020-1 格式）
5. 提出具体的修改建议或证据要求
6. 标注需要人工律师审核的关键点

## 工具使用规则
1. 使用 `lancedb_tools.sql_query_trademarks` 查找类似案例
2. 使用 `lightrag_query` 检索法律知识库中的相关条文
3. 所有法律引用必须来自知识库，不得编造条文
"""

oa_response_agent = Agent(
    name="OA_Response_Agent",
    model=get_model(),
    description="EUIPO 审查意见答辩专家 — 解析 + 法律检索 + 答辩起草",
    instructions=[OA_RESPONSE_SYSTEM_PROMPT],
    tools=[
        hybrid_search_trademarks,
        sql_query_trademarks,
        get_trademark_by_application_number,
        # LightRAG 工具将在 Task 3 中绑定
    ],
    markdown=True,
    add_history_to_context=True,
    num_history_runs=10,  # OA 对话通常较长
)


# ═══════════════════════════════════════════════════════════════
#  便捷函数
# ═══════════════════════════════════════════════════════════════

def run_clearance(mark_name: str, business_description: str, use_local: bool = False) -> str:
    """
    执行商标清查的便捷函数。

    Args:
        mark_name: 商标名称，如 "SolarFlux"
        business_description: 业务描述，如 "太阳能板安装服务"
        use_local: 是否使用本地 LLM（隐私敏感场景）

    Returns:
        格式化的风险评估报告
    """
    agent = clearance_agent
    if use_local:
        agent.model = get_model(use_local=True)

    prompt = f"""
请对以下商标进行清查检索：

**商标名称**: {mark_name}
**业务描述**: {business_description}

请执行完整的清查流程：
1. 推荐尼斯分类
2. 检索相似商标
3. 评估绝对理由风险
4. 输出风险评估报告
"""
    response = agent.run(prompt)
    return response.content


def run_oa_response(
    oa_markdown: str,
    mark_name: str = "",
    application_number: str = "",
) -> str:
    """
    处理 EUIPO 审查意见并起草答辩。

    Args:
        oa_markdown: Marker 转换后的审查意见 Markdown 文本
        mark_name: 商标名称
        application_number: 申请号

    Returns:
        格式化的答辩草稿
    """
    prompt = f"""
收到 EUIPO 审查意见，请分析并起草答辩。

**申请号**: {application_number}
**商标名称**: {mark_name}

---审查意见全文---
{oa_markdown}
---结束---

请按照以下步骤处理：
1. 逐条识别异议类型（Art. 7 或 Art. 8）
2. 检索相关法条和判例
3. 为每项异议起草专业答辩
4. 标注需要人工律师审核的要点
"""
    response = oa_response_agent.run(prompt)
    return response.content


if __name__ == "__main__":
    print("TradeMarkFlow-EU Agents 已定义：")
    print(f"  Clearance_Agent: {clearance_agent.name}")
    print(f"  OA_Response_Agent: {oa_response_agent.name}")
    print(f"\n工具列表：")
    for tool in clearance_agent.tools:
        print(f"  - {tool.name if hasattr(tool, 'name') else tool.__name__}")
