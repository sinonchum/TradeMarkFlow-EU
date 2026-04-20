"""
TradeMarkFlow-EU — Marker PDF 解析 Pipeline

使用 Marker 将 EUIPO 审查意见 PDF 高保真转换为 Markdown，
供 OA_Response_Agent 进行异议分析和答辩起草。

Marker 的核心优势：
- 保留表格、列表、标题的结构
- 支持多语言（EUIPO 审查意见可能是 DE/FR/EN 等）
- 比 PyPDF/pdfplumber 提取质量高 2-3 倍
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def parse_oa_pdf(pdf_path: str, output_dir: str = "data/pdfs") -> Optional[str]:
    """
    使用 Marker 将 EUIPO 审查意见 PDF 转换为 Markdown。

    Marker 处理流程：
    1. PDF → 图像序列
    2. 图像 → 文本检测（OCR）
    3. 文本 + 布局 → 结构化 Markdown
    4. 表格识别 → Markdown 表格

    Args:
        pdf_path: EUIPO 审查意见 PDF 文件路径
        output_dir: 输出目录（同时保存 .md 文件）

    Returns:
        Markdown 文本，失败返回 None
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"[Marker] 文件不存在: {pdf_path}")
        return None

    try:
        from marker.convert import convert_single_pdf
        from marker.models import load_all_models

        print(f"[Marker] 解析 PDF: {pdf_path.name} ({pdf_path.stat().st_size / 1024:.1f} KB)")

        # 加载模型（首次较慢，后续有缓存）
        model_lst = load_all_models()

        # 转换 PDF → Markdown
        full_text, images, metadata = convert_single_pdf(
            str(pdf_path),
            model_lst,
            max_pages=50,  # EUIPO OA 通常 < 20 页
        )

        # 保存 Markdown 文件
        output_path = Path(output_dir) / f"{pdf_path.stem}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(full_text, encoding="utf-8")

        print(f"[Marker] 转换完成: {output_path}")
        print(f"[Marker] Markdown 长度: {len(full_text)} 字符")
        print(f"[Marker] 提取图像: {len(images)} 张")

        return full_text

    except ImportError:
        print("[Marker] 未安装，使用 fallback 解析")
        return _fallback_pdf_parse(pdf_path)
    except Exception as e:
        print(f"[Marker] 转换错误: {e}")
        return _fallback_pdf_parse(pdf_path)


def _fallback_pdf_parse(pdf_path: Path) -> Optional[str]:
    """
    Fallback: 使用 PyPDF2/fitz 解析 PDF（质量不如 Marker）。
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        text_parts = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                text_parts.append(f"## Page {page_num + 1}\n\n{text}")

        full_text = "\n\n".join(text_parts)

        # 保存
        output_path = Path("data/pdfs") / f"{pdf_path.stem}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(full_text, encoding="utf-8")

        print(f"[PyMuPDF Fallback] 解析完成: {len(full_text)} 字符")
        return full_text

    except ImportError:
        # 最终 fallback：使用 pypdf
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf_path))
            text_parts = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_parts.append(f"## Page {i + 1}\n\n{text}")

            full_text = "\n\n".join(text_parts)

            output_path = Path("data/pdfs") / f"{pdf_path.stem}.md"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(full_text, encoding="utf-8")

            print(f"[pypdf Fallback] 解析完成: {len(full_text)} 字符")
            return full_text

        except Exception as e:
            print(f"[Fallback] 所有 PDF 解析器失败: {e}")
            return None


def parse_oa_text(text: str, output_path: str = "data/pdfs/latest_oa.md") -> str:
    """
    直接处理纯文本格式的审查意见（非 PDF 场景）。
    保存为 Markdown 供 Agent 读取。
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 简单的文本 → Markdown 格式化
    lines = text.split("\n")
    formatted = []
    for line in lines:
        line = line.strip()
        if not line:
            formatted.append("")
        elif line.isupper() and len(line) < 100:
            formatted.append(f"## {line.title()}")
        elif line.startswith(("Article", "Art.", "Section")):
            formatted.append(f"**{line}**")
        else:
            formatted.append(line)

    md_text = "\n".join(formatted)
    output.write_text(md_text, encoding="utf-8")

    print(f"[Text→MD] 已保存: {output}")
    return md_text


# ═══════════════════════════════════════════════════════════════
#  LightRAG 法律知识库初始化（集成）
# ═══════════════════════════════════════════════════════════════

async def init_lightrag_corpus(corpus_dir: str = "legal_corpus") -> Optional[object]:
    """
    初始化 LightRAG 实例，加载 EUTMR 法律语料。

    LightRAG 的核心优势：
    - 图谱驱动的 RAG：实体关系比纯向量检索更准确
    - 适合法律文本：条文之间的交叉引用形成自然的知识图谱
    - 支持增量更新：新指南发布时只需追加

    EUTMR 语料目录结构：
    legal_corpus/
    ├── eutmr_2017_1001.txt     # 条例全文
    ├── guidelines_part_b.txt   # 审查指南 Part B
    ├── guidelines_part_c.txt   # 异议指南 Part C
    └── board_decisions/        # Board of Appeal 裁决摘要
        ├── r_1234_2020.txt
        └── ...
    """
    try:
        from lightrag import LightRAG, QueryParam
        from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

        working_dir = "data/lightrag"
        os.makedirs(working_dir, exist_ok=True)

        rag = LightRAG(
            working_dir=working_dir,
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete,
        )

        # 加载语料文件
        corpus_path = Path(corpus_dir)
        if corpus_path.exists():
            for txt_file in corpus_path.glob("*.txt"):
                content = txt_file.read_text(encoding="utf-8")
                if content.strip():
                    print(f"[LightRAG] 正在索引: {txt_file.name} ({len(content)} chars)")
                    await rag.ainsert(content)

            # 子目录（如 Board of Appeal 裁决）
            for subdir in corpus_path.iterdir():
                if subdir.is_dir():
                    for txt_file in subdir.glob("*.txt"):
                        content = txt_file.read_text(encoding="utf-8")
                        if content.strip():
                            await rag.ainsert(content)
                            print(f"[LightRAG] 正在索引: {subdir.name}/{txt_file.name}")

        print(f"[LightRAG] 知识库初始化完成，存储在 {working_dir}")
        return rag

    except ImportError:
        print("[LightRAG] 未安装，知识库功能不可用")
        print("  安装: pip install lightrag-hku")
        return None


async def query_lightrag(rag: object, query: str, mode: str = "hybrid") -> str:
    """
    查询 LightRAG 法律知识库。

    模式：
    - "naive": 纯向量检索
    - "local": 基于实体的局部图谱检索
    - "global": 全局图谱检索
    - "hybrid": 混合（推荐 — 结合局部实体 + 全局关系）
    """
    try:
        from lightrag import QueryParam

        result = await rag.aquery(
            query,
            param=QueryParam(mode=mode),
        )
        return result

    except Exception as e:
        return f"[LightRAG] 查询失败: {e}"


# ═══════════════════════════════════════════════════════════════
#  Demo
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("TradeMarkFlow-EU — OA 解析 Demo")
    print("=" * 60)

    # 模拟 EUIPO 审查意见文本
    sample_oa = """
EUROPEAN UNION INTELLECTUAL PROPERTY OFFICE

EXAMINATION REPORT

Application Number: 019123456
Trade Mark: SolarFlux
Applicant: EnergiePlus SAS
Filing Date: 20/04/2026

EXAMINER'S OBSERVATIONS

Article 7(1)(b) EUTMR — Lack of Distinctive Character

The mark "SolarFlux" is composed of the words "Solar" and "Flux".
"Solar" directly describes the nature of the goods (Class 9: solar panels).
"Flux" is a common technical term in energy systems.

The combination does not create a sufficiently distinctive whole.
The average consumer would perceive it as a descriptive indication
rather than an indicator of commercial origin.

Article 7(1)(c) EUTMR — Descriptive Signs

The mark is also objectionable under Article 7(1)(c) as it describes
a characteristic of the goods/services, namely their solar energy nature.

DEADLINE FOR RESPONSE: 20/07/2026

Signed,
EUIPO Examiner
"""

    # 保存为 Markdown
    md_text = parse_oa_text(sample_oa)
    print(f"\n[Demo] 解析后 Markdown 长度: {len(md_text)} 字符")
    print(f"\n[Demo] 前 500 字符:\n{md_text[:500]}...")

    # 保存一份示例 EUTMR 片段供 LightRAG 测试
    os.makedirs("legal_corpus", exist_ok=True)
    sample_eutmr = """
Article 7 — Absolute grounds for refusal

1. The following shall not be registered:
(a) signs which do not conform to Article 4;
(b) trade marks which are devoid of any distinctive character;
(c) trade marks which consist exclusively of signs or indications
    which may serve, in trade, to designate the kind, quality,
    quantity, intended purpose, value, geographical origin,
    or the time of production of the goods or of rendering of
    the service, or other characteristics of the goods or service;
(d) trade marks which consist exclusively of signs or indications
    which have become customary in the current language or in the
    bona fide and established practices of the trade;

3. Paragraph 1(b), (c) and (d) shall not apply if the trade mark
has become distinctive in relation to the goods or services for
which registration is requested in consequence of the use which
has been made of it.
"""
    Path("legal_corpus/eutmr_sample.txt").write_text(sample_eutmr, encoding="utf-8")
    print(f"\n[Demo] 示例 EUTMR 语料已保存到 legal_corpus/eutmr_sample.txt")

    print("\n✅ OA 解析 Demo 完成")
