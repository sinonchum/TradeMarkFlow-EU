"""
TradeMarkFlow-EU — EUIPO 数据抓取 Pipeline

使用 Crawl4AI 提取 EUIPO 商标详情页数据，
结构化后存入 LanceDB。

同时展示 Scrapling 在高频/反爬场景下的用法（EUIPO 公告监控）。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.models.schema import TrademarkRecord, upsert_trademarks


# ═══════════════════════════════════════════════════════════════
#  1. Crawl4AI — 网页 → 结构化数据
# ═══════════════════════════════════════════════════════════════

async def crawl_euipo_detail_page(url: str) -> Optional[TrademarkRecord]:
    """
    使用 Crawl4AI 抓取 EUIPO 商标详情页并提取结构化数据。

    Crawl4AI 的核心优势：
    - 任何网页 → 纯净 Markdown（自动去除广告、导航、JS 残留）
    - 基于 LLM 的智能内容提取（不需要手写 CSS 选择器）
    - 异步并发，适合批量抓取

    Args:
        url: EUIPO 商标详情页 URL
              如 https://euipo.europa.eu/eSearch/#details/trademarks/019123456

    Returns:
        结构化的 TrademarkRecord，失败返回 None
    """
    try:
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode

        config = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED,  # 缓存避免重复抓取
            wait_until="networkidle",      # 等待 JS 加载完成
            page_timeout=30000,            # 30 秒超时
        )

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, config=config)

            if not result.success:
                print(f"[Crawl4AI] 抓取失败: {url} — {result.error_message}")
                return None

            # Crawl4AI 输出纯净 Markdown
            markdown = result.markdown

            # 使用 LLM 提取结构化字段（Crawl4AI 内置提取能力）
            # 也可以用正则/关键词匹配提取，更确定性
            record = _parse_euipo_markdown(markdown, url)

            if record:
                print(f"[Crawl4AI] 成功提取: {record.mark_name} ({record.application_number})")
            return record

    except ImportError:
        print("[Crawl4AI] 未安装，使用模拟数据演示")
        return _mock_trademark_record(url)
    except Exception as e:
        print(f"[Crawl4AI] 错误: {e}")
        return None


def _parse_euipo_markdown(markdown: str, source_url: str) -> Optional[TrademarkRecord]:
    """
    从 Crawl4AI 输出的 Markdown 中解析 EUIPO 商标字段。

    EUIPO 详情页的 Markdown 结构大致为：
    # Trade mark details
    **Mark name:** SolarFlux
    **Application number:** 019123456
    **Filing date:** 20/04/2026
    **Nice classes:** 9, 35, 42
    ...
    """
    import re

    def extract(pattern: str, text: str, default: str = "") -> str:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip() if match else default

    def extract_list(pattern: str, text: str) -> list[str]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        return [m.strip() for m in matches if m.strip()]

    try:
        mark_name = extract(r"(?:Mark name|Trade mark|Mark):\s*(.+)", markdown)
        app_number = extract(r"(?:Application number|App\.?\s*No\.?):\s*(\d+)", markdown)
        reg_number = extract(r"(?:Registration number|Reg\.?\s*No\.?):\s*(\d+)", markdown)
        filing_date = extract(r"Filing date:\s*(\d{2}/\d{2}/\d{4})", markdown)
        reg_date = extract(r"Registration date:\s*(\d{2}/\d{2}/\d{4})", markdown)

        # Nice classes — 可能是 "9, 35, 42" 或 "Class 9: Computers; Class 35: Advertising"
        nice_raw = extract(r"Nice classes?:?\s*(.+)", markdown)
        nice_classes = [int(x) for x in re.findall(r"\d+", nice_raw) if 1 <= int(x) <= 45]

        # 申请人
        applicant = extract(r"(?:Applicant|Holder|Proprietor):\s*(.+)", markdown)
        country = extract(r"(?:Country|Origin):\s*([A-Z]{2})", markdown)

        # 状态
        status_raw = extract(r"Status:\s*(.+)", markdown)
        status = _normalize_status(status_raw)

        # Goods/Services
        goods_raw = extract(r"(?:Goods|Services|List of goods):\s*(.+)", markdown)
        goods_services = [g.strip() for g in goods_raw.split(";") if g.strip()] if goods_raw else []

        if not mark_name or not app_number:
            print(f"[Parser] 关键字段缺失: mark_name={mark_name}, app_number={app_number}")
            return None

        # 日期格式转换: DD/MM/YYYY → YYYY-MM-DD
        def convert_date(d: str) -> Optional[str]:
            if not d:
                return None
            try:
                parts = d.replace("-", "/").split("/")
                if len(parts) == 3:
                    return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
            except Exception:
                pass
            return None

        return TrademarkRecord(
            id=str(uuid.uuid4()),
            application_number=app_number,
            registration_number=reg_number or None,
            mark_name=mark_name,
            mark_type="word",  # 默认，可从页面图形信息扩展
            nice_classes=nice_classes,
            goods_services=goods_services,
            applicant_name=applicant,
            applicant_country=country,
            filing_date=convert_date(filing_date),
            registration_date=convert_date(reg_date),
            status=status,
            jurisdiction="EU",
            source="euipo",
            text_vector=[],   # 后续由嵌入模型填充
            image_vector=[],
            raw_url=source_url,
            scraped_at=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        print(f"[Parser] 解析错误: {e}")
        return None


def _normalize_status(raw: str) -> str:
    """将 EUIPO 原始状态映射到标准枚举"""
    raw_lower = raw.lower().strip()
    mapping = {
        "registered": "registered",
        "application": "application",
        "published": "published",
        "opposed": "opposed",
        "refused": "refused",
        "withdrawn": "withdrawn",
        "expired": "expired",
        "surrendered": "surrendered",
        "registration": "registered",
        "filed": "application",
    }
    for key, val in mapping.items():
        if key in raw_lower:
            return val
    return "application"


def _mock_trademark_record(url: str) -> TrademarkRecord:
    """模拟数据（Crawl4AI 不可用时的 fallback）"""
    return TrademarkRecord(
        id=str(uuid.uuid4()),
        application_number="019123456",
        mark_name="SolarFlux",
        mark_type="word",
        nice_classes=[9, 37],
        goods_services=["Solar panels", "Installation of solar panels"],
        applicant_name="EnergiePlus SAS",
        applicant_country="FR",
        filing_date="2026-04-20",
        status="application",
        jurisdiction="EU",
        source="euipo",
        text_vector=[],
        image_vector=[],
        raw_url=url,
        scraped_at=datetime.utcnow().isoformat(),
    )


# ═══════════════════════════════════════════════════════════════
#  2. Scrapling — 高频反爬监控（EUIPO 公告）
# ═══════════════════════════════════════════════════════════════

async def monitor_euipo_bulletin(max_pages: int = 3) -> list[dict]:
    """
    使用 Scrapling 监控 EUIPO 商标公报（Weekly Bulletin）。

    Scrapling 的核心优势：
    - 自适应解析器：页面改版后自动重定位元素
    - 内置反爬绕过（Cloudflare Turnstile）
    - Spider 框架支持并发 + 断点续爬

    EUIPO 公告页面结构会定期变化，Scrapling 的 adaptive 特性
    可以在页面改版后自动调整选择器，减少维护成本。
    """
    try:
        from scrapling.fetchers import Fetcher, StealthyFetcher

        bulletin_url = "https://euipo.europa.eu/eSearch/#bulletins/Trade%20Marks"

        # 普通请求尝试
        page = Fetcher.get(bulletin_url, stealthy_headers=True)

        if not page or len(page.text) < 1000:
            # 被反爬拦截，切换 Stealthy 模式
            print("[Scrapling] 普通请求被拦截，切换 Stealthy 模式")
            page = StealthyFetcher.fetch(
                bulletin_url,
                headless=True,
                solve_cloudflare=True,
            )

        # 提取最新公告链接
        bulletins = []
        links = page.css('a[href*="bulletin"]') or page.css('.bulletin-item a')

        for link in links[:max_pages * 20]:
            title = link.text.strip()
            href = link.attrib.get("href", "")
            if title and href:
                bulletins.append({
                    "title": title,
                    "url": href if href.startswith("http") else f"https://euipo.europa.eu{href}",
                    "scraped_at": datetime.utcnow().isoformat(),
                })

        print(f"[Scrapling] 提取到 {len(bulletins)} 条公告")
        return bulletins

    except ImportError:
        print("[Scrapling] 未安装，返回模拟公告数据")
        return [
            {
                "title": "Weekly Bulletin 2026/16 — New Applications",
                "url": "https://euipo.europa.eu/eSearch/#bulletins/2026/16",
                "scraped_at": datetime.utcnow().isoformat(),
            }
        ]


# ═══════════════════════════════════════════════════════════════
#  3. 批量 Ingest Pipeline
# ═══════════════════════════════════════════════════════════════

async def ingest_batch(urls: list[str], batch_size: int = 10) -> int:
    """
    批量抓取 EUIPO 商标详情页并存入 LanceDB。

    Args:
        urls: EUIPO 商标详情页 URL 列表
        batch_size: 并发批次大小

    Returns:
        成功入库的记录数
    """
    from app.models.schema import upsert_trademarks

    records: list[TrademarkRecord] = []
    semaphore = asyncio.Semaphore(batch_size)

    async def fetch_one(url: str):
        async with semaphore:
            record = await crawl_euipo_detail_page(url)
            if record:
                records.append(record)

    tasks = [fetch_one(url) for url in urls]
    await asyncio.gather(*tasks, return_exceptions=True)

    if records:
        count = upsert_trademarks(records)
        print(f"[Pipeline] 成功入库 {count} 条记录")
        return count
    return 0


# ═══════════════════════════════════════════════════════════════
#  4. Demo 入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("=" * 60)
        print("TradeMarkFlow-EU — 数据抓取 Demo")
        print("=" * 60)

        # 1. 模拟抓取单个商标详情页
        print("\n[1] Crawl4AI → 结构化数据 → LanceDB")
        record = await crawl_euipo_detail_page(
            "https://euipo.europa.eu/eSearch/#details/trademarks/019123456"
        )
        if record:
            print(f"  提取: {record.mark_name}")
            print(f"  申请号: {record.application_number}")
            print(f"  尼斯分类: {record.nice_classes}")
            from app.models.schema import upsert_trademarks
            upsert_trademarks([record])

        # 2. Scrapling 公告监控
        print("\n[2] Scrapling → EUIPO 公告监控")
        bulletins = await monitor_euipo_bulletin(max_pages=1)
        for b in bulletins[:3]:
            print(f"  • {b['title']}")

        print("\n✅ Demo 完成")

    asyncio.run(demo())
