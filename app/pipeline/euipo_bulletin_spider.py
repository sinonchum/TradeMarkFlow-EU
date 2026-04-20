"""
TradeMarkFlow-EU — EUIPO Bulletin Spider (Scrapling)

监控 EUIPO 每周商标公报（Weekly Bulletin），提取新申请信息。

Spider 特性：
- 自适应解析：页面改版后自动调整选择器
- 断点续爬：支持暂停/恢复
- 反爬绕过：内置 Cloudflare Turnstile 解决（stealthy 模式）
- 数据导出：JSON/CSV 格式，可直接入库 LanceDB

使用方法：
    python -m app.pipeline.euipo_bulletin_spider
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import AsyncGenerator

from scrapling.spiders import Spider, Response


class EUIPOBulletinSpider(Spider):
    """
    爬取 EUIPO 商标公报页面，提取新申请信息。

    目标页面结构（示例）：
    https://euipo.europa.eu/eSearch/#bulletins/Trade%20Marks

    每个公告条目包含：
    - 标题（含申请号、商标名称）
    - 发布日期
    - 链接到详情页
    """

    name = "euipo_bulletin"
    start_urls = ["https://euipo.europa.eu/eSearch/#bulletins/Trade%20Marks"]

    # 反爬设置
    concurrent_requests = 4
    download_delay = 1.0  # 每请求间隔 1 秒
    robots_txt_obey = True

    # 启用开发模式调试时请设为 True
    development_mode = False

    async def parse(self, response: Response) -> AsyncGenerator[dict, None]:
        """
        解析公报列表页面，提取每条公告的基本信息。
        """
        self.logger.info(f"Parsing bulletin page: {response.url}")

        # 尝试多种可能的选择器（EUIPO 页面结构可能变化）
        selectors = [
            ".bulletin-item",
            ".search-result-item",
            ".result-item",
            "[data-testid='bulletin-item']",
            "li.bulletin-entry",
            "tr.bulletin-row",
        ]

        bulletin_items = None
        for selector in selectors:
            bulletin_items = response.css(selector)
            if bulletin_items:
                self.logger.info(f"Found {len(bulletin_items)} items with selector: {selector}")
                break

        if not bulletin_items:
            #  fallback: 找所有链接包含 "bulletin" 或 "application" 的
            self.logger.warning("No items found with standard selectors, trying fallback")
            bulletin_items = response.css('a[href*="bulletin"], a[href*="application"]')
            if not bulletin_items:
                self.logger.error("Could not find any bulletin items on the page")
                return

        for item in bulletin_items:
            # 提取标题和链接
            title_el = item.css("a").first or item
            title = title_el.css("::text").get("").strip()
            url = title_el.css("::attr(href)").get("")

            # 处理相对 URL
            if url and not url.startswith("http"):
                url = f"https://euipo.europa.eu{url}"

            # 尝试提取申请号和商标名（常见模式）
            application_number = None
            mark_name = None

            # 模式 1: "Application No. 019123456 - SolarFlux"
            import re
            app_match = re.search(r"(?:Application\s*No\.?|App\.?\s*No\.?)\s*:?\s*(\d{6,12})", title, re.IGNORECASE)
            if app_match:
                application_number = app_match.group(1)
                # 提取申请号后的内容作为商标名
                mark_part = title[app_match.end():].strip(" -:")
                if mark_part:
                    mark_name = mark_part

            # 模式 2: "019123456 SolarFlux"
            if not application_number:
                num_match = re.search(r"(\d{6,12})\s+([^\d]+)", title)
                if num_match:
                    application_number = num_match.group(1)
                    mark_name = num_match.group(2).strip()

            # 模式 3: 仅从链接中提取申请号
            if not application_number:
                url_match = re.search(r"/trademarks/(\d{6,12})", url)
                if url_match:
                    application_number = url_match.group(1)

            # 如果还是没有标题，使用链接文本或占位符
            if not mark_name:
                mark_name = title or "Unknown Mark"

            yield {
                "id": str(uuid.uuid4()),
                "title": title,
                "application_number": application_number,
                "mark_name": mark_name,
                "url": url,
                "source": "euipo_bulletin",
                "scraped_at": datetime.utcnow().isoformat(),
                "bulletin_date": self._extract_date(item),
            }

    def _extract_date(self, item: Response) -> str | None:
        """尝试从公告项中提取日期。"""
        date_selectors = [
            ".date",
            ".published-date",
            ".bulletin-date",
            "time",
            "[data-date]",
        ]
        for selector in date_selectors:
            date_el = item.css(selector).first
            if date_el:
                # 尝试属性
                date_attr = date_el.css("::attr(datetime)").get() or date_el.css("::attr(data-date)").get()
                if date_attr:
                    return date_attr
                # 尝试文本
                date_text = date_el.css("::text").get("").strip()
                if date_text and len(date_text) < 30:
                    return date_text
        return None

    async def on_start(self, resuming: bool = False):
        if resuming:
            self.logger.info("Resuming EUIPO bulletin crawl from checkpoint")
        else:
            self.logger.info("Starting fresh EUIPO bulletin crawl")

    async def on_close(self):
        self.logger.info("EUIPO bulletin crawl finished")


def run_bulletin_spider(max_pages: int = 5) -> dict:
    """
    运行公报爬虫并返回结果。

    Args:
        max_pages: 最大爬取页数（通过翻页实现）

    Returns:
        包含统计和数据的字典
    """
    spider = EUIPOBulletinSpider()
    # 为了演示，我们只爬取第一页（实际应实现翻页）
    # 由于 EUIPO 的公告页面可能需要 JavaScript 加载，这里简化处理
    result = spider.start()

    items = []
    for item in result.items:
        items.append(item)

    return {
        "spider": spider.name,
        "completed": result.completed,
        "paused": result.paused,
        "items_count": len(items),
        "items": items[:20],  # 限制返回数量
        "stats": result.stats.to_dict(),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("TradeMarkFlow-EU — EUIPO Bulletin Spider")
    print("=" * 60)

    result = run_bulletin_spider(max_pages=3)

    print(f"✅ 爬虫运行完成")
    print(f"   完成状态: {result['completed']}")
    print(f"   暂停状态: {result['paused']}")
    print(f"   提取条目: {result['items_count']}")
    print(f"   请求总数: {result['stats']['requests_count']}")
    print(f"   失败请求: {result['stats']['failed_requests_count']}")

    if result["items"]:
        print(f"\n📋 前 {min(5, len(result['items']))} 条公告:")
        for i, item in enumerate(result["items"][:5]):
            print(f"  {i+1:2d}. {item.get('mark_name', 'N/A'):30s} "
                  f"App: {item.get('application_number', 'N/A'):12s} "
                  f"URL: {item.get('url', 'N/A')[:60]}...")

    print(f"\n💾 建议的下一步：")
    print(f"   1. 将提取的 application_number 用于详情页抓取")
    print(f"   2. 调用 Crawl4AI 获取完整商标数据")
    print(f"   3. 生成嵌入向量并存入 LanceDB")
    print(f"   4. 触发清查检索流程")
