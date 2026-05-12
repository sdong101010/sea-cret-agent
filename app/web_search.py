"""Search the web and fetch page content for answer research."""

import asyncio
import logging
import re
from functools import partial

import httpx
from ddgs import DDGS

logger = logging.getLogger(__name__)

SALESFORCE_SITES = (
    "site:help.salesforce.com"
    " OR site:developer.salesforce.com"
    " OR site:trailhead.salesforce.com"
    " OR site:trust.salesforce.com"
    " OR site:architect.salesforce.com"
    " OR site:salesforce.com/blog"
)


def _search_sync(query: str, max_results: int, scoped: bool = True) -> list[dict]:
    q = f"{query} ({SALESFORCE_SITES})" if scoped else f"{query} Salesforce"
    with DDGS() as ddgs:
        return list(ddgs.text(q, max_results=max_results))


def _extract_text_from_html(html: str) -> str:
    """Strip HTML to readable text, removing nav/scripts/styles."""
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
    html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL)
    html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL)
    html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL)
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


async def fetch_page_content(url: str, max_chars: int = 8000) -> str:
    """Fetch a web page and return extracted text content."""
    try:
        async with httpx.AsyncClient(
            timeout=10.0, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
        ) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return ""
            text = _extract_text_from_html(resp.text)
            return text[:max_chars]
    except Exception:
        logger.debug("Failed to fetch %s", url)
        return ""


async def research_web(query: str, num_results: int = 3) -> tuple[list[dict], str]:
    """Search the web AND fetch content from top results.

    Returns (results_with_links, combined_context_text).
    Each result has {title, url, snippet, content}.
    """
    loop = asyncio.get_running_loop()

    # Run scoped (Salesforce sites) and broad searches in parallel
    try:
        scoped_raw, broad_raw = await asyncio.gather(
            asyncio.wait_for(
                loop.run_in_executor(None, partial(_search_sync, query, num_results, True)),
                timeout=6.0,
            ),
            asyncio.wait_for(
                loop.run_in_executor(None, partial(_search_sync, query, num_results, False)),
                timeout=6.0,
            ),
            return_exceptions=True,
        )
    except Exception:
        logger.exception("Web research search failed")
        return [], ""

    # Merge results, dedup by URL, scoped results first
    seen_urls: set[str] = set()
    merged: list[dict] = []
    for raw in (scoped_raw, broad_raw):
        if isinstance(raw, Exception):
            continue
        for item in raw:
            url = item.get("href", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                merged.append({
                    "title": item.get("title", ""),
                    "url": url,
                    "snippet": item.get("body", ""),
                })

    if not merged:
        return [], ""

    # Fetch actual page content from top results in parallel
    top = merged[:5]
    fetch_tasks = [fetch_page_content(r["url"]) for r in top]
    contents = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    for result, content in zip(top, contents):
        result["content"] = content if isinstance(content, str) else ""

    # Build combined context from pages that returned content
    context_parts = []
    for r in top:
        page_text = r.get("content", "")
        if len(page_text) > 100:
            context_parts.append(f"[Source: {r['title']}]\nURL: {r['url']}\n{page_text}")
        elif r.get("snippet"):
            context_parts.append(f"[Source: {r['title']}]\nURL: {r['url']}\n{r['snippet']}")

    context = "\n\n---\n\n".join(context_parts)
    logger.info("Web research: %d pages searched, %d with content for: %s",
                len(top), len(context_parts), query[:60])

    return merged[:6], context
