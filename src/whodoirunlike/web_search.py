from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse, parse_qs

from whodoirunlike.discovery import candidate_id


YOUTUBE_VIDEO_RE = re.compile(r"(?:watch\?v=|/watch\?v=)([A-Za-z0-9_-]{11})")


def youtube_search_url(query: str) -> str:
    return f"https://www.youtube.com/results?search_query={quote_plus(query)}"


def video_id_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.hostname and "youtube.com" in parsed.hostname:
        video_id = parse_qs(parsed.query).get("v", [None])[0]
        if video_id:
            return video_id[:11]
    match = YOUTUBE_VIDEO_RE.search(url)
    return match.group(1) if match else None


def extract_youtube_ids_from_html(html: str, limit: int) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for match in YOUTUBE_VIDEO_RE.finditer(html):
        video_id = match.group(1)
        if video_id not in seen:
            ids.append(video_id)
            seen.add(video_id)
        if len(ids) >= limit:
            break
    return ids


def search_with_scrapling(query: str, limit: int) -> list[dict[str, Any]]:
    from scrapling.fetchers import Fetcher

    response = Fetcher.get(youtube_search_url(query), timeout=30000)
    html = response.body.decode("utf-8", errors="ignore")
    rows = []
    for rank, video_id in enumerate(extract_youtube_ids_from_html(html, limit), start=1):
        rows.append(
            {
                "backend": "scrapling",
                "rank": rank,
                "query": query,
                "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "title": None,
            }
        )
    return rows


def search_with_camoufox(query: str, limit: int) -> list[dict[str, Any]]:
    from camoufox.sync_api import Camoufox

    rows_by_id: dict[str, dict[str, Any]] = {}
    with Camoufox(headless=True, os="macos", locale="en-US") as browser:
        page = browser.new_page()
        page.goto(youtube_search_url(query), wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(2500)
        links = page.eval_on_selector_all(
            'ytd-video-renderer a#video-title[href*="/watch?v="]',
            """els => els.map(a => ({
                title: (a.textContent || '').replace(/\\s+/g, ' ').trim(),
                href: a.href
            }))""",
        )

    for link in links:
        video_id = video_id_from_url(link.get("href") or "")
        if not video_id or video_id in rows_by_id:
            continue
        title = link.get("title") or None
        if title and (title == "Watch" or "Now playing" in title or re.fullmatch(r"\d+:\d+", title)):
            title = None
        rows_by_id[video_id] = {
            "backend": "camoufox",
            "rank": len(rows_by_id) + 1,
            "query": query,
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "title": title,
        }
        if len(rows_by_id) >= limit:
            break

    return list(rows_by_id.values())


def make_search_row(row: dict[str, Any], runner_slug: str | None = None) -> dict[str, Any]:
    stable_runner_slug = runner_slug or "ad-hoc"
    return {
        "search_candidate_id": candidate_id(stable_runner_slug, row["video_id"], row["query"]),
        "runner_slug": runner_slug,
        "backend": row["backend"],
        "rank": row["rank"],
        "query": row["query"],
        "source": "youtube",
        "url": row["url"],
        "video_id": row["video_id"],
        "title": row.get("title"),
        "searched_at": datetime.now(UTC).isoformat(),
    }


def write_search_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    return len(rows)
