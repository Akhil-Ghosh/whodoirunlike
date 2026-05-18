from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml


@dataclass(frozen=True)
class Runner:
    slug: str
    name: str
    country: str
    primary_bucket: str
    event_tags: tuple[str, ...]
    profile: str
    search_terms: tuple[str, ...]


def load_runners(path: Path) -> list[Runner]:
    data = yaml.safe_load(path.read_text())
    runners = []
    for item in data["runners"]:
        runners.append(
            Runner(
                slug=item["slug"],
                name=item["name"],
                country=item["country"],
                primary_bucket=item["primary_bucket"],
                event_tags=tuple(item["event_tags"]),
                profile=item["profile"],
                search_terms=tuple(item["search_terms"]),
            )
        )
    return runners


def build_queries(runner: Runner, max_queries: Optional[int] = None) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()

    for term in runner.search_terms:
        candidates = [f"{runner.name} {term}"]
        if "running form" not in term:
            candidates.append(f"{runner.name} running form {term}")
        if "slow motion" not in term:
            candidates.append(f"{runner.name} slow motion {term}")
        if "training" not in term:
            candidates.append(f"{runner.name} training {term}")

        for query in candidates:
            normalized_query = " ".join(query.split())
            if normalized_query not in seen:
                queries.append(normalized_query)
                seen.add(normalized_query)

    if max_queries is not None:
        return queries[:max_queries]
    return queries


def candidate_id(runner_slug: str, video_id: str, query: str) -> str:
    digest = hashlib.sha1(f"{runner_slug}:{video_id}:{query}".encode("utf-8")).hexdigest()
    return digest[:16]


def normalize_yt_entry(runner: Runner, query: str, entry: dict[str, Any]) -> dict[str, Any]:
    video_id = entry.get("id") or entry.get("display_id") or ""
    url = entry.get("webpage_url") or entry.get("url") or ""
    if video_id and not url.startswith("http"):
        url = f"https://www.youtube.com/watch?v={video_id}"

    thumbnails = entry.get("thumbnails") or []
    thumbnail = None
    if thumbnails:
        thumbnail = thumbnails[-1].get("url")

    return {
        "candidate_id": candidate_id(runner.slug, video_id, query),
        "runner_slug": runner.slug,
        "runner_name": runner.name,
        "primary_bucket": runner.primary_bucket,
        "query": query,
        "source": "youtube",
        "url": url,
        "video_id": video_id,
        "title": entry.get("title") or "",
        "channel": entry.get("channel") or entry.get("uploader"),
        "duration_seconds": entry.get("duration"),
        "view_count": entry.get("view_count"),
        "upload_date": entry.get("upload_date"),
        "thumbnail": thumbnail,
        "discovered_at": datetime.now(timezone.utc).isoformat(),
        "review_status": "unreviewed",
        "review_notes": None,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_candidates_csv(jsonl_path: Path, csv_path: Path) -> int:
    rows = read_jsonl(jsonl_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "review_status",
        "runner_name",
        "primary_bucket",
        "title",
        "channel",
        "duration_seconds",
        "view_count",
        "upload_date",
        "url",
        "query",
        "candidate_id",
        "review_notes",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)
