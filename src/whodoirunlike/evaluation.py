from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from whodoirunlike.discovery import read_jsonl


APPROVAL_CHANNEL_HINTS = {
    "world athletics": 14,
    "olympics": 14,
    "nbc sports": 12,
    "wanda diamond league": 12,
    "citius mag": 8,
    "flotrack": 8,
    "sweat elite": 8,
    "james dunne": 5,
}

GOOD_TITLE_HINTS = {
    "running form": 12,
    "slow motion": 12,
    "training": 8,
    "workout": 8,
    "race": 7,
    "final": 6,
    "highlights": 4,
    "world record": 6,
    "diamond league": 6,
    "olympic": 6,
    "marathon": 6,
    "1500m": 5,
    "800m": 5,
    "5000m": 5,
    "10000m": 5,
    "10k": 5,
    "mile": 5,
}

BAD_TITLE_HINTS = {
    "podcast": -18,
    "interview": -14,
    "press conference": -18,
    "reaction": -10,
    "explained": -8,
    "documentary": -5,
    "sprint form": -8,
    "football": -20,
    "basketball": -20,
}

TITLE_PREFIX_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class CandidateScore:
    row: dict[str, Any]
    score: int
    recommendation: str
    reasons: tuple[str, ...]


def normalize_text(value: Any) -> str:
    return str(value or "").casefold()


def runner_tokens(name: str) -> set[str]:
    return {token for token in TITLE_PREFIX_RE.split(normalize_text(name)) if len(token) > 2}


def duration_score(duration_seconds: float | int | None) -> tuple[int, str]:
    if duration_seconds is None:
        return -2, "unknown duration"
    duration = float(duration_seconds)
    if duration <= 0:
        return -8, "invalid duration"
    if duration <= 20:
        return 6, "short clip"
    if duration <= 180:
        return 10, "reviewable short video"
    if duration <= 600:
        return 4, "medium source video"
    if duration <= 1800:
        return -2, "long source video"
    return -8, "very long source video"


def score_candidate(row: dict[str, Any]) -> CandidateScore:
    title = normalize_text(row.get("title"))
    channel = normalize_text(row.get("channel"))
    runner_name = str(row.get("runner_name") or "")
    tokens = runner_tokens(runner_name)

    score = 50
    reasons: list[str] = []

    if tokens and all(token in title for token in tokens):
        score += 22
        reasons.append("runner named in title")
        title_match = "full"
    elif tokens and any(token in title for token in tokens):
        score += 8
        reasons.append("partial runner name match")
        title_match = "partial"
    else:
        score -= 25
        reasons.append("runner not named in title")
        title_match = "none"

    if row.get("primary_bucket") == "marathon" and "marathon" in title:
        score += 8
        reasons.append("marathon title match")
    if row.get("primary_bucket") == "800_1500" and any(term in title for term in ["800m", "1500m", "mile"]):
        score += 8
        reasons.append("middle-distance title match")
    if row.get("primary_bucket") == "5k_10k" and any(
        term in title for term in ["5000m", "5k", "10000m", "10k", "cross country"]
    ):
        score += 8
        reasons.append("long-track title match")

    for hint, points in GOOD_TITLE_HINTS.items():
        if hint in title:
            score += points
            reasons.append(f"+{hint}")

    for hint, points in BAD_TITLE_HINTS.items():
        if hint in title:
            score += points
            reasons.append(f"{hint} penalty")

    for hint, points in APPROVAL_CHANNEL_HINTS.items():
        if hint in channel:
            score += points
            reasons.append(f"source: {hint}")
            break

    view_count = row.get("view_count") or 0
    if view_count >= 1_000_000:
        score += 6
        reasons.append("high views")
    elif view_count >= 100_000:
        score += 3
        reasons.append("moderate views")

    duration_points, duration_reason = duration_score(row.get("duration_seconds"))
    score += duration_points
    reasons.append(duration_reason)

    if title_match == "none":
        score = min(score, 58)
    elif title_match == "partial":
        score = min(score, 82)

    if score >= 85:
        recommendation = "review_first"
    elif score >= 65:
        recommendation = "review"
    elif score >= 45:
        recommendation = "maybe"
    else:
        recommendation = "skip"

    return CandidateScore(
        row=row,
        score=max(0, min(score, 100)),
        recommendation=recommendation,
        reasons=tuple(reasons[:8]),
    )


def evaluate_candidates(jsonl_path: Path, csv_path: Path) -> int:
    rows = read_jsonl(jsonl_path)
    scored = sorted(
        (score_candidate(row) for row in rows),
        key=lambda item: (item.score, item.row.get("view_count") or 0),
        reverse=True,
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "score",
        "recommendation",
        "runner_name",
        "primary_bucket",
        "title",
        "channel",
        "duration_seconds",
        "view_count",
        "url",
        "query",
        "candidate_id",
        "reasons",
        "review_status",
        "review_notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in scored:
            row = dict(item.row)
            row["score"] = item.score
            row["recommendation"] = item.recommendation
            row["reasons"] = "; ".join(item.reasons)
            writer.writerow(row)

    return len(scored)
