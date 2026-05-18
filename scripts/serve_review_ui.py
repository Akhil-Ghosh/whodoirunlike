#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from whodoirunlike.review_app import (
    DEFAULT_ANNOTATIONS,
    DEFAULT_SOURCE,
    DEFAULT_STATIC_DIR,
    REPO_ROOT,
    ReviewAppConfig,
    run_review_server,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the internal clip review UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--static-dir", type=Path, default=DEFAULT_STATIC_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ReviewAppConfig(
        source_path=args.source,
        annotations_path=args.annotations,
        static_dir=args.static_dir,
        repo_root=REPO_ROOT,
        limit=args.limit,
    )
    console = Console()
    server = run_review_server(args.host, args.port, config)
    console.print(f"[bold]Review UI[/bold] http://{args.host}:{args.port}")
    console.print(f"Source: {config.source_path}")
    console.print(f"Saving annotations: {config.annotations_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\nShutting down review UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
