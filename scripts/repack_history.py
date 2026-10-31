"""Rebuild the local history as many small commits.

The link that pushes this repository drops any HTTP request body larger than
roughly 100 KB, and a single `git push` sends one pack containing every
object the remote is missing.  One commit holding the whole repository
therefore becomes one oversized request that never gets an answer.

Splitting the same tree into many small commits makes each `git push` carry a
pack small enough to survive.  Files are grouped by their raw byte size so
that every commit stays well below the limit.

    python scripts/repack_history.py --budget 32000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return out.stdout


def tracked_files() -> list[str]:
    return sorted(p for p in git("ls-files").splitlines() if p.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=32000)
    ap.add_argument("--branch", default="upload-main")
    args = ap.parse_args()

    files = tracked_files()
    if not files:
        print("nothing tracked", file=sys.stderr)
        return 1

    subprocess.run(["git", "-C", str(REPO), "checkout", "--orphan", args.branch], check=True)
    subprocess.run(["git", "-C", str(REPO), "read-tree", "--empty"], check=True)

    groups: list[tuple[list[str], int]] = []
    group: list[str] = []
    total = 0
    for name in files:
        size = (REPO / name).stat().st_size
        if group and total + size > args.budget:
            groups.append((group, total))
            group = []
            total = 0
        group.append(name)
        total += size
    if group:
        groups.append((group, total))

    for index, (names, size) in enumerate(groups, start=1):
        subprocess.run(["git", "-C", str(REPO), "add", "--", *names], check=True)
        message = f"seance: {names[0]}" + (f" (+{len(names) - 1} more)" if len(names) > 1 else "")
        subprocess.run(["git", "-C", str(REPO), "commit", "-q", "-m", message], check=True)
        print(f"[{index}/{len(groups)}] {len(names)} files, {size / 1024:.1f} KB -> {message}")

    print(f"{len(groups)} commits, budget {args.budget} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
