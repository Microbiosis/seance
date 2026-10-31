#!/usr/bin/env python3
"""Rebuild every artefact from the finished audit, in the right order.

    python scripts/rebuild.py

1. report       figures + summary.json (atlas metric)
2. export       the Space assets (float16 probe/features, figures as base64)
3. pack         split the assets into parts the upload link can carry
4. readme       regenerate the Results section from the same numbers

The order matters: the atlas metric has to exist before the report is built,
and the figures have to be shrunk before they are embedded in the assets.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seance.report import build_report  # noqa: E402


def run(*args: str) -> None:
    print(f"\n=== {' '.join(args)} ===", flush=True)
    r = subprocess.run([sys.executable, "-u", *args], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"failed: {' '.join(args)}")


def main() -> None:
    result = json.loads((ROOT / "results" / "audit.json").read_text(encoding="utf-8"))
    if "atlas" not in result:
        raise SystemExit("results/audit.json has no atlas metric; run scripts/atlas_metric.py")

    print("=== report (figures + summary) ===", flush=True)
    build_report(result, ROOT / "results")

    run("scripts/export_assets.py", "--out", str(ROOT / "app" / "_assets"), "--k", "16")
    run("scripts/pack_assets.py", "--assets", str(ROOT / "app" / "_assets"),
        "--out", str(ROOT / "app" / "assets"))
    run("scripts/fill_readme.py")

    print("\n=== done ===")


if __name__ == "__main__":
    main()
