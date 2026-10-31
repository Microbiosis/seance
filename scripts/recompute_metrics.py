#!/usr/bin/env python3
"""Recompute the audit metrics from the stored transcripts.

The messages are the expensive part (360 generations); the metrics are not.
This script reloads `results/audit.json`, rebuilds the metric layer, and
rewrites the JSON and the figures.  Use it after changing a metric.

    python scripts/recompute_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seance.ghost import Sitting  # noqa: E402
from seance.metrics import compute_metrics  # noqa: E402
from seance.report import build_report  # noqa: E402


def to_sitting(row: dict) -> Sitting:
    return Sitting(
        key=row["key"], subject=0, image_idx=0, rep=0,
        true_concept=row["true_concept"], sighted_concept=row["sighted_concept"],
        clarity=row["clarity"],
    )


def main() -> None:
    path = ROOT / "results" / "audit.json"
    result = json.loads(path.read_text(encoding="utf-8"))

    for level, blob in result["levels"].items():
        rows = blob["sittings"]
        sittings = [to_sitting(r) for r in rows]
        blob["metrics"] = compute_metrics(
            sittings,
            [r["message_real"] for r in rows],
            [r["message_swapped"] for r in rows],
            [r["message_blind"] for r in rows],
        )

    primary = max(result["levels"], key=lambda k: int(k))
    result["metrics"] = result["levels"][primary]["metrics"]
    result["primary_level"] = primary

    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    build_report(result, ROOT / "results")

    m2 = result["metrics"]["wordmatch_names_true_concept"]
    m2i = result["metrics"]["wordmatch_names_impression"]
    print(f"primary level K={primary}")
    for c in ("real", "swapped", "blind"):
        lo, hi = m2[c]["ci95"]
        print(f"  word-boundary match  {c:<8} {m2[c]['rate']*100:5.1f}%  CI {lo*100:.1f}-{hi*100:.1f}%")
    print(f"  word-boundary impression        {m2i['real']['rate']*100:5.1f}%")
    print(f"  impression correct           {result['metrics']['impression_correct_rate']*100:5.1f}%")


if __name__ == "__main__":
    main()
