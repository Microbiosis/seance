#!/usr/bin/env python3
"""Export the demo Space assets from an existing audit result.

The probe is refitted with the same deterministic code path (closed-form ridge,
fixed CV seed), so the shipped probe is bit-identical to the one used in the
audit; no generation is repeated.

    python scripts/export_assets.py --out app/assets
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seance import audit as audit_mod  # noqa: E402
from seance.data import load_raw, load_subject, window_features  # noqa: E402
from seance.exports import export_space_assets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--sittings", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", default=str(ROOT / "results" / "audit.json"))
    ap.add_argument("--out", default=str(ROOT / "app" / "assets"))
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()

    result = json.loads(Path(args.results).read_text(encoding="utf-8"))
    subject = load_subject(args.subject)
    raw = load_raw(args.subject)
    feats = window_features(raw)
    del raw

    cfg = audit_mod.AuditConfig(subject=args.subject, n_sittings=args.sittings, seed=args.seed)
    dec, _, _ = audit_mod.fit_decoder(feats, cfg)

    out = export_space_assets(feats, subject.concepts, dec, result, Path(args.out), args.k)
    # the ghost's own 200 concept descriptions are what the M2 metric matches
    # against; ship them with the assets so the metric can be re-checked
    atlas = ROOT / "results" / "atlas.json"
    if atlas.exists():
        shutil.copy2(atlas, Path(args.out) / "atlas.json")
        out["atlas"] = str(Path(args.out) / "atlas.json")
    for name, path in out.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
