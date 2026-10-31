#!/usr/bin/env python3
"""Run the séance audit end to end.

    # decoder only (no language model needed)
    python scripts/run_audit.py --decode-only

    # full audit with the local ghost
    python scripts/run_audit.py --model models/gemma-3-4b-it-bnb-4bit

Outputs land in results/ as JSON + figures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seance import audit as audit_mod  # noqa: E402
from seance.data import load_raw, load_subject, window_features  # noqa: E402
from seance.decoder import Decoder  # noqa: E402
from seance.report import build_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--sittings", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", type=str, default=None, help="local Gemma 3 directory")
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--out", type=str, default=str(ROOT / "results"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    subject = load_subject(args.subject)
    raw = load_raw(args.subject)
    feats = window_features(raw)
    del raw
    print(f"[data] subject {subject.sid}: features {feats.shape}, "
          f"{subject.n_img} concepts", flush=True)

    cfg = audit_mod.AuditConfig(
        subject=args.subject,
        n_sittings=args.sittings,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )

    dec, dec_shuf, cv = audit_mod.fit_decoder(feats, cfg)
    dec_report = audit_mod.decoder_report(feats, dec, dec_shuf, cfg)

    print(f"[decoder] alpha={dec.alpha}  cv={cv}", flush=True)
    print(f"[decoder] single trial top-1 = {dec_report['single_trial_accuracy']*100:.2f}% "
          f"({dec_report['single_trial_x_chance']:.1f}x chance)", flush=True)
    print(f"[decoder] shuffled-label control = "
          f"{dec_report['shuffled_label_best_accuracy']*100:.2f}%", flush=True)
    for c in dec_report["evidence_curve"]:
        print(f"           {c['n_trials_averaged']:>3} sitting(s): "
              f"{c['accuracy']*100:6.2f}%  ({c['accuracy_x_chance']:5.1f}x chance)  "
              f"control {c['shuffled_label_accuracy']*100:5.2f}%", flush=True)

    if args.decode_only:
        payload = {
            "config": {"subject": subject.sid, "seed": args.seed},
            "decoder": dec_report,
            "evidence_curve": dec_report["evidence_curve"],
            "sittings": [],
            "metrics": {},
        }
        (out_dir / "audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[done] {time.time()-t0:.0f}s -> {out_dir/'audit.json'}")
        return

    if not args.model:
        raise SystemExit("--model is required unless --decode-only")

    from seance.ghost import Ghost

    ghost = Ghost(args.model, max_new_tokens=args.max_new_tokens, seed=args.seed)
    print(f"[ghost] ready ({time.time()-t0:.0f}s)", flush=True)

    result = audit_mod.run_audit(feats, subject.concepts, dec, ghost, cfg, dec_report)
    result.decoder = dec_report
    result.evidence_curve = dec_report["evidence_curve"]

    (out_dir / "audit.json").write_text(result.to_json(), encoding="utf-8")
    build_report(json.loads(result.to_json()), out_dir)

    from seance.exports import export_space_assets

    primary_k = max(cfg.evidence_levels)
    export_space_assets(feats, subject.concepts, dec, result, ROOT / "app" / "assets", primary_k)

    m2 = result.metrics["M2_names_true_concept"]
    m3 = result.metrics["M3_message_similarity"]
    print("\n=== M2: does the ghost name the concept the sitter saw? ===")
    for c in ("real", "swapped", "blind"):
        lo, hi = m2[c]["ci95"]
        print(f"  {c:<8} {m2[c]['rate']*100:5.1f}%  (95% CI {lo*100:.1f}-{hi*100:.1f}%)")
    print("\n=== M3: is it talking about this sitter? ===")
    print(f"  different sitters            {m3['different_sitters_mean']:.3f}")
    print(f"  impression swapped           {m3['same_sitting_impression_swapped_mean']:.3f}")
    print(f"  impression dropped           {m3['same_sitting_impression_dropped_mean']:.3f}")
    print(f"  blind to blind               {m3['blind_to_blind_mean']:.3f}")
    print(f"\n[done] {time.time()-t0:.0f}s -> {out_dir/'audit.json'}")


if __name__ == "__main__":
    main()
