#!/usr/bin/env python3
"""The atlas metric: what is the ghost actually talking about?

The ghost almost never says the concept out loud -- it evokes it ("the heat...
copper... burnt sugar" for a blowtorch).  A word-match metric therefore
undercounts badly, and a generic sentence encoder does not fix it either (we
tried: the true concept lands at median rank 110/200).

So we let the *same local model* write a short spirit description of every one
of the 200 concepts -- the ghost's own atlas -- and match each séance message
against that atlas with content-word overlap.  Both sides are in the same
register, which is what makes the overlap meaningful.

Two numbers come out:

    matches_impression   the message is about what the medium reported
    matches_truth        the message is about what the sitter actually saw

The first is the ghost's fidelity.  The second is the whole channel from
brain to prose: fidelity x how often the medium was right.

    python scripts/atlas_metric.py --model models/gemma-3-4b-it-bnb-4bit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seance.data import load_subject  # noqa: E402
from seance.ghost import Ghost, Sitting, build_prompt  # noqa: E402
from seance.metrics import bootstrap_ci, content_words  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "gemma-3-4b-it-bnb-4bit"
ATLAS_PATH = ROOT / "results" / "atlas.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                    help="local Gemma 3 directory")
    return ap.parse_args()


def build_atlas(concepts: list[str], model: str) -> dict[str, str]:
    if ATLAS_PATH.exists():
        cached = json.loads(ATLAS_PATH.read_text(encoding="utf-8"))
        if set(cached) == set(concepts):
            print(f"[atlas] cached ({len(cached)} descriptions)")
            return cached

    ghost = Ghost(model, max_new_tokens=64, seed=0)
    atlas = {}
    for i, c in enumerate(concepts, 1):
        s = Sitting(key=c, subject=1, image_idx=0, rep=0, true_concept=c,
                    sighted_concept=c, clarity=0.5)
        atlas[c] = ghost.speak(build_prompt(s))
        if i % 25 == 0:
            print(f"[atlas] {i}/{len(concepts)}", flush=True)
    ATLAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ATLAS_PATH.write_text(json.dumps(atlas, indent=2, ensure_ascii=False), encoding="utf-8")
    return atlas


def match(message: str, atlas: dict[str, str], keys: list[str]) -> str:
    """Nearest concept in the ghost's own voice (content-word Jaccard)."""
    mw = content_words(message)
    best, best_score = keys[0], -1.0
    for c in keys:
        aw = content_words(atlas[c])
        score = len(mw & aw) / max(len(mw | aw), 1)
        if score > best_score:
            best, best_score = c, score
    return best


def main() -> None:
    args = parse_args()
    subject = load_subject(1)
    concepts = list(subject.concepts)
    atlas = build_atlas(concepts, args.model)
    keys = list(atlas)

    path = ROOT / "results" / "audit.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    out = {}

    for level, blob in result["levels"].items():
        rows = blob["sittings"]
        level_out = {}
        for cond, impression_key in (("message_real", "sighted_concept"),
                                     ("message_swapped", "swapped_concept"),
                                     ("message_blind", None)):
            hits_impression, hits_truth = [], []
            # Fidelity restricted to the sittings where the medium was right.
            # This is the factor that makes the brain-to-prose channel close
            # arithmetically: a message can only match the true concept if the
            # medium reported it, so
            #     P(match truth) = P(medium right) x P(faithful | medium right).
            fid_right_num, fid_right_den = 0, 0
            for r in rows:
                c = match(r[cond], atlas, keys)
                hit_truth = 1.0 if c == r["true_concept"] else 0.0
                hits_truth.append(hit_truth)
                if impression_key:
                    hit_imp = 1.0 if c == r[impression_key] else 0.0
                    hits_impression.append(hit_imp)
                    if r[impression_key] == r["true_concept"]:
                        fid_right_den += 1
                        fid_right_num += hit_imp
            level_out[cond] = {
                "matches_truth_rate": float(np.mean(hits_truth)),
                "matches_truth_ci95": bootstrap_ci(hits_truth),
                "matches_impression_rate": float(np.mean(hits_impression)) if hits_impression else None,
                "matches_impression_ci95": bootstrap_ci(hits_impression) if hits_impression else None,
                "fidelity_given_medium_right": (fid_right_num / fid_right_den) if fid_right_den else None,
                "n_sittings_medium_right": fid_right_den,
            }
            print(f"K={level:<3} {cond:<16} matches truth {np.mean(hits_truth)*100:5.1f}% "
                  f"| matches impression "
                  f"{'' if not hits_impression else f'{np.mean(hits_impression)*100:5.1f}%'}"
                  f"{'' if not fid_right_den else f' | faithful given medium right {fid_right_num/fid_right_den*100:5.1f}% (n={fid_right_den})'}")
        out[level] = level_out

    result["atlas"] = out
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Rebuild the figures and the summary now that the atlas metric exists, so
    # the plots and the write-up quote the same numbers.
    from seance.report import build_report
    build_report(result, ROOT / "results")
    print("[atlas] written; figures and summary rebuilt with the atlas metric")


if __name__ == "__main__":
    main()
