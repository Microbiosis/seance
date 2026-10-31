"""Export the pieces the (CPU-only) demo Space needs.

The Space cannot run a 4B model at interactive speed on CPU, so it ships:
  * the fitted linear probe -- the *sensor* runs live there
  * the sitting corpus with the ghost's transcripts -- the *voice* was recorded
    offline on the local rig
  * the full audit result

Everything the console shows is either computed live from the probe or is a
verbatim transcript from the offline run; the Space says so on screen.

The probe and the sitting features are stored as base64 float32 inside JSON
rather than as .npz: the Hub treats .npz as LFS, and LFS needs a second
(presigned-S3) upload step that this machine's network cannot carry
reliably.  Plain JSON goes through as an ordinary git blob.

The figures go the same way -- base64 PNG inside assets/figures.json -- because
the Hub's LFS decision is per-file and unpredictable (two of the three PNGs
come back "regular", one comes back "lfs"), and a single LFS file is enough to
send the whole upload down the path that does not work here.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from .audit import _trial_matrix


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode("ascii")


FIGURES = ("fig_evidence.png", "fig_levels.png", "fig_audit.png", "fig_example.png")


def _b64(arr: np.ndarray, dtype: str = "float32") -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode("ascii")


def export_space_assets(feats: np.ndarray, concepts: list[str], decoder, result,
                         out_dir: Path, k: int, figures_dir: Path | None = None,
                         dtype: str = "float16") -> dict:
    """`result` is the audit JSON dict (as written to results/audit.json).

    The numeric arrays ship as float16: the build network that uploads the
    Space can only carry ~12 KB per request, so the assets have to be small.
    float16 was checked against the audited probe -- it reproduces all 60
    audited impressions (int8 does not: it flips 3 of them, because some
    top-1/top-2 margins are razor thin).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    level = result["levels"][str(k)]
    # the calibration belongs to the evidence level, not to a fresh fit
    calib = level.get("calibration", {})
    a = float(calib.get("calibration_a", 0.0))
    b = float(calib.get("calibration_b", -1.0))

    probe = {
        "note": f"{dtype} arrays, base64, C order",
        "dtype": dtype,
        "w": _b64(decoder.w, dtype),
        "mu": _b64(decoder.mu, dtype),
        "sd": _b64(decoder.sd, dtype),
        "shapes": {"w": list(decoder.w.shape), "mu": list(decoder.mu.shape), "sd": list(decoder.sd.shape)},
        "alpha": float(decoder.alpha),
        "n_classes": int(decoder.n_classes),
        "calib_a": a,
        "calib_b": b,
        "evidence_k": int(k),
    }
    (out_dir / "probe.json").write_text(json.dumps(probe), encoding="utf-8")

    rows = level["sittings"]

    def locate(row: dict) -> tuple[int, int]:
        """(image index, repetition) for a sitting row."""
        if "image_idx" in row:
            return int(row["image_idx"]), int(row["rep"])
        parts = row["key"].split("/")
        return int(parts[0].replace("img", "")), int(parts[1].replace("rep", ""))

    feats_trial = np.asarray(
        [feats[locate(r)[0], locate(r)[1]:locate(r)[1] + k].mean(axis=0) for r in rows],
        dtype=np.float32,
    )
    (out_dir / "features.json").write_text(json.dumps({
        "note": f"{dtype} array, base64, C order; one row per sitting",
        "dtype": dtype,
        "features": _b64(feats_trial, dtype),
        "shape": list(feats_trial.shape),
    }), encoding="utf-8")

    corpus = {
        "note": (
            "Ghost transcripts were generated offline with a local Gemma 3 4B IT "
            "(open weights, 4-bit NF4) on the author's machine. The impression shown "
            "in the console is recomputed live from the linear probe using the "
            "features shipped alongside it."
        ),
        "concepts": list(concepts),
        "sittings": [
            {
                "key": r["key"],
                "true_concept": r["true_concept"],
                "sighted_concept": r["sighted_concept"],
                "swapped_concept": r["swapped_concept"],
                "clarity": r["clarity"],
                "clarity_band": r["clarity_band"],
                "message_real": r["message_real"],
                "message_swapped": r["message_swapped"],
                "message_blind": r["message_blind"],
            }
            for r in rows
        ],
    }
    (out_dir / "corpus.json").write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # figures as base64 inside one JSON blob: no binary file, so no LFS
    figures_dir = figures_dir or (Path(__file__).resolve().parents[1] / "results")
    figures = {}
    for name in FIGURES:
        src = figures_dir / name
        if src.exists():
            figures[name] = base64.b64encode(src.read_bytes()).decode("ascii")
        else:
            print(f"[export] missing figure: {src}")
    if figures:
        (out_dir / "figures.json").write_text(json.dumps({
            "note": "PNG files, base64",
            "figures": figures,
        }), encoding="utf-8")

    return {
        "probe": str(out_dir / "probe.json"),
        "features": str(out_dir / "features.json"),
        "corpus": str(out_dir / "corpus.json"),
        "audit": str(out_dir / "audit.json"),
        "figures": str(out_dir / "figures.json"),
    }
