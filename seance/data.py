"""Data access for the séance rig.

Source: THINGS-EEG 4D test split, published by the same author as
`illusion-brain/brain-domains` (CC BY 4.0).  Two files per subject:
`meta.parquet` for the concept labels, and `sub-XX.npz` for the raw EEG
(~445 MB per subject).

Axes of `sub-XX.npz`    -> `eeg`: (image, repetition, channel, sample)
Axes of `meta.parquet` -> one row per (subject, image_idx, rep)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = "illusion-brain/brain-domains"
TEST_DIR = "omnibrain4d/eeg"

# Feature extraction, byte-for-byte the same convention as the published
# protocol this project follows: baseline correction over the 48 ms before
# stimulus onset, then ten 100 ms windows -> 63 channels x 10 windows = 630
# dimensions per trial.
WIN = 25       # 100 ms at 250 Hz
BASELINE = 12  # 48 ms before onset


@dataclass(frozen=True)
class Subject:
    """Everything the rig needs about one subject."""

    sid: int
    concepts: list[str]  # concept name per image index
    images: list[str]  # relative image path per image index
    labels: np.ndarray  # class id per image index


def local_dirs() -> list[Path]:
    """Optional mirror directories holding the files already fetched.

    `SEANCE_DATA_DIR` may list several directories (os.pathsep separated), so
    that big files and small ones can live in different places.  An offline
    machine (or one that already has the files) can skip the hub entirely.
    """
    d = os.environ.get("SEANCE_DATA_DIR")
    if not d:
        return []
    return [Path(p) for p in d.split(os.pathsep) if p]


def _hub_file(name: str, cache_dir: str | None = None) -> str:
    for mirror in local_dirs():
        # a mirror may be flat (name) or keep the repo layout (omnibrain4d/eeg/name)
        for candidate in (mirror / name, mirror / TEST_DIR / name):
            if candidate.exists():
                return str(candidate)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        REPO,
        f"{TEST_DIR}/{name}",
        repo_type="dataset",
        cache_dir=cache_dir,
    )


def _meta(cache_dir: str | None = None):
    import pandas as pd

    path = _hub_file("meta.parquet", cache_dir)
    return pd.read_parquet(path)


def load_subject(sid: int, cache_dir: str | None = None) -> Subject:
    """Load one subject's concept labels."""
    sid = int(sid)
    meta = _meta(cache_dir)
    rows = meta[meta["subject"] == f"sub-{sid:02d}"]
    # one row per (image_idx, rep); the image-level table is the same for all reps
    per_image = rows.drop_duplicates(subset=["image_idx"]).sort_values("image_idx")
    return Subject(
        sid=sid,
        concepts=[str(t) for t in per_image["text"].tolist()],
        images=[str(t) for t in per_image["img"].tolist()],
        labels=per_image["label"].to_numpy(),
    )


# ---------------------------------------------------------------------------
# raw EEG
# ---------------------------------------------------------------------------
def load_raw(sid: int, cache_dir: str | None = None) -> np.ndarray:
    """Raw EEG of one subject: (n_img, n_rep, n_ch, T) float16."""
    sid = int(sid)
    path = _hub_file(f"sub-{sid:02d}.npz", cache_dir)
    eeg = np.load(path)["eeg"]
    n_img = len(load_subject(sid, cache_dir).concepts)
    if eeg.shape[0] != n_img:
        raise ValueError(
            f"sub-{sid:02d}: meta has {n_img} images, EEG tensor has {eeg.shape[0]}"
        )
    return eeg


def window_features(eeg: np.ndarray, chunk: int = 8) -> np.ndarray:
    """(n_img, n_rep, n_ch, T) -> (n_img, n_rep, n_ch * n_win) float32.

    Baseline correction over the pre-stimulus samples, then the recording is
    cut into ten 100 ms windows and averaged inside each window.  Window-major
    ordering (channel-major within a window) matches the published protocol.
    """
    n_img, n_rep, n_ch, T = eeg.shape
    n_bin = T // WIN
    out = np.empty((n_img, n_rep, n_ch * n_bin), dtype=np.float32)
    for s in range(0, n_img, chunk):
        e = eeg[s:s + chunk].astype(np.float32)
        e -= e[:, :, :, :BASELINE].mean(axis=3, keepdims=True)
        e = e.reshape(e.shape[0], n_rep, n_ch, n_bin, WIN).mean(axis=4)
        e = e.transpose(0, 1, 3, 2)
        out[s:s + chunk] = e.reshape(e.shape[0], n_rep, n_ch * n_bin)
        del e
    return out
