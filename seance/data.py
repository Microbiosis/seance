"""Data access for the séance rig.

Source: THINGS-EEG 4D test split, published by the same author as
`illusion-brain/brain-domains` (CC BY 4.0).  We only need the *band power*
tensors (`*_band.npz`, ~18 MB per subject), which are small enough to fetch
on a laptop and large enough to be real data.

Axes of `sub-XX_band.npz` -> `band`: (image, repetition, channel, band)
Axes of `meta.parquet`       -> one row per (subject, image_idx, rep)
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = "illusion-brain/brain-domains"
TEST_DIR = "omnibrain4d/eeg"
BANDS = ("delta", "theta", "alpha", "beta", "gamma")

# THINGS-EEG test split: 200 images x 80 repetitions.  The card's protocol is
# "40 repetitions to train, 40 to test" -- we keep that split exactly.
TRAIN_REPS = slice(0, 40)
TEST_REPS = slice(40, 80)

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
    band: np.ndarray  # (n_img, n_rep, n_ch, n_band) float32
    concepts: list[str]  # concept name per image index
    images: list[str]  # relative image path per image index
    labels: np.ndarray  # class id per image index

    @property
    def n_img(self) -> int:
        return self.band.shape[0]

    @property
    def n_rep(self) -> int:
        return self.band.shape[1]

    @property
    def n_ch(self) -> int:
        return self.band.shape[2]

    @property
    def n_band(self) -> int:
        return self.band.shape[3]


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


def local_dir() -> Path | None:
    dirs = local_dirs()
    return dirs[0] if dirs else None


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
    """Load one subject's band powers and their concept labels."""
    sid = int(sid)
    band = np.load(_hub_file(f"sub-{sid:02d}_band.npz", cache_dir))["band"]

    meta = _meta(cache_dir)
    rows = meta[meta["subject"] == f"sub-{sid:02d}"]
    # one row per (image_idx, rep); the image-level table is the same for all reps
    per_image = rows.drop_duplicates(subset=["image_idx"]).sort_values("image_idx")
    if len(per_image) != band.shape[0]:
        raise ValueError(
            f"sub-{sid:02d}: meta has {len(per_image)} images, band tensor has {band.shape[0]}"
        )
    return Subject(
        sid=sid,
        band=band.astype(np.float32),
        concepts=[str(t) for t in per_image["text"].tolist()],
        images=[str(t) for t in per_image["img"].tolist()],
        labels=per_image["label"].to_numpy(),
    )


def subject_ids(cache_dir: str | None = None) -> list[int]:
    meta = _meta(cache_dir)
    ids = sorted(int(s.split("-")[1]) for s in meta["subject"].unique())
    return ids


def features(subject: Subject, reps: slice) -> np.ndarray:
    """(n_img, n_rep, n_ch*n_band) -> features, averaged over the given reps.

    We average the repetitions *before* decoding so that a single "sitting"
    can be built from a chosen number of repetitions: the séance device reads
    a short window of the recording, not the whole thing.
    """
    x = subject.band[:, reps]  # (n_img, r, n_ch, n_band)
    return x.mean(axis=1).reshape(subject.n_img, -1).astype(np.float64)


def trial_features(subject: Subject, rep: int) -> np.ndarray:
    """Single-trial features: (n_img, n_ch*n_band)."""
    x = subject.band[:, rep]
    return x.reshape(subject.n_img, -1).astype(np.float64)


def zscore_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sd = x.std(axis=0) + 1e-6
    return mu, sd


def zscore_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


# ---------------------------------------------------------------------------
# raw EEG
# ---------------------------------------------------------------------------
def load_raw(sid: int, cache_dir: str | None = None) -> np.ndarray:
    """Raw EEG of one subject: (n_img, n_rep, n_ch, T) float16."""
    sid = int(sid)
    path = _hub_file(f"sub-{sid:02d}.npz", cache_dir)
    return np.load(path)["eeg"]


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
