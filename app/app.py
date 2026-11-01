"""Séance console -- the interactive face of the rig.

Two panels:

  * the séance   : hold a sitting, see the medium's impression and what the
                    ghost said, with the two control readings side by side
  * the audit    : the numbers -- can it see, does it say what it saw, is it
                    talking about *you*

The console runs on a CPU-only Streamlit host: the linear probe (the sensor)
is computed live from the shipped features, while the ghost's words are
transcripts recorded offline with the same local model.  The screen says so.
"""

from __future__ import annotations

import base64
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get("SEANCE_ASSETS", ROOT / "assets"))
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Séance — a local medium that claims to read a brain",
                   page_icon="🕯️", layout="centered")

N_CH = 63
N_WIN = 10


@st.cache_resource
def load_assets() -> dict:
    """Read the packed assets: manifest -> parts -> gzip -> base64 -> bytes."""
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))

    def unpack(name: str) -> bytes:
        entry = manifest["assets"][name]
        blob = "".join((ASSETS / p).read_text(encoding="ascii") for p in entry["parts"])
        return gzip.decompress(base64.b64decode(blob))

    probe = json.loads(unpack("probe.json"))
    feats_blob = json.loads(unpack("features.json"))
    corpus = json.loads(unpack("corpus.json"))
    audit = json.loads(unpack("audit.json"))
    figures = json.loads(unpack("figures.json")).get("figures", {})
    levels = audit.get("levels", {})
    primary = max(levels, key=lambda k: int(k)) if levels else None
    sittings = levels.get(primary, {}).get("sittings", []) if primary else []
    return {
        "probe": probe,
        "features": _decode(feats_blob),
        "corpus": corpus, "audit": audit,
        "figures": figures,
        "levels": levels, "primary": primary, "sittings": sittings,
    }


def _decode(blob: dict) -> np.ndarray:
    """base64 -> array (the assets ship as float16; see scripts/export_assets.py)."""
    raw = base64.b64decode(blob["features"])
    dtype = np.float16 if blob.get("dtype") == "float16" else np.float32
    return np.frombuffer(raw, dtype=dtype).reshape(blob["shape"]).astype(np.float32)


def sight(probe: dict, x: np.ndarray) -> tuple[int, float]:
    """The medium: linear probe -> top concept + calibrated clarity."""
    dtype = np.float16 if probe.get("dtype") == "float16" else np.float32
    w = np.frombuffer(base64.b64decode(probe["w"]), dtype=dtype).reshape(probe["shapes"]["w"]).astype(np.float32)
    mu = np.frombuffer(base64.b64decode(probe["mu"]), dtype=dtype).reshape(probe["shapes"]["mu"]).astype(np.float32)
    sd = np.frombuffer(base64.b64decode(probe["sd"]), dtype=dtype).reshape(probe["shapes"]["sd"]).astype(np.float32)
    s = ((x - mu) / sd) @ w
    order = np.argsort(-s)
    top1, top2 = int(order[0]), int(order[1])
    a = float(probe.get("calib_a", 0.0))
    b = float(probe.get("calib_b", -1.0))
    margin = float(s[top1] - s[top2])
    clarity = 1.0 / (1.0 + np.exp(-(a * margin + b)))
    return top1, clarity


def _window_plot(grid: np.ndarray):
    """The electrode grid, or None when this host has no matplotlib.

    A missing plotting library must not take the whole console down with it.
    """
    try:
        import matplotlib
    except ImportError:
        return None

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.4, 2.6), dpi=140)
    z = (grid - grid.mean()) / (grid.std() + 1e-9)
    im = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xlabel("100 ms window")
    ax.set_ylabel("electrode")
    ax.set_xticks(range(N_WIN))
    ax.set_xticklabels([f"{100 * (i + 1)}" for i in range(N_WIN)], fontsize=6)
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    return fig


def main() -> None:
    assets = load_assets()
    probe = assets["probe"]
    corpus = assets["corpus"]
    audit = assets["audit"]
    sittings = assets["sittings"]
    concepts = corpus["concepts"]

    st.title("🕯️ Séance")
    st.markdown(
        "A local, open-weight model sits at a séance table and claims to read what "
        "is in your head. Then we check whether it can."
    )
    st.caption(
        "Model: Gemma 3 4B IT (open weights, 4-bit NF4), running offline. "
        "Signal: THINGS-EEG (CC BY 4.0). No API key, no account, no data leaving this machine."
    )
    if assets["primary"]:
        st.caption(
            f"This console replays the K={assets['primary']} evidence level "
            "(presentations of the same image averaged by the medium)."
        )

    tab_seance, tab_audit, tab_about = st.tabs(["Hold the séance", "The audit", "About"])

    with tab_seance:
        st.markdown(
            "The medium reads one image from this sitter's EEG and reports a single "
            "impression, plus how clear it is. The ghost speaks from that impression."
        )
        if st.button("Hold the séance", type="primary"):
            st.session_state["sitting_idx"] = int(np.random.randint(0, len(sittings)))
        idx = st.session_state.get("sitting_idx", 0)
        row = sittings[idx]

        feat = assets["features"][idx].astype(np.float64)
        top_concept, clarity = sight(probe, feat)

        left, right = st.columns([1, 1.4])
        with left:
            st.subheader("The medium")
            st.markdown(f"**Impression:** `{concepts[top_concept]}`")
            st.metric("clarity", f"{clarity * 100:.0f}%")
            st.caption("calibrated P(the impression is right)")
            with st.expander("the signal behind the impression"):
                grid = feat.reshape(N_CH, N_WIN)
                fig = _window_plot(grid)
                if fig is None:
                    st.caption("the electrode grid needs matplotlib, which this host does not have")
                else:
                    st.pyplot(fig, clear_figure=True)
                    st.caption("amplitude per electrode across ten 100 ms windows")
            with st.expander("what the sitter actually saw"):
                st.write(f"**{row['true_concept']}**")
                st.caption("ground truth from the experiment, not shown to the ghost")
        with right:
            st.subheader("The ghost")
            st.markdown(f"*{row['message_real']}*")
            st.divider()
            with st.expander("control: the medium read somebody else's brain"):
                st.markdown(f"*{row['message_swapped']}*")
            with st.expander("control: the medium read nothing at all"):
                st.markdown(f"*{row['message_blind']}*")

    with tab_audit:
        dec = audit["decoder"]
        m3 = audit["metrics"]["M3_message_similarity"]
        # M2 is the atlas metric: each message is matched against the ghost's
        # own 200 concept descriptions by content-word overlap.
        level = assets["primary"]
        atlas = (audit.get("atlas") or {}).get(level, {})
        st.subheader("Can it see?")
        c1, c2, c3 = st.columns(3)
        c1.metric("single-trial top-1", f"{dec['single_trial_accuracy'] * 100:.1f}%",
                  f"{dec['single_trial_x_chance']:.1f}x chance")
        c2.metric("chance", f"{dec['chance_accuracy'] * 100:.1f}%")
        c3.metric("shuffled-label control", f"{dec['shuffled_label_best_accuracy'] * 100:.1f}%")
        st.caption("200-way classification from one second of EEG, held-out repetitions.")

        st.subheader("Does it say what the sitter saw?")
        rows = []
        if atlas:
            for cond, label in (("message_real", "real EEG"),
                                ("message_swapped", "swapped EEG"),
                                ("message_blind", "blind (no EEG)")):
                cell = atlas[cond]
                lo, hi = cell["matches_truth_ci95"]
                rows.append({"condition": label,
                             "matches the true concept": f"{cell['matches_truth_rate'] * 100:.1f}%",
                             "95% CI": f"{lo * 100:.1f}–{hi * 100:.1f}%"})
        st.dataframe(rows, use_container_width=True, hide_index=True)
        if atlas:
            fid = atlas["message_real"]["matches_impression_rate"]
            st.caption(
                f"The ghost talks about the impression it was handed in {fid * 100:.1f}% of "
                "sittings; matched against the ghost's own 200 concept descriptions."
            )

        st.subheader("Is it talking about this sitter?")
        st.dataframe([
            {"pairing": "two different sitters", "mean content-word overlap": f"{m3['different_sitters_mean']:.3f}"},
            {"pairing": "same sitter, impression swapped", "mean content-word overlap": f"{m3['same_sitting_impression_swapped_mean']:.3f}"},
            {"pairing": "same sitter, impression dropped", "mean content-word overlap": f"{m3['same_sitting_impression_dropped_mean']:.3f}"},
            {"pairing": "blind to blind", "mean content-word overlap": f"{m3['blind_to_blind_mean']:.3f}"},
        ], use_container_width=True, hide_index=True)

        for fig, blob in (assets.get("figures") or {}).items():
            try:
                st.image(base64.b64decode(blob), use_container_width=True, caption=fig)
            except Exception:
                pass

    with tab_about:
        st.markdown(
            "**What this is.** A séance console: a linear probe reads one second of "
            "EEG and reports a single concept; a local Gemma 3 4B turns that impression "
            "into a message from beyond. The interesting part is not the spooky part -- "
            "it is the audit. We run the same pipeline three times: with the real brain, "
            "with somebody else's brain, and with no brain at all, and we measure how much "
            "of the ghost's speech is actually about the person sitting in front of it."
        )
        st.markdown(
            "**Data.** THINGS-EEG (Gifford et al., NeuroImage 2022), 10 subjects, "
            "63 electrodes, 200 test concepts, published as "
            "[`illusion-brain/brain-domains`](https://huggingface.co/datasets/illusion-brain/brain-domains) (CC BY 4.0)."
        )
        st.markdown(
            "**Honesty note.** The ghost's words on this page were generated offline on "
            "the author's machine with the same local model and are replayed verbatim; "
            "the *impression* is recomputed live from the shipped linear probe."
        )


if __name__ == "__main__":
    main()
