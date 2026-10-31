"""Figures and tables for the audit report.

Three figures carry the story:

  fig_evidence.png    how many presentations the medium needs before it sees
                      anything (decoder level)
  fig_levels.png     the same question at the ghost level: the séance has a dial
  fig_audit.png      the three conditions for one sitting -- what the ghost said
                      when the brain signal was real, swapped, or absent
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)


# The link that uploads this project can only carry ~12 KB per request, and the
# post links the figures as raw URLs, so the PNGs have to stay small enough to
# be pushed at all.  Re-render smaller and palette-quantise until they fit.
FIG_LIMIT_BYTES = 10_500


def shrink_figure(path: Path) -> Path:
    try:
        import io

        from PIL import Image
    except ImportError:
        return path
    if path.stat().st_size <= FIG_LIMIT_BYTES:
        return path
    im = Image.open(path).convert("RGB")
    best = None
    for scale in (0.8, 0.66, 0.55, 0.46):
        w = max(320, int(im.width * scale))
        h = max(200, int(im.height * scale))
        small = im.resize((w, h), Image.LANCZOS)
        for colors in (64, 32, 16):
            buf = io.BytesIO()
            small.convert("P", palette=Image.ADAPTIVE, colors=colors).save(buf, format="PNG", optimize=True)
            data = buf.getvalue()
            if best is None or len(data) < len(best[0]):
                best = (data, w, h)
            if len(data) <= FIG_LIMIT_BYTES:
                path.write_bytes(data)
                return path
    if best:
        path.write_bytes(best[0])
    return path


def _primary_level(result: dict) -> str:
    levels = result.get("levels", {})
    return max(levels, key=lambda k: int(k)) if levels else ""


# The ghost almost never says the concept out loud -- it evokes it.  The
# word-boundary metric therefore undercounts badly, so the headline numbers use
# the atlas metric (see scripts/atlas_metric.py): every message is matched
# against the *same local model's* own 200 concept descriptions by content-word
# overlap.  Both sides are in the same register, which is what makes the
# overlap meaningful.
_ATLAS_COND = {"message_real": "real", "message_swapped": "swapped", "message_blind": "blind"}


def atlas_m2(result: dict, level: str) -> dict | None:
    """M2 in the atlas metric's shape, or None when the atlas pass has not run.

    Returns {cond: {"rate": float, "ci95": [lo, hi]}} for real/swapped/blind.
    """
    atlas = result.get("atlas")
    if not atlas or level not in atlas:
        return None
    out = {}
    for cond, name in _ATLAS_COND.items():
        cell = atlas[level].get(cond)
        if not cell:
            continue
        out[name] = {"rate": cell["matches_truth_rate"], "ci95": cell["matches_truth_ci95"]}
    return out or None


def m2_for(result: dict, level: str) -> tuple[dict, str]:
    """The M2 numbers to report, plus the name of the metric they came from."""
    atlas = atlas_m2(result, level)
    if atlas:
        return atlas, "atlas"
    return result["levels"][level]["metrics"]["M2_names_true_concept"], "word-boundary"


def atlas_fidelity(result: dict, level: str) -> dict | None:
    """How often the ghost actually talks about the impression it was given."""
    atlas = result.get("atlas")
    if not atlas or level not in atlas:
        return None
    out = {}
    for cond, name in _ATLAS_COND.items():
        cell = atlas[level].get(cond)
        if cell and cell.get("matches_impression_rate") is not None:
            out[name] = {"rate": cell["matches_impression_rate"],
                         "ci95": cell["matches_impression_ci95"]}
    return out or None


def plot_evidence_curve(curve: list[dict], chance: float, out: Path) -> Path:
    ks = [c["n_trials_averaged"] for c in curve]
    acc = [c["accuracy"] * 100 for c in curve]
    shuf = [c["shuffled_label_accuracy"] * 100 for c in curve]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    ax.plot(ks, acc, "o-", color="#8b2f6b", lw=2, label="real EEG")
    ax.plot(ks, shuf, "s--", color="#9aa0a6", lw=1.6, label="shuffled labels (control)")
    ax.axhline(chance * 100, color="#c0392b", lw=1.4, ls=":", label=f"chance ({chance*100:.1f}%)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("presentations averaged per image")
    ax.set_ylabel("top-1 accuracy on 200 concepts (%)")
    ax.set_title("How many presentations does the medium need?")
    _style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_levels(result: dict, out: Path) -> Path:
    """How the naming rate moves with the number of presentations averaged."""
    levels = result.get("levels", {})
    if not levels:
        return out
    ks = sorted(int(k) for k in levels)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    width = 0.26
    xs = np.arange(len(ks))
    metric_name = "atlas"
    for i, (cond, label, color) in enumerate(
        (("real", "real EEG", "#8b2f6b"), ("swapped", "swapped EEG", "#c08457"),
         ("blind", "blind (no EEG)", "#9aa0a6"))
    ):
        vals, lo, hi = [], [], []
        for k in ks:
            m2, metric_name = m2_for(result, str(k))
            m = m2[cond]
            vals.append(m["rate"] * 100)
            lo.append(m["rate"] * 100 - m["ci95"][0] * 100)
            hi.append(m["ci95"][1] * 100 - m["rate"] * 100)
        ax.bar(xs + (i - 1) * width, vals, width=width, color=color, label=label)
        ax.errorbar(xs + (i - 1) * width, vals, yerr=[lo, hi], fmt="none",
                    ecolor="#333333", capsize=3, lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_xlabel("presentations of the same image averaged by the medium")
    ax.set_ylabel(_m2_label(metric_name))
    ax.set_title("The séance has a dial")
    _style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _m2_label(metric_name: str) -> str:
    if metric_name == "atlas":
        return "messages matching the true concept (%)"
    return "messages naming the true concept (%)"


def plot_audit(result: dict, out: Path) -> Path:
    """Two panels: naming rate and similarity."""
    level = _primary_level(result)
    m2, metric_name = m2_for(result, level)
    m3 = result["metrics"]["M3_message_similarity"]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), dpi=160)

    ax = axes[0]
    conds = ["real", "swapped", "blind"]
    labels = ["real EEG", "swapped EEG", "blind (no EEG)"]
    vals = [m2[c]["rate"] * 100 for c in conds]
    cis = [m2[c]["ci95"] for c in conds]
    errs = [[v - lo * 100 for v, (lo, _) in zip(vals, cis)],
            [hi * 100 - v for v, (_, hi) in zip(vals, cis)]]
    colors = ["#8b2f6b", "#c08457", "#9aa0a6"]
    ax.bar(labels, vals, color=colors, width=0.6)
    ax.errorbar(labels, vals, yerr=errs, fmt="none", ecolor="#333333", capsize=4, lw=1.2)
    ax.set_ylabel(_m2_label(metric_name))
    ax.set_title("Does the ghost say what the sitter saw?")
    _style(ax)

    ax = axes[1]
    names = ["different\nsitters", "impression\nswapped", "impression\ndropped", "blind\nto blind"]
    means = [
        m3["different_sitters_mean"],
        m3["same_sitting_impression_swapped_mean"],
        m3["same_sitting_impression_dropped_mean"],
        m3["blind_to_blind_mean"],
    ]
    ax.bar(names, means, color=["#4a6fa5", "#c08457", "#9aa0a6", "#cfd3d6"], width=0.6)
    ax.set_ylabel("mean content-word Jaccard")
    ax.set_title("Is it talking about this sitter?")
    _style(ax)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_example(result: dict, out: Path) -> Path:
    """A worked example: what the ghost said, next to what actually happened."""
    level = _primary_level(result)
    rows = result.get("levels", {}).get(level, {}).get("sittings", [])
    if not rows:
        return out

    fig, ax = plt.subplots(figsize=(9.4, 3.4), dpi=160)
    ax.axis("off")
    picked = [r for r in rows if r["sighted_concept"] == r["true_concept"]][:2] or rows[:2]
    txt = []
    for r in picked:
        txt.append(
            f"saw: {r['true_concept']}   medium: {r['sighted_concept']}\n"
            f"REAL   : {r['message_real'][:150].strip()}\n"
            f"SWAPPED: {r['message_swapped'][:150].strip()}\n"
            + "-" * 78
        )
    ax.text(0.0, 1.0, "\n".join(txt), va="top", ha="left", fontsize=7.0, family="monospace",
            transform=ax.transAxes)
    ax.set_title("two sittings, one ghost", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def build_report(result: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    dec = result["decoder"]
    level = _primary_level(result)
    m2, metric_name = m2_for(result, level)
    m3 = result["metrics"]["M3_message_similarity"]
    fidelity = atlas_fidelity(result, level) or {}

    figures = [
        shrink_figure(plot_evidence_curve(dec["evidence_curve"], dec["chance_accuracy"],
                                          out_dir / "fig_evidence.png")),
        shrink_figure(plot_levels(result, out_dir / "fig_levels.png")),
        shrink_figure(plot_audit(result, out_dir / "fig_audit.png")),
        shrink_figure(plot_example(result, out_dir / "fig_example.png")),
    ]

    summary = {
        "primary_level_K": int(level) if level else None,
        "m2_metric": metric_name,
        "decoder": {
            "single_trial_accuracy_pct": dec["single_trial_accuracy"] * 100,
            "single_trial_x_chance": dec["single_trial_x_chance"],
            "best_accuracy_pct": dec["best_accuracy"] * 100,
            "shuffled_label_best_accuracy_pct": dec["shuffled_label_best_accuracy"] * 100,
            "chance_pct": dec["chance_accuracy"] * 100,
        },
        "ghost": {
            "names_true_real_pct": m2["real"]["rate"] * 100,
            "names_true_real_ci95": [x * 100 for x in m2["real"]["ci95"]],
            "names_true_swapped_pct": m2["swapped"]["rate"] * 100,
            "names_true_blind_pct": m2["blind"]["rate"] * 100,
            "impression_correct_rate_pct": result["metrics"].get("impression_correct_rate", 0) * 100,
            "mean_content_words": result["metrics"].get("mean_content_words_per_message", 0),
            "similarity_different_sitters": m3["different_sitters_mean"],
            "similarity_impression_swapped": m3["same_sitting_impression_swapped_mean"],
            "similarity_impression_dropped": m3["same_sitting_impression_dropped_mean"],
            "similarity_blind_to_blind": m3["blind_to_blind_mean"],
        },
    }
    if fidelity:
        summary["ghost"]["fidelity_real_pct"] = fidelity["real"]["rate"] * 100
        summary["ghost"]["fidelity_real_ci95"] = [x * 100 for x in fidelity["real"]["ci95"]]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"figures": [str(p) for p in figures], "summary": summary}
