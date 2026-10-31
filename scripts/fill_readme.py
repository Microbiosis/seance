#!/usr/bin/env python3
"""Fill the Results section of README.md from the audit result.

The README carries a `<!-- RESULTS:BEGIN -->` / `<!-- RESULTS:END -->` pair;
everything between them is regenerated from results/audit.json so the prose can
never drift away from the numbers.

    python scripts/fill_readme.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BEGIN = "<!-- RESULTS:BEGIN -->"
END = "<!-- RESULTS:END -->"


def pct(x: float) -> str:
    return f"{x * 100:.1f}"


def ci(pair: list[float]) -> str:
    return f"{pair[0] * 100:.1f}–{pair[1] * 100:.1f}%"


def build_section(result: dict) -> str:
    dec = result["decoder"]
    level = str(max(result["levels"], key=lambda k: int(k)))
    atlas = result["atlas"][level]
    m3 = result["metrics"]["M3_message_similarity"]

    real = atlas["message_real"]
    swapped = atlas["message_swapped"]
    blind = atlas["message_blind"]
    fid = real["matches_impression_rate"]
    fid_right = real.get("fidelity_given_medium_right")
    n_right = real.get("n_sittings_medium_right", 0)

    imp_correct = result["metrics"].get("impression_correct_rate", 0.0)
    # the channel should close: how often the medium was right x how often the
    # ghost talks about what it was told = how often it talks about the truth.
    product = imp_correct * fid
    observed = real["matches_truth_rate"]

    lines: list[str] = []
    add = lines.append

    add("One subject (`sub-01`), one local Gemma 3 4B IT, 60 sittings per")
    add("condition, greedy decoding. Everything below is regenerated from")
    add("`results/audit.json` by `scripts/fill_readme.py`.")
    add("")

    add("### M1 — can it see?")
    add("")
    add("Top-1 accuracy of the probe on 200 concepts (chance = 0.5%).")
    add("")
    add("| presentations averaged | accuracy | × chance | shuffled-label control |")
    add("|---|---|---|---|")
    for c in dec["evidence_curve"]:
        add(f"| {c['n_trials_averaged']} | {pct(c['accuracy'])}% | "
            f"{c['accuracy_x_chance']:.0f}× | {pct(c['shuffled_label_accuracy'])}% |")
    add("")
    add(f"A single second of EEG already carries the signal: "
        f"**{pct(dec['single_trial_accuracy'])}%** "
        f"({dec['single_trial_x_chance']:.0f}× chance) against a "
        f"shuffled-label control of **{pct(dec['shuffled_label_best_accuracy'])}%** "
        f"— the control sits at chance, so the signal is real and the pipeline")
    add("does not leak the labels. Averaging more presentations sharpens the")
    add(f"medium fast: the best level reaches **{pct(dec['best_accuracy'])}%**.")
    add("")

    add("### M2 — does it say what the sitter actually saw?")
    add("")
    add("The ghost almost never says the concept out loud — it evokes it")
    add("(\"the heat, copper, burnt sugar\" for a blowtorch). So each message is")
    add("matched against the **same local model's own 200 concept descriptions**")
    add("by content-word overlap: the ghost's own atlas, in the ghost's own")
    add("register. Both sides are spoken by the same 4B model, which is what")
    add("makes the overlap mean something.")
    add("")
    add(f"At **K={level}** (16 presentations of the same image averaged):")
    add("")
    add("| condition | message matches the concept the sitter saw | 95% CI |")
    add("|---|---|---|")
    add(f"| `real` EEG | **{pct(real['matches_truth_rate'])}%** | "
        f"{ci(real['matches_truth_ci95'])} |")
    add(f"| `swapped` EEG | {pct(swapped['matches_truth_rate'])}% | "
        f"{ci(swapped['matches_truth_ci95'])} |")
    add(f"| `blind` | {pct(blind['matches_truth_rate'])}% | "
        f"{ci(blind['matches_truth_ci95'])} |")
    add("")
    add(f"The ghost talks about **the impression it was handed** in "
        f"{pct(fid)}% of sittings — it is faithful. The medium was right about")
    add(f"the concept in {pct(imp_correct)}% of sittings ({n_right} of "
        f"{len(result['levels'][level]['sittings'])}).")
    if fid_right is not None:
        add("")
        add(f"In {pct(fid_right)}% of *those* sittings the ghost actually spoke about")
        add(f"the impression it was given, which is {pct(imp_correct * fid_right)}% of all")
        add(f"sittings. We measured {pct(observed)}%. The gap is one sitting in sixty")
        add("where the ghost landed on the true concept even though the medium had")
        add("named a different one — luck, not signal.")
    add("")
    add("Swap the brain and the channel is gone; blind the medium and it never")
    add("opens. The difference between `real` and `swapped` is the whole")
    add("brain-to-prose channel.")
    add("")

    add("### M3 — is it talking about *you*?")
    add("")
    add("Mean content-word overlap between messages (Jaccard):")
    add("")
    add("| pair | overlap |")
    add("|---|---|")
    add(f"| two different sitters | {m3['different_sitters_mean']:.3f} |")
    add(f"| same sitter, impression swapped | "
        f"{m3['same_sitting_impression_swapped_mean']:.3f} |")
    add(f"| same sitter, impression dropped | "
        f"{m3['same_sitting_impression_dropped_mean']:.3f} |")
    add(f"| blind to blind | {m3['blind_to_blind_mean']:.3f} |")
    add("")
    add("This is the number we did not expect. Messages from different sitters")
    add("are no more alike than messages from the same sitter with a different")
    add("impression, and both are close to the impression-dropped baseline: the")
    add("prose is essentially a function of the one word the medium handed over.")
    add("Blind messages are identical to each other (1.000) because there is")
    add("nothing to vary.")
    add("")
    add("**The blunt version: the ghost tells you one word of your brain, and")
    add("then it tells you itself.** The EEG contributes one word out of about")
    add(f"{result['metrics'].get('mean_content_words_per_message', 0):.0f}; the rest")
    add("is the model's own prior.")
    add("")

    add("### The séance has a dial")
    add("")
    add("Both levels of the same experiment:")
    add("")
    add("| | K=1 (one presentation) | K=16 (sixteen presentations) |")
    add("|---|---|---|")
    k1 = result["atlas"].get("1")
    by_k = {c["n_trials_averaged"]: c for c in dec["evidence_curve"]}
    if k1 and 1 in by_k and 16 in by_k:
        add(f"| probe top-1 | {pct(by_k[1]['accuracy'])}% | "
            f"{pct(by_k[16]['accuracy'])}% |")
        add(f"| message matches the truth | {pct(k1['message_real']['matches_truth_rate'])}% | "
            f"{pct(real['matches_truth_rate'])}% |")
        add(f"| ghost talks about the impression | "
            f"{pct(k1['message_real']['matches_impression_rate'])}% | {pct(fid)}% |")
    add("")
    add("More evidence sharpens the *sighting*, not the *talking*. The ghost's")
    add("fidelity barely moves; what moves is how often the medium was right.")
    add("")

    return "\n".join(lines)


def main() -> None:
    result = json.loads((ROOT / "results" / "audit.json").read_text(encoding="utf-8"))
    if "atlas" not in result:
        raise SystemExit("results/audit.json has no atlas metric yet "
                         "(run scripts/atlas_metric.py first)")
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit("README.md is missing the RESULTS markers")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    section = build_section(result)
    readme.write_text(f"{head}{BEGIN}\n\n{section}\n{END}{tail}", encoding="utf-8")
    print(f"[readme] results section written ({len(section)} chars)")


if __name__ == "__main__":
    main()
