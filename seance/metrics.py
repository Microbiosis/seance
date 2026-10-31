"""Text metrics: how do we score what the ghost said?

Kept apart from the audit machinery (seance/audit.py) because these are the
pieces the M2 and M3 numbers are made of:

  mentions        word-boundary match (the strict metric)
  content_words   the vocabulary of a message, stopwords removed
  jaccard         overlap between two messages
  bootstrap_ci    percentile bootstrap confidence interval
  compute_metrics M2 and M3 for one evidence level
"""

from __future__ import annotations

import re
import string

import numpy as np

STOPWORDS = set(
    """a an the and or but if then than that this these those of in on at to for from by with
    without into over under again further once here there when where why how all any both each
    few more most other some such no nor not only own same so too very can will just should now
    i me my we our you your he him his she her it its they them their what which who whom
    is are was were be been being have has had do does did as it its about across behind beyond
    through during before after above below up down out off between among against because
    until while above below up down out off""".split()
)


def norm(text: str) -> str:
    return text.lower().strip()


def mentions(message: str, concept: str) -> bool:
    """Word-boundary match, tolerant of multi-word concepts.

    Concept labels in the dataset use underscores ("aircraft_carrier") while the
    ghost writes prose ("aircraft carrier"), so both separators are accepted.
    """
    concept = norm(concept).replace("_", " ")
    parts = [p for p in concept.split() if p]
    if not parts:
        return False
    pattern = r"\b" + r"[\s\-_]+".join(re.escape(p) for p in parts) + r"\b"
    return re.search(pattern, norm(message)) is not None


def content_words(text: str) -> set[str]:
    words = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text.lower()).split()
    return {w for w in words if len(w) >= 3 and w not in STOPWORDS}


def jaccard(a: str, b: str) -> float:
    wa, wb = content_words(a), content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0, alpha: float = 0.05):
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def compute_metrics(sittings, real_msgs, scr_msgs, blind_msgs) -> dict:
    n = len(sittings)

    def rate(msgs, predicate):
        vals = [1.0 if predicate(m, i) else 0.0 for i, m in enumerate(msgs)]
        return float(np.mean(vals)), bootstrap_ci(vals)

    real_true, real_true_ci = rate(real_msgs, lambda m, i: mentions(m, sittings[i].true_concept))
    scr_true, scr_true_ci = rate(scr_msgs, lambda m, i: mentions(m, sittings[i].true_concept))
    blind_true, blind_true_ci = rate(blind_msgs, lambda m, i: mentions(m, sittings[i].true_concept))

    real_sight, real_sight_ci = rate(real_msgs, lambda m, i: mentions(m, sittings[i].sighted_concept))
    scr_sight, scr_sight_ci = rate(scr_msgs, lambda m, i: mentions(m, sittings[i].sighted_concept))

    # M3: how much does the message depend on the impression?
    swap = [jaccard(real_msgs[i], scr_msgs[i]) for i in range(n)]
    drop = [jaccard(real_msgs[i], blind_msgs[i]) for i in range(n)]
    within = []
    for i in range(n):
        for j in range(i + 1, n):
            within.append(jaccard(real_msgs[i], real_msgs[j]))
    blind_within = []
    for i in range(n):
        for j in range(i + 1, n):
            blind_within.append(jaccard(blind_msgs[i], blind_msgs[j]))
    within_mean = float(np.mean(within)) if within else float("nan")
    blind_within_mean = float(np.mean(blind_within)) if blind_within else float("nan")

    # how many content words does a message have?
    lens = [len(content_words(m)) for m in real_msgs]

    # impression correctness: how often was the medium's impression right?
    impression_correct = float(np.mean([
        1.0 if s.sighted_concept == s.true_concept else 0.0 for s in sittings
    ]))

    return {
        "M2_names_true_concept": {
            "real": {"rate": real_true, "ci95": real_true_ci},
            "swapped": {"rate": scr_true, "ci95": scr_true_ci},
            "blind": {"rate": blind_true, "ci95": blind_true_ci},
            "delta_real_minus_swapped": real_true - scr_true,
            "delta_real_minus_swapped_ci95": bootstrap_ci(
                [(1.0 if mentions(real_msgs[i], sittings[i].true_concept) else 0.0)
                 - (1.0 if mentions(scr_msgs[i], sittings[i].true_concept) else 0.0)
                 for i in range(n)]
            ),
        },
        "M2_names_impression": {
            "real": {"rate": real_sight, "ci95": real_sight_ci},
            "swapped": {"rate": scr_sight, "ci95": scr_sight_ci},
        },
        "impression_correct_rate": impression_correct,
        "M3_message_similarity": {
            "different_sitters_mean": within_mean,
            "same_sitting_impression_swapped_mean": float(np.mean(swap)),
            "same_sitting_impression_dropped_mean": float(np.mean(drop)),
            "blind_to_blind_mean": blind_within_mean,
            "impression_swapped_ci95": bootstrap_ci(swap),
            "impression_dropped_ci95": bootstrap_ci(drop),
            "n_pairs_within": len(within),
        },
        "mean_content_words_per_message": float(np.mean(lens)),
        "n_messages": n,
    }
