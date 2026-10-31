"""The audit: does the ghost actually see, or does it just talk?

For every evidence level K (how many presentations of the same image the
medium is allowed to average), we run the identical pipeline three times:

    real       the medium reports what the decoder saw in the sitter's EEG
    swapped    the medium reports what the decoder saw in *somebody else's* EEG
    blind      the medium reports nothing at all (the prior, with no signal)

`real` and `swapped` differ in exactly one variable: the impression.  The EEG
is the same in both, and the clarity band is held fixed on purpose, so any
difference in the prose is attributable to the sighted image and nothing else.

Three questions, three numbers:

  M1  Can it see?        decoder top-1 accuracy on 200 concepts, single trial,
                         against chance and against a shuffled-label control.
  M2  Does it say what it
      saw?               how often the message names the concept the sitter
                         actually looked at, per condition.
  M3  Is it talking about
      *you*?             how similar two messages are when the impression is
                         swapped vs. when it is kept vs. when it is dropped.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict, dataclass, field

import numpy as np

from .decoder import Decoder
from .ghost import Ghost, Sitting, build_prompt, clarity_band
from .metrics import compute_metrics

ROOT_RESULTS = Path(__file__).resolve().parents[1] / "results"


# ---------------------------------------------------------------------------
@dataclass
class AuditConfig:
    subject: int = 1
    n_sittings: int = 60
    seed: int = 0
    max_new_tokens: int = 96
    # the 80 repetitions are split three ways: 0-39 train the probe, 40-59
    # calibrate the medium, 60-79 are the sittings.  Calibration must not see
    # the sittings.
    train_reps: tuple[int, int] = (0, 40)
    calib_reps: tuple[int, int] = (40, 60)
    sitting_reps: tuple[int, int] = (60, 80)
    evidence_curve: tuple[int, ...] = (1, 2, 4, 8, 16, 20)
    # how many presentations the medium may average, per ghost audit level
    evidence_levels: tuple[int, ...] = (1, 16)
    max_new_tokens_per_message: int = 96


@dataclass
class AuditResult:
    config: dict
    decoder: dict
    evidence_curve: list[dict] = field(default_factory=list)
    levels: dict = field(default_factory=dict)  # K -> {"metrics": ..., "sittings": [...]}
    metrics: dict = field(default_factory=dict)  # primary level (largest K)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
def trial_matrix(feats: np.ndarray, reps) -> np.ndarray:
    """(n_img, n_reps, d) features for the given repetitions."""
    x = feats[:, reps]
    return x.reshape(x.shape[0], x.shape[1], -1).astype(np.float64)


def fit_decoder(feats: np.ndarray, cfg: AuditConfig):
    """Train on repetitions 0-39, with a mandatory shuffled-label control."""
    n_img = feats.shape[0]
    x_tr = trial_matrix(feats, slice(*cfg.train_reps)).reshape(-1, feats.shape[-1])
    y_tr = np.repeat(np.arange(n_img), cfg.train_reps[1] - cfg.train_reps[0])
    groups = y_tr.copy()

    dec = Decoder()
    cv = dec.fit_cv(x_tr, y_tr, groups)
    dec.fit(x_tr, y_tr)

    # shuffled-label control: same pipeline, permuted image<->trial mapping
    rng = np.random.default_rng(cfg.seed + 1)
    y_shuf = rng.permutation(y_tr)
    dec_shuf = Decoder()
    dec_shuf.fit_cv(x_tr, y_shuf, groups)
    dec_shuf.fit(x_tr, y_shuf)

    return dec, dec_shuf, cv


def decoder_report(feats: np.ndarray, dec: Decoder, dec_shuf: Decoder, cfg: AuditConfig) -> dict:
    """M1 plus the evidence curve: how much averaging does the medium need?"""
    n_img = feats.shape[0]
    x_te = trial_matrix(feats, slice(*cfg.sitting_reps))
    x_cal = trial_matrix(feats, slice(*cfg.calib_reps))
    y_cal = np.repeat(np.arange(n_img), x_cal.shape[1])

    chance = 1.0 / n_img
    curve = []
    for k in cfg.evidence_curve:
        kk = min(k, x_te.shape[1])
        feats_k = x_te[:, :kk].mean(axis=1)  # (n_img, d): k trials of the same image
        pred = dec.predict(feats_k)
        acc = float((pred == np.arange(n_img)).mean())
        pred_s = dec_shuf.predict(feats_k)
        acc_s = float((pred_s == np.arange(n_img)).mean())
        curve.append({
            "n_trials_averaged": int(kk),
            "accuracy": acc,
            "accuracy_x_chance": acc / chance,
            "shuffled_label_accuracy": acc_s,
        })

    # calibration of the medium on the calibration repetitions
    cal_feats = x_cal.reshape(-1, x_cal.shape[-1])
    cal = dec.calibrate(cal_feats, y_cal)

    return {
        "n_classes": int(n_img),
        "chance_accuracy": chance,
        "single_trial_accuracy": curve[0]["accuracy"],
        "single_trial_x_chance": curve[0]["accuracy_x_chance"],
        "best_accuracy": max(c["accuracy"] for c in curve),
        "shuffled_label_single_trial_accuracy": curve[0]["shuffled_label_accuracy"],
        "shuffled_label_best_accuracy": max(c["shuffled_label_accuracy"] for c in curve),
        "calibration": cal,
        "evidence_curve": curve,
    }


def calibrate_level(feats: np.ndarray, dec: Decoder, cfg: AuditConfig, k: int) -> dict:
    """Calibrate the medium's certainty *for this evidence level*.

    The probe's margins grow when K presentations are averaged, so a
    calibration fitted on single trials saturates (every sitting would be
    reported as certain).  We therefore fit P(top-1 correct | margin) on
    K-averaged blocks of the calibration repetitions, which is the same kind
    of feature the medium will be asked about.

    Blocks use a stride of K/4 so the calibration set has more than one block
    per image without reaching into the sittings.
    """
    lo, hi = cfg.calib_reps
    stride = max(1, k // 4)
    blocks = []
    for start in range(lo, hi - k + 1, stride):
        blocks.append(feats[:, start:start + k].mean(axis=1))
    if not blocks:
        blocks = [feats[:, lo:lo + 1].mean(axis=1)]
    x_cal = np.concatenate(blocks, axis=0)  # (n_img * n_blocks, d)
    y_cal = np.repeat(np.arange(feats.shape[0]), len(blocks))
    return dec.calibrate(x_cal, y_cal)


def build_sittings(feats: np.ndarray, concepts: list[str], dec: Decoder, cfg: AuditConfig,
                   k: int) -> list[Sitting]:
    """Pick distinct images, average K presentations of each, run the medium."""
    rng = np.random.default_rng(cfg.seed)
    rep_lo, rep_hi = cfg.sitting_reps
    n_img, n_rep = feats.shape[0], feats.shape[1]
    # every level must use the same sittings, so the repetition pool is bounded
    # by the largest level
    pool_hi = min(rep_hi, n_rep - max(cfg.evidence_levels) + 1)
    images = rng.choice(n_img, size=min(cfg.n_sittings, n_img), replace=False)
    reps = rng.integers(rep_lo, pool_hi + 1, size=len(images))

    trials = np.stack([
        feats[img, r:r + k].mean(axis=0) for img, r in zip(images, reps)
    ])
    pred = dec.predict(trials)
    clarity = dec.clarity(trials)

    sittings = []
    for i, img in enumerate(images):
        sittings.append(Sitting(
            key=f"img{int(img):03d}/rep{int(reps[i])}",
            subject=cfg.subject,
            image_idx=int(img),
            rep=int(reps[i]),
            true_concept=concepts[int(img)],
            sighted_concept=concepts[int(pred[i])],
            clarity=float(clarity[i]),
        ))
    return sittings


def derangement(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    for i in range(n):
        if p[i] == i:
            j = (i + 1 + rng.integers(0, max(n - 1, 1))) % n
            p[i], p[j] = p[j], p[i]
    if (p == np.arange(n)).any():  # belt and braces
        p = np.roll(np.arange(n), 1)
    return p


def run_level(feats: np.ndarray, concepts: list[str], dec: Decoder, ghost: Ghost,
              cfg: AuditConfig, k: int) -> dict:
    """Run the three conditions at one evidence level."""
    calibration = calibrate_level(feats, dec, cfg, k)
    print(f"[audit] K={k}: calibration {calibration['accuracy']*100:.1f}% actual / "
          f"{calibration['mean_predicted']*100:.1f}% predicted", flush=True)
    sittings = build_sittings(feats, concepts, dec, cfg, k)
    n = len(sittings)
    perm = derangement(n, cfg.seed + 2)

    real_prompts, scr_prompts, blind_prompts = [], [], []
    for i, s in enumerate(sittings):
        real_prompts.append(build_prompt(s, with_impression=True))
        other = sittings[perm[i]]
        scr = Sitting(
            key=s.key + "/swapped",
            subject=s.subject,
            image_idx=s.image_idx,
            rep=s.rep,
            true_concept=s.true_concept,          # the truth stays the truth
            sighted_concept=other.sighted_concept,  # the medium saw another brain
            clarity=s.clarity,                    # clarity held fixed on purpose
        )
        scr_prompts.append(build_prompt(scr, with_impression=True))
        blind_prompts.append(build_prompt(s, with_impression=False))

    print(f"[audit] K={k}: generating {3 * n} messages ...", flush=True)
    real_msgs = ghost.speak_many(real_prompts, progress_every=20)
    scr_msgs = ghost.speak_many(scr_prompts, progress_every=20)
    blind_msgs = ghost.speak_many(blind_prompts, progress_every=20)

    metrics = compute_metrics(sittings, real_msgs, scr_msgs, blind_msgs)

    rows = []
    for i, s in enumerate(sittings):
        rows.append({
            "key": s.key,
            "image_idx": s.image_idx,
            "rep": s.rep,
            "true_concept": s.true_concept,
            "sighted_concept": s.sighted_concept,
            "swapped_concept": sittings[perm[i]].sighted_concept,
            "clarity": s.clarity,
            "clarity_band": s.band,
            "message_real": real_msgs[i],
            "message_swapped": scr_msgs[i],
            "message_blind": blind_msgs[i],
        })

    return {"metrics": metrics, "sittings": rows, "evidence_k": k, "calibration": calibration}


def run_audit(feats: np.ndarray, concepts: list[str], dec: Decoder, ghost: Ghost,
              cfg: AuditConfig, dec_report: dict) -> AuditResult:
    levels = {}
    for k in cfg.evidence_levels:
        levels[str(k)] = run_level(feats, concepts, dec, ghost, cfg, k)
        # checkpoint after every level: a killed run should not lose the
        # generations it already paid for
        partial = ROOT_RESULTS / "audit_partial.json"
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps({
            "config": {**asdict(cfg)},
            "decoder": dec_report,
            "levels": levels,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    primary = max(cfg.evidence_levels)
    return AuditResult(
        config={**asdict(cfg), "n_sittings": cfg.n_sittings},
        decoder=dec_report,
        evidence_curve=dec_report.get("evidence_curve", []),
        levels=levels,
        metrics=levels[str(primary)]["metrics"],
    )
