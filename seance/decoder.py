"""The "sensor": EEG -> concept.

A deliberately boring linear probe (ridge to one-hot targets) rather than a
deep model, because the whole point of the audit is that the *decoder* is
measuring something small and real, and the *ghost* is the thing under
suspicion.  A fancy decoder would blur the attribution.

Protocol (matches the published dataset card):
  * 200-way classification, single trial, THINGS-EEG test split
  * repetitions 0-39 train, 40-79 test
  * alpha chosen by image-level 5-fold CV on the training repetitions
  * a shuffled-label control is mandatory: a decoder that cannot beat chance
    when the labels are permuted is not measuring anything
"""

from __future__ import annotations

import numpy as np

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
N_FOLDS = 5
SEED = 0


def _solve_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """(X^T X + alpha*I) W = X^T Y  --  closed form, deterministic."""
    d = x.shape[1]
    gram = x.T @ x
    gram.flat[:: d + 1] += alpha * len(x)
    return np.linalg.solve(gram, x.T @ y)


def _one_hot(n: int) -> np.ndarray:
    return np.eye(n)


class Decoder:
    """Ridge probe over EEG band-power features."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.w: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.n_classes: int = 0
        self._margin_calib: tuple[float, float] | None = None  # (a, b) for logistic

    # -- training ----------------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray) -> "Decoder":
        """x: (n_samples, d) features, y: (n_samples,) class ids."""
        self.mu = x.mean(axis=0)
        self.sd = x.std(axis=0) + 1e-6
        xs = (x - self.mu) / self.sd
        self.n_classes = int(y.max()) + 1
        self.w = _solve_ridge(xs, _one_hot(self.n_classes)[y], self.alpha)
        return self

    def fit_cv(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray,
               alphas=ALPHAS, n_folds: int = N_FOLDS) -> dict:
        """Choose alpha by cross-validation over *repetitions*.

        The folds split each image's repetitions, not the images themselves:
        the test protocol is "same images, new repetitions", so a held-out
        image would be a class the classifier has never seen and the CV
        accuracy would be identically zero for every alpha.
        """
        xs = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)
        self.n_classes = int(y.max()) + 1
        rng = np.random.default_rng(SEED)
        fold_id = np.zeros(len(y), dtype=int)
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            idx = idx[rng.permutation(len(idx))]
            fold_id[idx] = np.arange(len(idx)) % n_folds

        curve = {}
        for a in alphas:
            accs = []
            for k in range(n_folds):
                va = fold_id == k
                tr = ~va
                w = _solve_ridge(xs[tr], _one_hot(self.n_classes)[y[tr]], a)
                pred = xs[va] @ w
                accs.append(float((pred.argmax(axis=1) == y[va]).mean()))
            curve[a] = float(np.mean(accs))
        self.alpha = max(curve, key=lambda a: curve[a])
        self.fit(x, y)
        self.cv_curve = {str(k): v for k, v in curve.items()}
        return self.cv_curve

    # -- inference ---------------------------------------------------------
    def scores(self, x: np.ndarray) -> np.ndarray:
        if self.w is None or self.mu is None or self.sd is None:
            raise RuntimeError("decoder is not fitted")
        return ((x - self.mu) / self.sd) @ self.w

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.scores(x).argmax(axis=1)

    def margin(self, x: np.ndarray) -> np.ndarray:
        """Top-1 minus top-2 score: how separated the winner is."""
        s = self.scores(x)
        part = np.partition(s, -2, axis=1)
        return part[:, -1] - part[:, -2]

    # -- calibration -------------------------------------------------------
    def calibrate(self, x: np.ndarray, y: np.ndarray) -> dict:
        """Fit P(top-1 correct | margin) on held-out data.

        The séance console needs an honest number to hand the ghost:
        "the medium senses this clearly (37%)" is only meaningful if the
        37% is calibrated against reality.
        """
        pred = self.predict(x)
        correct = (pred == y).astype(float)
        m = self.margin(x)
        # 1-D logistic regression by Newton's method (deterministic).
        a, b = 0.0, -1.0
        for _ in range(50):
            z = a * m + b
            p = 1.0 / (1.0 + np.exp(-z))
            w = np.clip(p * (1 - p), 1e-9, None)
            g0 = float((p - correct).sum())
            g1 = float(((p - correct) * m).sum())
            h00 = float(w.sum()) + 1e-9
            h01 = float((w * m).sum())
            h11 = float((w * m * m).sum()) + 1e-9
            det = h00 * h11 - h01 * h01
            if abs(det) < 1e-12:
                break
            da = (h11 * g0 - h01 * g1) / det
            db = (h00 * g1 - h01 * g0) / det
            a -= da
            b -= db
            if abs(da) + abs(db) < 1e-10:
                break
        self._margin_calib = (a, b)
        p = 1.0 / (1.0 + np.exp(-(a * m + b)))
        return {
            "alpha": self.alpha,
            "accuracy": float(correct.mean()),
            "calibration_a": float(a),
            "calibration_b": float(b),
            "mean_predicted": float(p.mean()),
            "mean_actual": float(correct.mean()),
            "cv_curve": getattr(self, "cv_curve", {}),
        }

    def clarity(self, x: np.ndarray) -> np.ndarray:
        """Calibrated P(top-1 correct) per sample."""
        if self._margin_calib is None:
            raise RuntimeError("call calibrate() first")
        a, b = self._margin_calib
        return 1.0 / (1.0 + np.exp(-(a * self.margin(x) + b)))
