"""Matérn Gaussian-process posterior and deterministic grid selection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

SQRT5 = math.sqrt(5.0)


def matern52(a: np.ndarray, b: np.ndarray, lengthscale: float, sigma_f2: float) -> np.ndarray:
    """Isotropic Matern 5/2 covariance between two sets of points."""
    diff = a[:, None, :] - b[None, :, :]
    dist = np.sqrt(np.maximum((diff * diff).sum(axis=-1), 0.0))
    s = SQRT5 * dist / lengthscale
    return sigma_f2 * (1.0 + s + (s * s) / 3.0) * np.exp(-s)


@dataclass
class GPFit:
    """A fitted GP, together with what it took to fit it."""

    lengthscale: float
    sigma_f2: float
    sigma_n2: float
    log_marginal_likelihood: float
    jitter: float
    jitter_attempts: int
    n_train: int
    #: mean and scale used to standardise the training residual
    residual_mean: float = 0.0
    residual_scale: float = 1.0
    _x: np.ndarray = field(default=None, repr=False)
    _lower: np.ndarray = field(default=None, repr=False)
    _alpha: np.ndarray = field(default=None, repr=False)

    def predict(self, x_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and variance of the standardised residual."""
        k_star = matern52(self._x, x_star, self.lengthscale, self.sigma_f2)
        mean = k_star.T @ self._alpha
        v = np.linalg.solve(self._lower, k_star)
        prior_var = self.sigma_f2  # matern52(x, x) == sigma_f2 on the diagonal
        var = prior_var - (v * v).sum(axis=0)
        return mean, np.maximum(var, 0.0)

    def predict_residual(self, x_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and variance back on the residual's own scale."""
        mean, var = self.predict(x_star)
        return (
            mean * self.residual_scale + self.residual_mean,
            var * self.residual_scale * self.residual_scale,
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "lengthscale": self.lengthscale,
            "sigma_f2": self.sigma_f2,
            "sigma_n2": self.sigma_n2,
            "log_marginal_likelihood": round(float(self.log_marginal_likelihood), 9),
            "jitter": self.jitter,
            "jitter_attempts": self.jitter_attempts,
            "n_train": self.n_train,
        }


def _cholesky_with_jitter(
    matrix: np.ndarray, start: float, growth: float, max_attempts: int
) -> tuple[np.ndarray, float, int]:
    n = matrix.shape[0]
    jitter = start
    for attempt in range(1, max_attempts + 1):
        try:
            lower = np.linalg.cholesky(matrix + jitter * np.eye(n))
            return lower, jitter, attempt
        except np.linalg.LinAlgError:
            jitter *= growth
    raise np.linalg.LinAlgError(
        f"Gram matrix stayed non-positive-definite after {max_attempts} jitter "
        f"escalations up to {jitter:.3g}. Refusing to silently regularise further."
    )


def _log_marginal_likelihood(y: np.ndarray, lower: np.ndarray, alpha: np.ndarray) -> float:
    n = y.shape[0]
    return float(
        -0.5 * y @ alpha
        - np.log(np.diag(lower)).sum()
        - 0.5 * n * math.log(2.0 * math.pi)
    )


def fit_gp(
    x: np.ndarray,
    y: np.ndarray,
    grid: dict[str, Sequence[float]],
    jitter_cfg: dict[str, Any],
    noise_inflation: float = 1.0,
) -> GPFit:
    """Fit by exhaustive grid search on the exact log marginal likelihood.

    ``y`` is standardised here rather than by the caller so that the mean and
    scale travel with the fit and cannot be applied twice. Ties in marginal
    likelihood are broken by the lowest flat grid index, which the contract fixes
    as the order (lengthscale, sigma_f2, sigma_n2); without that rule two runs
    could pick different equally-good hyperparameters and diverge.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError(f"expected X (n,d) and y (n,), got {x.shape} and {y.shape}")

    residual_mean = float(y.mean())
    residual_scale = float(y.std())
    if residual_scale <= 0.0:
        residual_scale = 1.0
    y_std = (y - residual_mean) / residual_scale

    n = len(y_std)
    eye = np.eye(n)
    best: tuple[float, int, dict[str, Any]] | None = None
    flat_index = 0
    for lengthscale in grid["lengthscale"]:
        base = matern52(x, x, float(lengthscale), 1.0)
        for sigma_f2 in grid["sigma_f2"]:
            scaled = float(sigma_f2) * base
            for sigma_n2 in grid["sigma_n2"]:
                effective_noise = float(sigma_n2) * noise_inflation
                gram = scaled + effective_noise * eye
                try:
                    lower, jitter, attempts = _cholesky_with_jitter(
                        gram,
                        float(jitter_cfg["start"]),
                        float(jitter_cfg["growth"]),
                        int(jitter_cfg["max_attempts"]),
                    )
                except np.linalg.LinAlgError:
                    flat_index += 1
                    continue
                alpha = np.linalg.solve(lower.T, np.linalg.solve(lower, y_std))
                lml = _log_marginal_likelihood(y_std, lower, alpha)
                if best is None or lml > best[0]:
                    best = (
                        lml,
                        flat_index,
                        {
                            "lengthscale": float(lengthscale),
                            "sigma_f2": float(sigma_f2),
                            "sigma_n2": effective_noise,
                            "log_marginal_likelihood": lml,
                            "jitter": jitter,
                            "jitter_attempts": attempts,
                            "lower": lower,
                            "alpha": alpha,
                        },
                    )
                flat_index += 1

    if best is None:
        raise np.linalg.LinAlgError(
            "no hyperparameter combination in the fixed grid produced a "
            "positive-definite Gram matrix"
        )

    chosen = best[2]
    return GPFit(
        lengthscale=chosen["lengthscale"],
        sigma_f2=chosen["sigma_f2"],
        sigma_n2=chosen["sigma_n2"],
        log_marginal_likelihood=chosen["log_marginal_likelihood"],
        jitter=chosen["jitter"],
        jitter_attempts=chosen["jitter_attempts"],
        n_train=n,
        residual_mean=residual_mean,
        residual_scale=residual_scale,
        _x=x,
        _lower=chosen["lower"],
        _alpha=chosen["alpha"],
    )
