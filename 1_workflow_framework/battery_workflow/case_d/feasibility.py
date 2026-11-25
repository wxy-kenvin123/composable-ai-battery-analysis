"""Probability of feasibility from candidate features and observed attempt statuses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

FEASIBLE_STATUS = "ok"
INFEASIBLE_STATUS = "hard_fail"
EXCLUDED_STATUSES = frozenset({"data_missing"})


@dataclass
class FeasibilityConfig:


    l2_lambda: float = 1.0
    min_positives: int = 2
    min_negatives: int = 2
    inactive_pof: float = 1.0
    pof_floor: float = 0.05
    #: candidates whose probability of feasibility is below this are withheld from
    #: the acquisition (Gelbart et al. 2014, the probabilistic feasibility
    #: constraint); 0.0 disables the constraint and leaves the pure product
    pof_veto_threshold: float = 0.5


    veto_rule: str = "point"
    lcb_kappa: float = 1.0
    max_iter: int = 100
    tol: float = 1e-8

    @classmethod
    def from_contract(cls, contract: dict[str, Any]) -> "FeasibilityConfig":
        block = contract["feasibility_model"]
        act = block["activation"]
        num = block["numerics"]
        return cls(
            l2_lambda=float(block["l2_lambda"]),
            min_positives=int(act["min_positives"]),
            min_negatives=int(act["min_negatives"]),
            inactive_pof=float(act["inactive_pof"]),
            pof_floor=float(block["pof_floor"]),
            pof_veto_threshold=float(block.get("pof_veto_threshold", 0.0)),
            veto_rule=str(block.get("veto_rule", "point")),
            lcb_kappa=float(block.get("lcb_kappa", 1.0)),
            max_iter=int(num["max_iter"]),
            tol=float(num["tol"]),
        )


def labels_from_statuses(statuses: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Map attempt statuses to (keep_mask, y): ok -> 1, hard_fail -> 0, others dropped."""
    keep = np.array([s in (FEASIBLE_STATUS, INFEASIBLE_STATUS) for s in statuses], dtype=bool)
    y = np.array([1.0 if s == FEASIBLE_STATUS else 0.0 for s in statuses], dtype=float)
    return keep, y[keep]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass
class FeasibilityModel:
    """Class-balanced, L2-regularised logistic regression with an unpenalised intercept.

    ``fit`` returns whether the model became active. When inactive, ``pof``
    returns ``inactive_pof`` for every candidate and ``ei_scale`` returns
    ``None`` so the plan stage runs untouched.
    """

    config: FeasibilityConfig = field(default_factory=FeasibilityConfig)
    coef_: np.ndarray | None = None      # shape (d + 1,), intercept first
    active: bool = False
    n_positives: int = 0
    n_negatives: int = 0
    n_iter: int = 0
    converged: bool = False

    # ----- fitting ------------------------------------------------------------
    def fit(self, features: np.ndarray, y: np.ndarray) -> bool:
        X = np.asarray(features, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError("features and labels disagree in length")
        n_pos = int(np.sum(y == 1.0))
        n_neg = int(np.sum(y == 0.0))
        if n_pos + n_neg != y.shape[0]:
            raise ValueError("labels must be 0/1 only; drop excluded statuses before fitting")
        self.n_positives, self.n_negatives = n_pos, n_neg
        if n_pos < self.config.min_positives or n_neg < self.config.min_negatives:
            self.coef_ = None
            self.active = False
            self.n_iter = 0
            self.converged = False
            return False

        n, d = X.shape
        # class-balanced weights: each class contributes half of the total weight
        w = np.where(y == 1.0, n / (2.0 * n_pos), n / (2.0 * n_neg))
        A = np.hstack([np.ones((n, 1)), X])                     # intercept first
        penalty = np.full(d + 1, float(self.config.l2_lambda))
        penalty[0] = 0.0                                          # intercept unpenalised
        beta = np.zeros(d + 1)
        converged = False
        it = 0
        for it in range(1, self.config.max_iter + 1):
            z = A @ beta
            p = _sigmoid(z)
            grad = A.T @ (w * (p - y)) + penalty * beta
            s = w * p * (1.0 - p)
            hess = (A * s[:, None]).T @ A + np.diag(penalty)
            hess[np.diag_indices_from(hess)] += 1e-10           # deterministic numerical guard
            step = np.linalg.solve(hess, grad)
            beta = beta - step
            if float(np.max(np.abs(step))) < self.config.tol:
                converged = True
                break
        self.coef_ = beta
        self.active = True
        self.n_iter = it
        self.converged = converged
        return True

    # ----- prediction ---------------------------------------------------------
    def pof(self, features: np.ndarray) -> np.ndarray:
        X = np.asarray(features, dtype=float)
        if not self.active or self.coef_ is None:
            return np.full(X.shape[0], self.config.inactive_pof, dtype=float)
        z = self.coef_[0] + X @ self.coef_[1:]
        return _sigmoid(z)

    def ei_scale(self, features: np.ndarray) -> np.ndarray | None:
        """The factor handed to ``plan_stage``; ``None`` while the model is inactive."""
        if not self.active:
            return None
        return np.clip(self.pof(features), self.config.pof_floor, 1.0)


    def event_row(self) -> dict[str, Any]:
        return {
            "pof_active": bool(self.active),
            "pof_n_positives": int(self.n_positives),
            "pof_n_negatives": int(self.n_negatives),
            "pof_iterations": int(self.n_iter),
            "pof_converged": bool(self.converged),
        }
