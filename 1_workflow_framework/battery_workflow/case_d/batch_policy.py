"""Batch selection on the observed Gaussian-process posterior."""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import numpy as np

from . import acquisition as acq
from .policy import Proposal, prior_columns
from .surrogate import fit_gp


def plan_one(
    state,
    mechanism,
    memory,
    unseen: Sequence[int],
    weights: np.ndarray,
    ei_scale: np.ndarray | None = None,
) -> Proposal:
    """One ParEGO-EI pick with an explicit weight vector.

    The body after the weight lookup is the same arithmetic as
    ``policy.plan_stage``. Weights are an argument so a batch can walk a
    stride schedule without touching ``state.round_index`` mid-batch (except
    when the caller is reproducing ``plan_stage``, in which case it passes
    ``weight_for_round(lattice, state.round_index)``).
    """
    contract = state.contract
    acq_cfg = contract["acquisition"]
    grid = contract["surrogate"]["grid"]
    jitter = contract["surrogate"]["cholesky_jitter"]
    rho = float(acq_cfg["rho"])
    weights = np.asarray(weights, dtype=float)
    m = np.asarray(state.m, dtype=float)
    lower, upper = acq.observed_bounds(m)

    s_obs = acq.augmented_chebyshev(
        acq.min_max_normalise(m, lower, upper), weights, rho
    )

    if mechanism.enabled:
        prior_obs, prior_unseen = prior_columns(state, mechanism, m)
        mu0_obs = acq.augmented_chebyshev(
            acq.min_max_normalise(prior_obs, lower, upper), weights, rho
        )
        mu0_unseen = acq.augmented_chebyshev(
            acq.min_max_normalise(prior_unseen, lower, upper), weights, rho
        )
    else:
        mu0_obs = np.zeros(len(s_obs), dtype=float)
        mu0_unseen = np.zeros(len(unseen), dtype=float)

    inflation, _ = memory.inflation(state.scalarised_residuals)

    x_obs = state.observed_features()
    x_unseen = np.asarray(state.space.feature_matrix(unseen), dtype=float)
    gp = fit_gp(x_obs, s_obs - mu0_obs, grid, jitter, noise_inflation=inflation)
    residual_mean, residual_var = gp.predict_residual(x_unseen)
    mean = mu0_unseen + residual_mean

    incumbent = float(s_obs.min())
    ei = acq.expected_improvement(
        mean, residual_var, incumbent, float(acq_cfg["expected_improvement"]["xi"])
    )
    if ei_scale is not None:
        ei = ei * np.asarray(ei_scale, dtype=float)
    pick = acq.argmax_first(ei)

    return Proposal(
        candidate_id=int(unseen[pick.index]),
        expected_improvement=pick.value,
        tied=pick.tied,
        n_tied=pick.n_tied,
        predicted_scalarised=float(mean[pick.index]),
        predicted_variance=float(residual_var[pick.index]),
        incumbent=incumbent,
        weights=tuple(float(w) for w in weights),
        noise_inflation=inflation,
        gp=gp,
        decision_source="parego_ei",
    )


def _drop_candidate(mechanism, unseen: Sequence[int], ei_scale, candidate_id: int):
    idx = list(unseen).index(int(candidate_id))
    remaining = tuple(cid for i, cid in enumerate(unseen) if i != idx)
    remaining_scale = None
    if ei_scale is not None:
        remaining_scale = np.delete(np.asarray(ei_scale, dtype=float), idx)
    remaining_mech = mechanism
    if mechanism.enabled:
        extra: dict[str, Any] = {}
        if mechanism.tplus_hat_unseen is not None:
            extra["tplus_hat_unseen"] = np.delete(
                np.asarray(mechanism.tplus_hat_unseen), idx
            )
        remaining_mech = dataclasses.replace(
            mechanism,
            m1_prior_unseen=np.delete(np.asarray(mechanism.m1_prior_unseen), idx),
            m2_hat_unseen=np.delete(np.asarray(mechanism.m2_hat_unseen), idx),
            m3_hat_unseen=np.delete(np.asarray(mechanism.m3_hat_unseen), idx),
            **extra,
        )
    return remaining, remaining_mech, remaining_scale


def plan_batch(
    state,
    mechanism,
    memory,
    unseen: Sequence[int],
    lattice: np.ndarray,
    q: int,
    stride: int = 8,
    start: int = 0,
    ei_scale: np.ndarray | None = None,
    weight_selection: str = "stride",
) -> tuple[list[Proposal], list[dict[str, Any]]]:
    """Choose ``q`` distinct candidates, one weight vector each.

    The feasibility filter is the caller's job: ``unseen`` and ``ei_scale``
    are already the survivors of the batch-start veto. No random numbers.
    Same inputs twice produce the same proposals.
    """
    if q < 1:
        raise ValueError("q must be at least 1")
    if len(unseen) < q:
        raise ValueError(f"cannot pick {q} from {len(unseen)} unseen candidates")

    remaining = tuple(int(cid) for cid in unseen)
    remaining_mech = mechanism
    remaining_scale = None if ei_scale is None else np.asarray(ei_scale, dtype=float)
    proposals: list[Proposal] = []
    diagnostics: list[dict[str, Any]] = []

    for k in range(q):
        if weight_selection == "sequential":
            weights = acq.weight_for_round(lattice, int(state.round_index) + k)
        elif weight_selection == "stride":
            weights = lattice[(int(start) + k * int(stride)) % len(lattice)]
        else:
            raise ValueError(f"unknown weight_selection {weight_selection!r}")
        proposal = plan_one(
            state, remaining_mech, memory, remaining, weights, ei_scale=remaining_scale
        )
        if proposal.candidate_id in {p.candidate_id for p in proposals}:
            raise RuntimeError(
                "plan_batch selected the same candidate twice; the remaining "
                "pool was not updated"
            )
        proposals.append(proposal)
        diagnostics.append(
            {
                "within_batch_index": k + 1,
                "candidate_id": int(proposal.candidate_id),
                "weights": list(proposal.weights),
                "weight_selection": weight_selection,
                "expected_improvement": float(proposal.expected_improvement),
            }
        )
        remaining, remaining_mech, remaining_scale = _drop_candidate(
            remaining_mech, remaining, remaining_scale, proposal.candidate_id
        )
    return proposals, diagnostics
