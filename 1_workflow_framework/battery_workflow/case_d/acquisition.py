"""Expected improvement, deterministic designs and simplex weight schedules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

_ERF = np.frompyfunc(math.erf, 1, 1)
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


def normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _ERF(np.asarray(z, dtype=float) * _INV_SQRT2).astype(float))


def normal_pdf(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return _INV_SQRT2PI * np.exp(-0.5 * z * z)


def simplex_lattice(divisions: int, dimension: int = 3) -> np.ndarray:


    if dimension not in (2, 3):
        raise NotImplementedError(
            "The lattice supports two or three objectives. "
            f"dimension={dimension!r} is neither."
        )
    rows = []
    if dimension == 2:
        for i in range(divisions + 1):
            rows.append((i / divisions, (divisions - i) / divisions))
        return np.array(rows, dtype=float)
    for i in range(divisions + 1):
        for j in range(divisions - i + 1):
            k = divisions - i - j
            rows.append((i / divisions, j / divisions, k / divisions))
    return np.array(rows, dtype=float)


def weight_for_round(lattice: np.ndarray, round_index: int) -> np.ndarray:
    """Weight vector for a 1-based optimised-round index."""
    if round_index < 1:
        raise ValueError("round_index is 1-based")
    return lattice[(round_index - 1) % len(lattice)]


def stride_weights(
    lattice: np.ndarray, q: int, stride: int = 8, start: int = 0
) -> np.ndarray:


    if q < 1:
        raise ValueError("q must be at least 1")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if start < 0:
        raise ValueError("start must be non-negative")
    n = len(lattice)
    return np.array([lattice[(start + k * stride) % n] for k in range(q)], dtype=float)


def min_max_normalise(
    values: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Scale to [0, 1] with a pre-computed range; a zero range maps to zeros."""
    span = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    span = np.where(span > 0.0, span, 1.0)
    return (np.asarray(values, dtype=float) - lower) / span


def augmented_chebyshev(
    normalised: np.ndarray, weights: np.ndarray, rho: float
) -> np.ndarray:
    """Augmented Chebyshev scalarisation for minimisation.

    ``normalised`` is (n, 3) already on [0, 1]; the result is (n,).
    """
    weighted = np.asarray(normalised, dtype=float) * np.asarray(weights, dtype=float)
    return weighted.max(axis=1) + rho * weighted.sum(axis=1)


def expected_improvement(
    mean: np.ndarray, variance: np.ndarray, incumbent: float, xi: float
) -> np.ndarray:
    """Expected improvement for minimisation; zero where the posterior is exact."""
    mean = np.asarray(mean, dtype=float)
    sigma = np.sqrt(np.maximum(np.asarray(variance, dtype=float), 0.0))
    improvement = incumbent - mean - xi
    out = np.zeros_like(mean)
    live = sigma > 0.0
    z = np.zeros_like(mean)
    z[live] = improvement[live] / sigma[live]
    out[live] = improvement[live] * normal_cdf(z[live]) + sigma[live] * normal_pdf(z[live])
    # A point the model is certain about can still be an improvement in the
    # noiseless limit; EI collapses to the improvement itself, floored at zero.
    out[~live] = np.maximum(improvement[~live], 0.0)
    return np.maximum(out, 0.0)


@dataclass(frozen=True)
class ArgmaxResult:
    index: int
    value: float
    tied: bool
    n_tied: int


def argmax_first(values: np.ndarray) -> ArgmaxResult:
    """Largest value, ties resolved to the lowest index.

    Candidate arrays are held in ascending id order, so "lowest index" is the
    contract's "smallest id" rule. The tie count travels with the answer instead
    of being discarded, because an unreported tie is how a reproduction quietly
    stops reproducing.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("no candidates to choose from")
    best = values.max()
    where = np.flatnonzero(values == best)
    return ArgmaxResult(
        index=int(where[0]),
        value=float(best),
        tied=bool(len(where) > 1),
        n_tied=int(len(where)),
    )


def greedy_maximin(
    features: np.ndarray, n_points: int, seed: int
) -> tuple[list[int], dict]:
    """Indices of a greedy maximin design over ``features``.

    The seed picks only the first point. Everything after that is the candidate
    whose nearest already-chosen neighbour is furthest away, ties to the lowest
    index. Two arms sharing a seed therefore share the whole initial design, not
    just its first member.
    """
    features = np.asarray(features, dtype=float)
    n_candidates = len(features)
    if not 0 < n_points <= n_candidates:
        raise ValueError(f"cannot take {n_points} of {n_candidates} candidates")

    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n_candidates))
    chosen = [first]

    diff = features - features[first]
    nearest = np.sqrt((diff * diff).sum(axis=1))
    ties = 0
    while len(chosen) < n_points:
        nearest[chosen] = -1.0
        pick = argmax_first(nearest)
        ties += int(pick.tied)
        chosen.append(pick.index)
        diff = features - features[pick.index]
        distance = np.sqrt((diff * diff).sum(axis=1))
        nearest = np.minimum(nearest, distance)

    return chosen, {
        "first_index": first,
        "rng": f"numpy.random.default_rng({seed}).integers(0, {n_candidates})",
        "ties_encountered": ties,
    }


def weight_schedule_report(lattice: np.ndarray, n_rounds: int) -> dict:


    used = np.array([weight_for_round(lattice, r) for r in range(1, n_rounds + 1)])
    return {
        "lattice_size": int(len(lattice)),
        "rounds": int(n_rounds),
        "distinct_vectors_visited": int(len(np.unique(used, axis=0))),
        "w1_conductivity": {
            "min": float(used[:, 0].min()),
            "max": float(used[:, 0].max()),
            "mean": round(float(used[:, 0].mean()), 6),
            "rounds_at_zero": int((used[:, 0] == 0).sum()),
        },
        "w2_li_diffusivity": {
            "min": float(used[:, 1].min()),
            "max": float(used[:, 1].max()),
            "mean": round(float(used[:, 1].mean()), 6),
            "rounds_at_zero": int((used[:, 1] == 0).sum()),
        },
        "w3_viscosity": {
            "min": float(used[:, 2].min()),
            "max": float(used[:, 2].max()),
            "mean": round(float(used[:, 2].mean()), 6),
            "rounds_at_zero": int((used[:, 2] == 0).sum()),
        },
        "limitation": (
            "the budget reaches only the first 32 of 66 lattice vectors, so w1 is "
            "capped at 0.3; shared by all arms, so between-arm comparisons are "
            "unaffected while absolute regret is biased for all of them alike"
        ),
    }


def weight_schedule_report_stride(
    lattice: np.ndarray,
    n_batches: int,
    q: int,
    stride: int,
    start: int,
) -> dict:

    rows = []
    for _ in range(n_batches):
        rows.append(stride_weights(lattice, q, stride=stride, start=start))
    n_objectives = int(lattice.shape[1]) if lattice.ndim == 2 else 3
    used = np.vstack(rows) if rows else np.zeros((0, n_objectives), dtype=float)
    report = {
        "lattice_size": int(len(lattice)),
        "n_batches": int(n_batches),
        "q": int(q),
        "stride": int(stride),
        "start": int(start),
        "selections": int(len(used)),
        "distinct_vectors_visited": int(len(np.unique(used, axis=0))) if len(used) else 0,
        "w1_conductivity": {
            "min": float(used[:, 0].min()) if len(used) else None,
            "max": float(used[:, 0].max()) if len(used) else None,
            "mean": round(float(used[:, 0].mean()), 6) if len(used) else None,
            "rounds_at_zero": int((used[:, 0] == 0).sum()) if len(used) else 0,
        },
        "w2_transference_number" if n_objectives == 2 else "w2_li_diffusivity": {
            "min": float(used[:, 1].min()) if len(used) else None,
            "max": float(used[:, 1].max()) if len(used) else None,
            "mean": round(float(used[:, 1].mean()), 6) if len(used) else None,
            "rounds_at_zero": int((used[:, 1] == 0).sum()) if len(used) else 0,
        },
        "limitation": (
            "Each batch takes lattice["
            f"{start} + k*{stride}] for k=0..{q - 1}; sequential take is not used. "
            "w1 coverage is the check that the batch did not collapse to the "
            "all-zero-w1 corner of the lexicographic lattice"
        ),
    }


    if n_objectives >= 3:
        report["w3_viscosity"] = {
            "min": float(used[:, 2].min()) if len(used) else None,
            "max": float(used[:, 2].max()) if len(used) else None,
            "mean": round(float(used[:, 2].mean()), 6) if len(used) else None,
            "rounds_at_zero": int((used[:, 2] == 0).sum()) if len(used) else 0,
        }
    report["n_objectives"] = n_objectives
    return report


def observed_bounds(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column min and max over the observed rows only.

    Separated out and named so that it is obvious at every call site that the
    normalisation range comes from what has been revealed, never from the full
    label set. Using the full set here would be the quietest possible leak.
    """
    values = np.asarray(values, dtype=float)
    return values.min(axis=0), values.max(axis=0)
