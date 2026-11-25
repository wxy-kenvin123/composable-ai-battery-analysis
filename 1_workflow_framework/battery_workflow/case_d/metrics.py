"""Pareto fronts, hypervolume and regret over the evaluation budget."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def pareto_mask(points: np.ndarray) -> np.ndarray:
    """Non-dominated mask for minimisation."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = np.all(points <= points[i], axis=1) & np.any(points < points[i], axis=1)
        if dominated.any():
            keep[i] = False
    return keep


def hypervolume3d(points: np.ndarray, reference: Sequence[float]) -> float:
    """Exact 3-D dominated hypervolume for minimisation, bounded by ``reference``.

    Sweeps the third coordinate in ascending order and accumulates the area of
    the two-dimensional non-dominated staircase in the first two, which is the
    standard slicing construction and is exact rather than sampled.
    """
    reference = np.asarray(reference, dtype=float)
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return 0.0
    points = points[np.all(points < reference, axis=1)]
    if len(points) == 0:
        return 0.0

    points = points[np.argsort(points[:, 2], kind="stable")]
    total = 0.0
    active: list[tuple[float, float]] = []
    i = 0
    n = len(points)
    while i < n:
        z = points[i][2]
        while i < n and points[i][2] == z:
            active.append((points[i][0], points[i][1]))
            i += 1
        active.sort(key=lambda p: (p[0], p[1]))
        staircase: list[tuple[float, float]] = []
        best_y = math.inf
        for x, y in active:
            if y < best_y:
                staircase.append((x, y))
                best_y = y
        active = staircase

        area = 0.0
        prev_y = None
        for x, y in active:
            if prev_y is None:
                area += (reference[0] - x) * (reference[1] - y)
            else:
                area += (reference[0] - x) * (prev_y - y)
            prev_y = y

        z_next = points[i][2] if i < n else reference[2]
        total += area * (min(z_next, reference[2]) - z)
        if z_next >= reference[2]:
            break
    return float(total)


def hypervolume2d(points: np.ndarray, reference: Sequence[float]) -> float:


    reference = np.asarray(reference, dtype=float)
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return 0.0
    points = points[np.all(points < reference, axis=1)]
    if len(points) == 0:
        return 0.0

    ordered = sorted((float(p[0]), float(p[1])) for p in points)
    staircase: list[tuple[float, float]] = []
    best_y = math.inf
    for x, y in ordered:
        if y < best_y:
            staircase.append((x, y))
            best_y = y

    total = 0.0
    prev_y = None
    for x, y in staircase:
        if prev_y is None:
            total += (reference[0] - x) * (reference[1] - y)
        else:
            total += (reference[0] - x) * (prev_y - y)
        prev_y = y
    return float(total)


@dataclass(frozen=True)
class TruthSet:
    ids: np.ndarray
    points: np.ndarray
    reference: np.ndarray
    hv_true: float
    pareto_ids: frozenset
    top1pct_ids: frozenset

    @property
    def size(self) -> int:
        return len(self.ids)


def build_truth_set(evaluator: Any) -> TruthSet:
    """Recompute the truth set from the snapshot, then check it against the contract."""
    rows = evaluator.truth_rows()
    ids = np.array([r[0] for r in rows], dtype=int)
    points = np.array([[r[1], r[2], r[3]] for r in rows], dtype=float)

    lower = points.min(axis=0)
    upper = points.max(axis=0)
    reference = np.round(upper + 0.05 * (upper - lower), 10)

    mask = pareto_mask(points)
    pareto_ids = frozenset(int(v) for v in ids[mask])
    hv_true = hypervolume3d(points, reference)

    normalised = (points - lower) / np.where(upper - lower > 0, upper - lower, 1.0)
    score = normalised.mean(axis=1)
    k = math.ceil(0.01 * len(ids))
    order = np.lexsort((ids, score))
    top1pct_ids = frozenset(int(ids[j]) for j in order[:k])

    truth = TruthSet(
        ids=ids,
        points=points,
        reference=reference,
        hv_true=hv_true,
        pareto_ids=pareto_ids,
        top1pct_ids=top1pct_ids,
    )
    verify_frozen_constants(truth, evaluator.frozen_constants)
    return truth


def verify_frozen_constants(truth: TruthSet, frozen: dict[str, Any]) -> None:
    problems: list[str] = []
    if truth.size != frozen["truth_set_size"]:
        problems.append(f"truth set size {truth.size} != {frozen['truth_set_size']}")
    if [round(v, 10) for v in truth.reference] != list(frozen["reference_point"]):
        problems.append(
            f"reference point {list(truth.reference)} != {frozen['reference_point']}"
        )
    if abs(truth.hv_true - frozen["hv_true"]) > float(frozen["hv_true_tolerance"]):
        problems.append(f"hv_true {truth.hv_true!r} != {frozen['hv_true']}")
    if sorted(truth.pareto_ids) != list(frozen["pareto_ids"]):
        problems.append("Pareto id set differs from the frozen one")
    if sorted(truth.top1pct_ids) != list(frozen["top1pct_ids"]):
        problems.append("top-1% id set differs from the frozen one")

    payload = {
        "truth_set_size": truth.size,
        "reference_point": [float(v) for v in truth.reference],
        "hv_true": round(truth.hv_true, 9),
        "pareto_ids": sorted(truth.pareto_ids),
        "top1pct_k": len(truth.top1pct_ids),
        "top1pct_ids": sorted(truth.top1pct_ids),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if digest != frozen["derived_constants_sha256"]:
        problems.append(
            f"derived-constants digest {digest} != {frozen['derived_constants_sha256']}"
        )

    if problems:
        raise RuntimeError(
            "The label-derived constants recomputed from the snapshot do not match the "
            "fixed configuration. Scoring is refused.\n  " + "\n  ".join(problems)
        )


def readout_sensitivity(
    trajectories: Sequence[dict[str, Any]],
    reference_arm: str,
    comparators: Sequence[str],
    budgets: Sequence[int],
    resamples: int,
    rng_seed: int,
) -> list[dict[str, Any]]:
    """Paired bootstrap of the regret difference at several read-out budgets.

    A diagnostic, not a result. The registered read-out is the exhausted
    budget; a single-point read-out taken after every arm has converged can hide
    differences that were present mid-campaign, so this table shows where in the
    budget the registered conclusion does and does not hold.
    """
    by_key: dict[tuple[str, int, int], float] = {}
    seeds: set[int] = set()
    for row in trajectories:
        by_key[(row["arm"], int(row["seed"]), int(row["budget_used"]))] = float(
            row["hv_regret_normalized"]
        )
        seeds.add(int(row["seed"]))
    ordered_seeds = sorted(seeds)

    rng = np.random.default_rng(rng_seed)
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        for arm in comparators:
            diffs = np.array(
                [
                    by_key[(arm, seed, budget)] - by_key[(reference_arm, seed, budget)]
                    for seed in ordered_seeds
                ],
                dtype=float,
            )
            draws = rng.integers(0, len(diffs), size=(resamples, len(diffs)))
            means = diffs[draws].mean(axis=1)
            lower, upper = np.percentile(means, [2.5, 97.5])
            rows.append(
                {
                    "read_out_budget": budget,
                    "comparator": arm,
                    "reference": reference_arm,
                    "n_seeds": len(ordered_seeds),
                    "mean_difference": round(float(diffs.mean()), 9),
                    "ci95_lower": round(float(lower), 9),
                    "ci95_upper": round(float(upper), 9),
                    "interval_excludes_zero": bool(lower > 0.0),
                    "is_contract_readout": budget == max(budgets),
                    "source": (
                        "post-hoc diagnostic over the frozen trajectories; the "
                        "registered read-out is the exhausted budget and is unchanged"
                    ),
                }
            )
    return rows


def score_campaign(
    campaign: Any, evaluator: Any, truth: TruthSet, hv_fn: Any = None
) -> dict[str, Any]:


    if hv_fn is None:
        hv_fn = hypervolume3d
    if not campaign.finished:
        raise RuntimeError("score_campaign() requires a campaign that has been finished")

    order = [r["candidate_id"] for r in campaign.rounds]
    usable = [
        r["candidate_id"] for r in campaign.rounds if r["status"] == "ok"
    ]
    index = {int(v): i for i, v in enumerate(truth.ids)}
    observed_points = np.array(
        [truth.points[index[cid]] for cid in usable if cid in index], dtype=float
    )

    hv_observed = hv_fn(observed_points, truth.reference)
    hv_regret = (truth.hv_true - hv_observed) / truth.hv_true

    found_pareto = truth.pareto_ids.intersection(usable)
    recall = len(found_pareto) / len(truth.pareto_ids)

    first_hit = None
    for position, cid in enumerate(order, start=1):
        if cid in truth.top1pct_ids:
            first_hit = position
            break

    trajectory = []
    running: list[np.ndarray] = []
    for position, cid in enumerate(order, start=1):
        if cid in index and cid in set(usable):
            running.append(truth.points[index[cid]])
        hv = hv_fn(np.array(running), truth.reference) if running else 0.0
        trajectory.append(
            {
                "arm": campaign.arm.name,
                "seed": campaign.seed,
                "budget_used": position,
                "hv_observed": round(hv, 9),
                "hv_regret_normalized": round((truth.hv_true - hv) / truth.hv_true, 9),
                "source": "scored after the oracle was frozen",
            }
        )

    summary = campaign.summary()
    summary.update(
        {
            "hv_true": round(truth.hv_true, 9),
            "hv_observed": round(hv_observed, 9),
            "hv_regret_normalized": round(hv_regret, 9),
            "pareto_recall": round(recall, 9),
            "pareto_front_size": len(truth.pareto_ids),
            "pareto_found": sorted(int(v) for v in found_pareto),
            "first_hit_top1pct": first_hit,
            "hit_top1pct": first_hit is not None,
            "truth_set_size": truth.size,
            "truth_set_note": (
                "the true front is defined on the 1,867 fully-labelled candidates, not "
                "on all 1,872; five have no ionic-conductivity label"
            ),
        }
    )
    return {"summary": summary, "trajectory": trajectory}
