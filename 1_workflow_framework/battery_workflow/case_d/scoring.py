"""Two-objective feasible fronts, regret trajectories and paired comparisons."""
from __future__ import annotations
import math
from typing import Any, Sequence
import numpy as np

from .metrics import TruthSet, pareto_mask, hypervolume2d, score_campaign
from .sharing import hard_fail_ids


def _truth_set_from_rows(rows: list[tuple[int, float, float]]) -> TruthSet:
    ids = np.array([r[0] for r in rows], dtype=int)
    points = np.array([[r[1], r[2]] for r in rows], dtype=float)

    lower = points.min(axis=0)
    upper = points.max(axis=0)
    reference = np.round(upper + 0.05 * (upper - lower), 10)

    mask = pareto_mask(points)
    pareto_ids = frozenset(int(v) for v in ids[mask])
    hv_true = hypervolume2d(points, reference)

    normalised = (points - lower) / np.where(upper - lower > 0, upper - lower, 1.0)
    score = normalised.mean(axis=1)
    k = math.ceil(0.01 * len(ids))
    order = np.lexsort((ids, score))
    top1pct_ids = frozenset(int(ids[j]) for j in order[:k])

    return TruthSet(
        ids=ids,
        points=points,
        reference=reference,
        hv_true=hv_true,
        pareto_ids=pareto_ids,
        top1pct_ids=top1pct_ids,
    )


def build_feasible_truth_set_2obj(
    evaluator: Any, hard_fail_ids: frozenset, expected_size: int
) -> TruthSet:
    """The achievable (sigma, t_plus) front, restricted to feasible rows."""
    rows = [r for r in evaluator.truth_rows_2obj() if int(r[0]) not in hard_fail_ids]
    if len(rows) != int(expected_size):
        raise RuntimeError(
            f"feasible truth set has {len(rows)} rows; the contract pinned {expected_size}"
        )
    return _truth_set_from_rows(rows)


def front_census_2obj(evaluator: Any, hard_fail_ids: frozenset) -> dict[str, Any]:

    rows = list(evaluator.truth_rows_2obj())
    ids = np.array([r[0] for r in rows], dtype=int)
    points = np.array([[r[1], r[2]] for r in rows], dtype=float)

    unconstrained = ids[pareto_mask(points)]
    blocked = [int(v) for v in unconstrained if int(v) in hard_fail_ids]
    feasible = build_feasible_truth_set_2obj(
        evaluator, hard_fail_ids, len(rows) - len(hard_fail_ids & set(int(i) for i in ids))
    )
    return {
        "unconstrained_front_size": int(len(unconstrained)),
        "unconstrained_front_ids": sorted(int(v) for v in unconstrained),
        "blocked_front_size": len(blocked),
        "blocked_front_ids": sorted(blocked),
        "feasible_front_size": len(feasible.pareto_ids),
        "feasible_front_ids": sorted(int(v) for v in feasible.pareto_ids),
    }


def _balanced_logit_auc(X: np.ndarray, y: np.ndarray) -> float:
    n, d = X.shape
    n_pos = int(np.sum(y == 1.0))
    n_neg = int(np.sum(y == 0.0))
    w = np.where(y == 1.0, n / (2.0 * n_pos), n / (2.0 * n_neg))
    A = np.hstack([np.ones((n, 1)), X])
    penalty = np.full(d + 1, 1.0)
    penalty[0] = 0.0
    beta = np.zeros(d + 1)
    for _ in range(100):
        z = A @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
        grad = A.T @ (w * (p - y)) + penalty * beta
        s = w * p * (1.0 - p)
        hess = (A * s[:, None]).T @ A + np.diag(penalty)
        hess[np.diag_indices_from(hess)] += 1e-10
        step = np.linalg.solve(hess, grad)
        beta = beta - step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    scores = 1.0 / (1.0 + np.exp(-np.clip(beta[0] + X @ beta[1:], -60.0, 60.0)))
    pos = scores[y == 1.0]
    neg = scores[y == 0.0]
    greater = float(np.mean(pos[:, None] > neg[None, :]))
    equal = float(np.mean(pos[:, None] == neg[None, :]))
    return greater + 0.5 * equal


def measure_inferability_2obj(
    snapshot: Any, protocols: Sequence[str]
) -> dict[str, dict[str, float]]:

    labelled = [r for r in snapshot._records if r.ionic_conductivity is not None]
    ids = [int(r.id) for r in labelled]
    x3 = np.array([r.minimize_form() for r in labelled], dtype=float)
    x2 = np.array([r.minimize_form_2obj() for r in labelled], dtype=float)
    out: dict[str, dict[str, float]] = {}
    for protocol in protocols:
        fail = hard_fail_ids(snapshot.contract, protocol)
        y = np.array([1.0 if cid in fail else 0.0 for cid in ids], dtype=float)
        out[protocol] = {
            "from_sigma_tplus": round(_balanced_logit_auc(x2, y), 9),
            "from_legacy_three": round(_balanced_logit_auc(x3, y), 9),
        }
    return out


def truth_sets(snapshot, protocols):
    oracle = snapshot.oracle()
    oracle.freeze()
    evaluator = snapshot.evaluator()
    evaluator.enable(oracle)
    cfg = snapshot.contract
    specs = {s["set"]: s for s in cfg["hard_failure_protocol"]["protocols"]}
    truths, census = {}, {}
    for p in protocols:
        ids = hard_fail_ids(cfg, p)
        truths[p] = build_feasible_truth_set_2obj(evaluator, ids, specs[p]["n_feasible_labelled"])
        census[p] = front_census_2obj(evaluator, ids)
        for key, value in cfg["front_census"].items():
            if census[p][key] != value:
                raise ValueError(f"{p} front census differs for {key}")
    return evaluator, truths, census


def hv_regret_auc(trajectory: Sequence[dict[str, Any]]) -> float:
    """Trapezoidal area under hv_regret_normalized over the budget, divided by 47.

    Normalising by the 47 unit intervals keeps the value on the same [0, 1]
    scale as the regret itself, so it reads as "average regret over the whole
    campaign, weighting every stage of the budget equally".
    """
    ordered = sorted(trajectory, key=lambda row: int(row["budget_used"]))
    budgets = np.array([int(row["budget_used"]) for row in ordered], dtype=float)
    regrets = np.array([float(row["hv_regret_normalized"]) for row in ordered], dtype=float)
    if len(budgets) < 2:
        raise ValueError("a trajectory needs at least two budget points for an area")
    return float(np.trapz(regrets, budgets) / (budgets[-1] - budgets[0]))


def log_hv_regret_auc(
    trajectory: Sequence[dict[str, Any]], floor: float = 1e-6
) -> tuple[float, int]:

    ordered = sorted(trajectory, key=lambda row: int(row["budget_used"]))
    budgets = np.array([int(row["budget_used"]) for row in ordered], dtype=float)
    regrets = np.array([float(row["hv_regret_normalized"]) for row in ordered], dtype=float)
    if len(budgets) < 2:
        raise ValueError("a trajectory needs at least two budget points for an area")
    n_clipped = int(np.sum(regrets <= float(floor)))
    logs = np.log10(np.maximum(regrets, float(floor)))
    value = float(np.trapz(logs, budgets) / (budgets[-1] - budgets[0]))
    return value, n_clipped


def score(campaign, evaluator, truth):
    scored = score_campaign(campaign, evaluator, truth, hv_fn=hypervolume2d)
    trajectory = scored["trajectory"]
    summary = scored["summary"]
    log_auc, clipped = log_hv_regret_auc(trajectory, campaign.contract["metrics"]["floor"])
    summary["hv_regret_auc"] = round(hv_regret_auc(trajectory), 9)
    summary["log_hv_regret_auc"] = round(log_auc, 9)
    summary["floor_clipped_points"] = int(clipped)
    for row in trajectory:
        row["protocol"] = campaign.protocol
        row["objectives"] = "sigma_tplus"
        row["source"] = campaign.contract["contract_tag"]
    return scored


def seed_summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    row = {
        "protocol": summary["protocol"],
        "arm": summary["arm"],
        "seed": summary["seed"],
        "world": summary["world"],
        "success_transfer": summary["success_transfer"],
        "void_own_failures": summary["void_own_failures"],
        "records_failures_for_successors": summary["records_failures_for_successors"],
        "failure_regime": summary["failure_regime"],
        "n_sessions": summary["n_sessions"],
        "per_session_budget": summary["per_session_budget"],
        "batch_size": summary["batch_size"],
        "weight_stride": summary["weight_stride"],
        "wasted_fraction": summary["wasted_fraction"],
        "cross_session_duplicate_attempts": summary["cross_session_duplicate_attempts"],
        "repeated_hard_fail_attempts": summary["repeated_hard_fail_attempts"],
        "hard_fail_attempts_total": summary["hard_fail_attempts_total"],
        "hard_fail_first_attempts": summary["hard_fail_first_attempts"],
        "numbered_attempts": summary["numbered_attempts"],
        "data_missing_attempts": summary["data_missing_attempts"],
        "within_session_duplicates": summary["within_session_duplicates"],
        "unique_candidates_attempted": summary["unique_candidates_attempted"],
        "hv_regret_auc": summary["hv_regret_auc"],
        "hv_regret_normalized_final": summary["hv_regret_normalized"],
        "pareto_recall_final": summary["pareto_recall"],
        "protocol_violations": summary["protocol_violations"],
        "hard_fail_first_attempts_later": summary["hard_fail_first_attempts_later"],
        "hard_fail_first_attempts_later_design": summary["hard_fail_first_attempts_later_design"],
        "hard_fail_first_attempts_later_optimised": summary[
            "hard_fail_first_attempts_later_optimised"
        ],
        "pof_active_batches": summary["pof_active_batches"],
        "unique_cells_attempted": summary["unique_cells_attempted"],
        "mean_batch_diversity": summary["mean_batch_diversity"],
        "pof_enabled": summary["pof_enabled"],
        "source": summary["source"],
    }
    for entry in summary["per_session"]:
        s = int(entry["session"])
        row[f"s{s}_cross_session_duplicates"] = entry["cross_session_duplicates"]
        row[f"s{s}_hard_fail"] = entry["hard_fail"]
        row[f"s{s}_numbered"] = entry["numbered"]
        row[f"s{s}_hard_fail_first"] = entry["hard_fail_first"]
        row[f"s{s}_hard_fail_first_design"] = entry["hard_fail_first_design"]
        row[f"s{s}_hard_fail_first_optimised"] = entry["hard_fail_first_optimised"]
        row[f"s{s}_pof_active_batches"] = entry["pof_active_batches"]
        row[f"s{s}_inherited_numbered"] = entry["inherited_numbered"]
        row[f"s{s}_folded_numbered"] = entry["folded_numbered"]
    for key in (
        "objectives",
        "log_hv_regret_auc",
        "floor_clipped_points",
        "hv_regret_normalized",
        "knowledge_pack",
        "knowledge_channel",
        "knowledge_combination",
    ):
        if key in summary:
            row[key] = summary[key]
    return row


def paired_bootstrap(differences: Sequence[float], resamples: int, rng_seed: int) -> dict[str, Any]:
    """Percentile interval for the mean paired difference, resampling seeds.

    The unit of resampling is the seed, not the round, because rounds within a
    campaign are anything but independent.
    """
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(rng_seed)
    n = len(values)
    draws = rng.integers(0, n, size=(resamples, n))
    means = values[draws].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {
        "n_pairs": n,
        "mean_difference": round(float(values.mean()), 9),
        "ci95_lower": round(float(lower), 9),
        "ci95_upper": round(float(upper), 9),
        "favours_reference": bool(lower > 0.0),
        "resamples": resamples,
        "rng_seed": rng_seed,
        "direction": (
            "comparator minus full_multiagent on hv_regret_normalized, so a positive "
            "value means the full method had less regret"
        ),
    }


def _paired(runs, protocol, left, right, metric, seeds):
    return [
        float(runs[(protocol, left, seed)]["summary"][metric])
        - float(runs[(protocol, right, seed)]["summary"][metric])
        for seed in seeds
    ]


def _yield_controlled(runs, protocol, left, right, seeds, metric) -> dict[str, Any]:
    """The contrast after correcting for the difference in numbered results."""

    def col(arm, key):
        return np.array([float(runs[(protocol, arm, s)]["summary"][key]) for s in seeds])

    d_metric = col(left, metric) - col(right, metric)
    d_n = col(left, "numbered_attempts") - col(right, "numbered_attempts")
    slopes = []
    for arm in (left, right):
        n, y = col(arm, "numbered_attempts"), col(arm, metric)
        if float(np.std(n)) > 0:
            slopes.append(float(np.polyfit(n, y, 1)[0]))
    slope = float(np.mean(slopes)) if slopes else 0.0
    return {
        "raw_mean_difference": round(float(d_metric.mean()), 9),
        "mean_numbered_difference": round(float(d_n.mean()), 6),
        "within_arm_slope": round(slope, 9),
        "yield_adjusted_mean_difference": round(float(d_metric.mean() - d_n.mean() * slope), 9),
    }
