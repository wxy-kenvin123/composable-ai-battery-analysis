"""Run a record-sharing experiment on the fixed OEDB electrolyte snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from battery_workflow.case_d import BLAS_STATE
from battery_workflow.case_d import scoring
from battery_workflow.case_d.sharing import SharingCampaign, load_config
from battery_workflow.datasets.oedb import load_oedb_snapshot
from battery_workflow.run_record import environment, write_csv, write_json
import numpy as np


def parse_seeds(value):
    seeds = []
    for element in value.split(","):
        bounds = element.strip().split("-")
        if len(bounds) == 1:
            seeds.append(int(bounds[0]))
        elif len(bounds) == 2 and int(bounds[0]) <= int(bounds[1]):
            seeds.extend(range(int(bounds[0]), int(bounds[1]) + 1))
        else:
            raise ValueError(f"invalid seed range {element}")
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seeds")
    return seeds


def initialize(config, protocols):
    snapshot = load_oedb_snapshot(contract=config)
    evaluator, truths, census = scoring.truth_sets(snapshot, protocols)
    return snapshot, evaluator, truths, census


_worker_state = None


def init_worker(config, protocols):
    global _worker_state
    _worker_state = initialize(config, protocols)


def run_campaign(task, state):
    protocol, arm, seed = task
    snapshot, evaluator, truths, _ = state
    campaign = SharingCampaign(snapshot, arm, seed, protocol)
    campaign.run()
    measured = scoring.score(campaign, evaluator, truths[protocol])
    return task, {**measured, "rounds": campaign.rounds}


def run_worker(task):
    return run_campaign(task, _worker_state)


def comparisons(config, runs, protocols, seeds, inferability):
    block = config["paired_bootstrap"]
    result = []
    for protocol in protocols:
        for spec in block["comparisons"]:
            left, right, metric = spec["comparator"], spec["reference"], spec["metric"]
            diff = scoring._paired(runs, protocol, left, right, metric, seeds)
            stats = scoring.paired_bootstrap(
                diff, block["bootstrap_resamples"], block["bootstrap_rng_seed"]
            )
            stats.update(
                protocol=protocol,
                comparator=left,
                reference=right,
                metric=metric,
                meaning=spec["meaning"],
            )
            if config["comparison_gate_flag"]:
                stats["in_promotion_gate"] = bool(spec.get("in_promotion_gate", False))
            stats.update(
                group=spec.get("group", "report"),
                comparison_id=spec["id"],
                direction=f"{left} minus {right} on {metric}",
                role="report",
                inferability_auc=inferability[protocol]["from_sigma_tplus"],
            )
            if metric in ("log_hv_regret_auc", "hv_regret_auc", "hv_regret_normalized"):
                stats.update(scoring._yield_controlled(runs, protocol, left, right, seeds, metric))
            result.append(stats)
    return result


def summarise_arms(config, runs, protocols, arms, seeds):
    fields = {
        "hard_fail_attempts_mean": "hard_fail_attempts_total",
        "cross_session_duplicates_mean": "cross_session_duplicate_attempts",
        "numbered_attempts_mean": "numbered_attempts",
        "unique_candidates_mean": "unique_candidates_attempted",
        "pof_active_batches_mean": "pof_active_batches",
        "pareto_recall_mean": "pareto_recall",
        "log_hv_regret_auc_mean": "log_hv_regret_auc",
        "hv_regret_auc_mean": "hv_regret_auc",
        "hv_regret_final_mean": "hv_regret_normalized",
    }
    rows = []
    for protocol in protocols:
        for arm in arms:
            row = dict(
                protocol=protocol,
                arm=arm,
                n_seeds=len(seeds),
                objectives="sigma_tplus",
                campaign_total_budget=config["sessions"]["campaign_total_budget"],
            )
            for label, metric in fields.items():
                vals = [runs[(protocol, arm, s)]["summary"].get(metric) for s in seeds]
                vals = [v for v in vals if v is not None]
                row[label] = round(float(np.mean(vals)), 9) if vals else None
            row["source"] = config["contract_tag"]
            rows.append(row)
    return rows


def check_budgets(config, budgets, sizes, clipped, allow_floor):
    sessions = config["sessions"]
    if budgets != [sessions["campaign_total_budget"]] or sizes != [
        sessions["per_session_budget"]["batch_size"]
    ]:
        raise ValueError("observed budgets or batch sizes differ from the configuration")
    if clipped and not allow_floor:
        raise ValueError(f"{clipped} evaluations reached the logarithmic floor")


def sequence(run):
    return [int(row["candidate_id"]) for row in run["rounds"]]


def compare_sequences(sequences, reference_dir):
    with (reference_dir / "run_record.json").open(encoding="utf-8") as f:
        reference = json.load(f)
    if reference["case"] != "ablation_ne_conductivity_scale":
        raise ValueError("expected the conductivity-prior ablation")
    results = {}
    for seed, ids in sequences.items():
        old = reference["full_shared_sequences"][seed]
        if ids != old:
            raise ValueError(f"full_shared selection differs for seed {seed}")
        results[seed] = dict(n_new=len(ids), n_old=len(old), identical=True, first_difference=None)
    return results


def run(
    config, outdir, seeds=None, protocols=None, workers=1, allow_floor=None, check_against=None
):
    seeds = config["seeds"] if seeds is None else list(seeds)
    protocols = config["protocols"] if protocols is None else list(protocols)
    if (
        not seeds
        or not protocols
        or len(set(protocols)) != len(protocols)
        or set(protocols) - set(config["protocols"])
    ):
        raise ValueError("expected seeds and configured protocols")
    if workers < 1:
        raise ValueError("workers must be at least one")
    if allow_floor is None:
        allow_floor = config["allow_log_floor"]
    arms = [a["name"] for a in config["arms"]]
    state = initialize(config, protocols)
    snapshot, _, truths, census = state
    inferability = scoring.measure_inferability_2obj(snapshot, protocols)
    tasks = [(p, a, s) for p in protocols for a in arms for s in seeds]
    if workers == 1:
        runs = dict(run_campaign(task, state) for task in tasks)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=init_worker, initargs=(config, protocols)
        ) as pool:
            runs = dict(pool.map(run_worker, tasks, chunksize=1))
    clipped = sum(int(r["summary"]["floor_clipped_points"]) for r in runs.values())
    budgets = sorted({len(r["rounds"]) for r in runs.values()})
    sizes = sorted({r["summary"]["batch_size"] for r in runs.values()})
    check_budgets(config, budgets, sizes, clipped, allow_floor)
    arm_table = summarise_arms(config, runs, protocols, arms, seeds)
    comparison_table = comparisons(config, runs, protocols, seeds, inferability)
    provenance = snapshot.provenance()
    provenance["resolved_root"] = "<BATTAGENT_OEDB_DIR>"
    rec = dict(
        schema="battery_workflow.run_record/5",
        case=config["name"],
        environment={**environment(), "executable": "<python>"},
        objective=config["purpose"],
        contract_tag=config["contract_tag"],
        config=config,
        seeds=seeds,
        arms=arms,
        protocols=protocols,
        campaign_total_budget=budgets[0],
        observed_batch_sizes=sizes,
        floor_clipped_points_total=clipped,
        front_census=census,
        truth_sizes={p: truths[p].size for p in protocols},
        feasible_front_sizes={p: len(truths[p].pareto_ids) for p in protocols},
        arm_comparison=arm_table,
        paired_bootstrap=comparison_table,
        blas=BLAS_STATE,
        workers=workers,
        source_provenance=provenance,
    )
    full_sequences = {str(s): sequence(runs[(protocols[0], "full_shared", s)]) for s in seeds}
    if config["experiment"] == "sharing":
        rec["knowledge_pack"] = config["rule_pack"]["file"]
        rec["sharing_rules"] = config["contract_tag"]
        rec["selection_hashes"] = {
            f"{a}/{s}": hashlib.sha256(
                ",".join(map(str, sequence(runs[(p, a, s)]))).encode("ascii")
            ).hexdigest()
            for p, a, s in sorted(runs)
        }
        rec["identity_check"] = (
            compare_sequences(full_sequences, check_against) if check_against else None
        )
    else:
        rec.update(
            world=config["name"],
            objectives="sigma_tplus",
            observed_budgets=budgets,
            inferability_aucs=inferability,
            weight_schedule_report=next(iter(runs.values()))["summary"]["weight_schedule_report"],
            allow_log_floor=bool(allow_floor),
        )
        rec["sign_check"] = {"applicable": config["sign_check_applicable"]}
        if not config["sign_check_applicable"]:
            rec["sign_check"]["note"] = "module ablation; no sharing-ladder gate"
        else:
            rec["alternate_budget"] = False
        if config["name"] == "ablation_ne_conductivity_scale":
            rec["full_shared_sequences"] = full_sequences
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / "run_record.json", rec)
    write_csv(
        outdir / "seed_summary.csv",
        [scoring.seed_summary_row(r["summary"]) for r in runs.values()],
        lineterminator="\n",
    )
    write_csv(outdir / "arm_comparison.csv", arm_table, lineterminator="\n")
    write_csv(outdir / "comparisons.csv", comparison_table, lineterminator="\n")
    write_csv(
        outdir / "trajectory.csv",
        [row for r in runs.values() for row in r["trajectory"]],
        lineterminator="\n",
    )
    if config["experiment"] == "sharing":
        write_csv(
            outdir / "rounds.csv",
            [row for r in runs.values() for row in r["rounds"]],
            lineterminator="\n",
        )
    print(f"{config['name']}: {len(runs)} campaigns, {budgets[0]} evaluations per campaign")
    return rec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=parse_seeds)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--protocols", help="comma-separated protocol names")
    parser.add_argument("--allow-log-floor", action="store_true", default=None)
    parser.add_argument(
        "--check-against",
        type=Path,
        help="compare shared selections with a completed conductivity-prior ablation",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run(
        config,
        args.outdir or Path("results") / config["name"],
        args.seeds,
        args.protocols.split(",") if args.protocols else None,
        args.workers,
        args.allow_log_floor,
        args.check_against,
    )


if __name__ == "__main__":
    main()
