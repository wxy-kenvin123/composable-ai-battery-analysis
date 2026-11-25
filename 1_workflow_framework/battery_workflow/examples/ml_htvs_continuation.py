"""
Case A continuation -- execute a second audited model on the published shortlist
================================================================================

The 56 compositions that survive the ML-HTVS-Li-SE cascade have no published
ionic-conductivity number. This run takes that shortlist, asks
``ProcessCoordinator`` to translate each composition into the input schema of
a second audited tool (ML-SSE), and **executes** that tool's released
RandomForest ensemble.

What this proves
----------------
Composition-level tool composition: one audited tool's survivors become
another audited tool's inputs, through the same OPEV driver, with a real
model call. The physical numbers are ML-SSE's predictions, not measurements
and not values this framework invented.

What this does not prove
------------------------
Prediction accuracy. ML-SSE never published scores for these 56 formulae.
The model is composition-only and cannot see crystal structure; the same
formula in a different structure can differ by orders of magnitude. Some
shortlist members sit outside the training element set and are flagged.
No accept/reject gate is applied: ML-SSE's notebooks do not state a
threshold, and this run will not invent one. The 1e-4 S/cm line is a field
convention used only as an annotation.

The original Case A replay (``ml_htvs_screening.py``) is unchanged.

Usage
-----
    python battery_workflow/examples/ml_htvs_continuation.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from battery_workflow.candidate_record import ComponentSpec
from battery_workflow.coordinator import Coordinator
from battery_workflow.datasets import ml_htvs, ml_sse
from battery_workflow.datasets._source import _sha256
from battery_workflow.examples import force_utf8_stdout
from battery_workflow.executors.ml_sse import CONVENTION_LOG10_S_CM, predict_formulas
from battery_workflow.modules.process_coordinator import ProcessCoordinator
from battery_workflow.modules.published_adapters import (
    ExecutedToolScreener,
    HtvsSurvivorCarrier,
)
from battery_workflow.run_record import environment, write_csv, write_json

PIPELINE = [
    ("PropertyScreener", "predict_material_properties"),
    ("ExecutedToolScreener", "predict_ionic_conductivity"),
]

HANDOFF_FIELDS = {
    "PropertyScreener->ExecutedToolScreener": {
        "formula": "composition",
    },
}

DEFAULT_OUTDIR = Path(__file__).resolve().parents[2] / "results" / "htvs_mlsse_continuation"


def build_coordinator(predictions, how: str, weight_sha256: str) -> Coordinator:
    coordinator = Coordinator(
        pipeline=PIPELINE,
        gate_map={"PropertyScreener": [], "ExecutedToolScreener": []},
        handoff_edges={"PropertyScreener": "ExecutedToolScreener"},
        deterministic=True,
    )
    coordinator.register_module("PropertyScreener", HtvsSurvivorCarrier())
    coordinator.register_module(
        "ExecutedToolScreener",
        ExecutedToolScreener(predictions, how, weight_sha256),
    )
    coordinator.register_module("ProcessCoordinator", ProcessCoordinator(HANDOFF_FIELDS))
    return coordinator


def run_ml_htvs_continuation(
    htvs_root: str | None = None,
    sse_root: str | None = None,
) -> dict:
    force_utf8_stdout()

    print("=" * 78)
    print("Case A continuation - execute ML-SSE on the published HTVS shortlist")
    print("=" * 78)

    htvs = ml_htvs.load(htvs_root)
    shortlist = list(htvs.published_survivors["electrochemical"])
    if len(shortlist) != 56:
        raise RuntimeError(f"expected 56 published survivors, got {len(shortlist)}")

    sse_source = ml_sse.SOURCE
    sse_source.verify(sse_root)
    sse_dir = sse_source.root(sse_root)
    weight_sha = sse_source.files["model.joblib"]
    observed = _sha256(sse_dir / "model.joblib")
    if observed != weight_sha:
        raise RuntimeError("ML-SSE model.joblib checksum mismatch")

    print(f"\nUpstream shortlist : {ml_htvs.SOURCE.tool} {ml_htvs.SOURCE.citation}")
    print(f"  compositions     : {len(shortlist)} (published electrochemical survivors)")
    print(f"Second tool        : {sse_source.tool} {sse_source.citation}")
    print(f"  repository       : {sse_source.repository} @ {sse_source.commit[:8]}")
    print(f"  model            : model.joblib @ {weight_sha[:12]}...")
    print(f"  licence          : {sse_source.licence}")
    print("  no accept/reject gate; 1e-4 S/cm is annotation only.")

    training_elements = ml_sse.training_elements(sse_dir)
    rows, how = predict_formulas(sse_dir, shortlist, training_elements)
    by_formula = {row.formula: row for row in rows}
    print(f"  execution        : {how}")
    print(f"  training elements: {len(training_elements)}")

    coordinator = build_coordinator(by_formula, how, weight_sha)
    ranked: list[dict] = []
    ood_count = 0

    for composition in shortlist:
        coordinator.create_candidate(
            candidate_id=f"htvs-cont:{composition}",
            system_name=composition,
            components=[ComponentSpec(role="electrolyte", composition=composition)],
            test_conditions={"continuation_of": "ml_htvs_screening"},
        )
        coordinator.run(
            f"htvs-cont:{composition}",
            objective="execute a second conductivity model on a published shortlist member",
            max_iterations=1,
        )
        prediction = by_formula[composition]
        ood_count += int(bool(prediction.ood_elements))
        ranked.append({
            "composition": composition,
            "log10_sigma": prediction.log10_sigma,
            "sigma_S_cm": prediction.sigma_S_cm,
            "above_convention_1e4": prediction.above_convention_1e4,
            "ood_elements": ",".join(prediction.ood_elements),
            "execution": True,
        })

    ranked.sort(key=lambda row: row["log10_sigma"], reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index

    n_above = sum(1 for row in ranked if row["above_convention_1e4"])
    trace = coordinator.agent_trace
    executed_steps = [s for s in trace if s.get("execution") is True]
    handoffs = [s for s in trace if s.get("kind") == "handoff"]

    print("\n" + "-" * 78)
    print("RANKED SHORTLIST  (executed ML-SSE log10 ionic conductivity)")
    print("-" * 78)
    print(f"{'rank':<6}{'composition':<22}{'log10(S/cm)':>12}{'S/cm':>14}{'>=1e-4?':>8}{'OOD':>8}")
    for row in ranked[:10]:
        print(
            f"{row['rank']:<6}{row['composition']:<22}"
            f"{row['log10_sigma']:>12.3f}{row['sigma_S_cm']:>14.3e}"
            f"{('yes' if row['above_convention_1e4'] else 'no'):>8}"
            f"{(row['ood_elements'] or '-'):>8}"
        )
    print(f"  ... {len(ranked) - 10} more")
    print(f"\n  above convention 1e-4 S/cm : {n_above} / {len(ranked)}  "
          f"(annotation only; threshold={CONVENTION_LOG10_S_CM})")
    print(f"  flagged out-of-training-elements : {ood_count} / {len(ranked)}")

    print("\n" + "-" * 78)
    print("AGENT TRACE")
    print("-" * 78)
    print(f"  {len(trace)} recorded steps: "
          f"{sum(1 for s in trace if s['kind'] == 'agent')} agent invocations, "
          f"including {len(executed_steps)} ExecutedToolScreener model executions, "
          f"plus {len(handoffs)} ProcessCoordinator field handoffs")
    print(f"  every ExecutedToolScreener step used the handed-off formula")
    print(f"  plan source : {coordinator.plan_source} (deterministic={coordinator.deterministic})")
    print("  no gate was applied.")

    print("\n" + "=" * 78)
    print("This ranking is an executed second-tool prediction, not a measured")
    print("conductivity and not an independent validation of ML-SSE.")
    print("=" * 78)

    return {
        "source_htvs": ml_htvs.SOURCE.provenance(htvs.root),
        "source_ml_sse": sse_source.provenance(sse_dir),
        "n_shortlist": len(shortlist),
        "ranked": ranked,
        "n_above_convention": n_above,
        "n_ood": ood_count,
        "convention_log10_S_cm": CONVENTION_LOG10_S_CM,
        "execution_how": how,
        "weight_sha256": weight_sha,
        "training_elements": sorted(training_elements),
        "coordinator": coordinator,
        "agent_trace": trace,
        "agents_invoked": sorted({s["agent"] for s in trace if s["kind"] == "agent"}),
        "handoffs": len(handoffs),
        "executed_steps": len(executed_steps),
        "mkb": coordinator.mkb.snapshot(),
    }


def export_continuation(outdir: Path | None = None, **kwargs) -> dict:
    """Write the executed-chain record next to, not over, the replay evidence."""
    case = run_ml_htvs_continuation(**kwargs)
    dest = Path(outdir) if outdir else DEFAULT_OUTDIR
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / "run_record_htvs_mlsse_continuation.json", {
        "schema": "battery_workflow.run_record/3",
        "case": "ml_htvs_mlsse_continuation",
        "generated_utc": None,
        "environment": environment(),
        "objective": "Execute a second audited conductivity model on the published "
                     "ML-HTVS-Li-SE shortlist through ProcessCoordinator field translation",
        "source_htvs": case["source_htvs"],
        "source_ml_sse": case["source_ml_sse"],
        "n_shortlist": case["n_shortlist"],
        "n_above_convention": case["n_above_convention"],
        "n_ood": case["n_ood"],
        "convention_log10_S_cm": case["convention_log10_S_cm"],
        "execution_how": case["execution_how"],
        "weight_sha256": case["weight_sha256"],
        "agents_invoked": case["agents_invoked"],
        "trace_steps": len(case["agent_trace"]),
        "field_handoffs": case["handoffs"],
        "executed_steps": case["executed_steps"],
        "mkb_snapshot": case["mkb"],
        "ranked": case["ranked"],
        "provenance": {
            "class": "executed",
            "metric_origin": (
                "log10 ionic conductivity predicted by executing ML-SSE "
                f"model.joblib @ {case['weight_sha256']}; not measured, "
                "not looked up, not a published score for these 56 formulae"
            ),
            "threshold_origin": (
                "no accept/reject gate; 1e-4 S/cm is a field convention used "
                "only as an annotation, not a published ML-SSE or ML-HTVS criterion"
            ),
            "execution": (
                f"{case['execution_how']}; ProcessCoordinator translated "
                "composition -> formula; ExecutedToolScreener refused to read "
                "the composition locally"
            ),
            "limitations": (
                "composition-level model, crystal structure invisible; "
                "out-of-training-set elements flagged; ranking is not a claim "
                "of predictive accuracy"
            ),
            "evidence_scope": (
                "cross-tool composition and real model execution only"
            ),
            "does_not_change": "ml_htvs_screening replay remains a lookup of published rows",
        },
    })
    write_csv(dest / "table_htvs_mlsse_ranked.csv", case["ranked"])
    write_csv(dest / "table_htvs_mlsse_agent_trace.csv", case["agent_trace"])
    print(f"\nexported to {dest}")
    return case


def main() -> int:
    export_continuation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
