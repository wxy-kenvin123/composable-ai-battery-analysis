"""
Case A -- re-deriving a published screening funnel through the agent loop
=========================================================================

Every one of the 438 candidates in the ML-HTVS-Li-SE release is driven through
``Coordinator.run()`` -- the same Observe-Plan-Execute-Validate driver the smoke
test uses -- with two agents backed by that project's own released tables:

    HTPredictor (PropertyScreener)     serves the ROOST predictions and the
                                       pymatgen stability windows
      -> htvs-thermodynamic   319 survivors  (02_1 cell[30]: E_form < 0.1, E_hull < 0.2)
      -> htvs-electronic      306 survivors  (02_2 cell[8]:  band gap > 0.5 eV)
      -> htvs-electrochemical  56 survivors  (02_3 cell[7]:  window > 1 V, range > 0.001 V)
    ProcessCoordinator                 hands the cleared properties across the
                                       stage boundary
    MechanismSimulator                 reports the structures the authors
                                       enumerated for the survivors

and the survivors after each gate are checked against the published output CSVs
as *composition sets*, not merely as counts.

This is deliberately a regression replay: inputs, thresholds and expected sets
all come from the same upstream release. It validates loading, gate plumbing and
set identity, not the scientific accuracy of ROOST or pymatgen.

Nothing here is computed by this framework and nothing is invented: the property
values are the published ROOST predictions and pymatgen stability windows, the
thresholds are literals in the published notebooks, and the expected survivor
lists are the published output files. What the framework contributes is the part
the audit found missing from all 68 tools -- carrying each candidate across
stage boundaries in a shared record, applying the gate, and writing structured
negative evidence for the ones that stop.

The run is deterministic and offline by construction: ``Coordinator`` is built
with ``deterministic=True``, which detaches the coordinator and every agent from
the LLM client, so a configured API key cannot change the funnel.

The dataset is not vendored (the upstream repository declares no licence). Fetch
it and point ``BATTAGENT_ML_HTVS_DIR`` at the clone; see ``datasets/ml_htvs.py``.

Usage
-----
    python battery_workflow/examples/ml_htvs_screening.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from battery_workflow.examples import force_utf8_stdout
from battery_workflow.candidate_record import ComponentSpec, Phase
from battery_workflow.coordinator import Coordinator
from battery_workflow.datasets import ml_htvs
from battery_workflow.modules.process_coordinator import ProcessCoordinator
from battery_workflow.modules.published_adapters import (
    HtvsPropertyScreener,
    HtvsStructureBuilder,
)
from pathlib import Path
import contextlib
import io
from battery_workflow.run_record import criterion_rows, environment, write_csv, write_json
from battery_workflow.validation_gates import ML_HTVS_CASCADE

#: Gate name -> the stage key used by the published survivor files.
_STAGE_OF_GATE = {
    "htvs-thermodynamic": "thermodynamic",
    "htvs-electronic": "electronic",
    "htvs-electrochemical": "electrochemical",
}

#: The agents this case runs, and the action the Coordinator dispatches to each.
PIPELINE = [
    ("PropertyScreener", "predict_material_properties"),
    ("MechanismSimulator", "enumerate_published_structures"),
]

#: Three material-level screens follow the screening agent; the structure stage
#: has no published acceptance criterion, so nothing follows it.
GATE_MAP = {
    "PropertyScreener": ML_HTVS_CASCADE,
    "MechanismSimulator": [],
}

#: What crosses the stage boundary: the screening properties the authors' next
#: stage consumes. Named for the published columns, not the fixture's fields.
HANDOFF_FIELDS = {
    "PropertyScreener->MechanismSimulator": {
        "screened_composition": "composition",
        "band_gap_eV": "roost_band_gap",
        "e_hull_eV_atom": "e_hull_from_roost",
        "window_upper_bound_V": "upper_bound",
    },
}


def build_coordinator(dataset) -> Coordinator:
    """Wire the agents for this case onto the standard OPEV driver."""
    coordinator = Coordinator(
        pipeline=PIPELINE,
        gate_map=GATE_MAP,
        handoff_edges={"PropertyScreener": "MechanismSimulator"},
        deterministic=True,
    )
    coordinator.register_module("PropertyScreener", HtvsPropertyScreener(dataset))
    coordinator.register_module("MechanismSimulator", HtvsStructureBuilder(dataset))
    coordinator.register_module("ProcessCoordinator", ProcessCoordinator(HANDOFF_FIELDS))
    return coordinator


def run_ml_htvs_screening(root: str | None = None) -> dict:
    force_utf8_stdout()

    print("=" * 78)
    print("Case A - ML-HTVS-Li-SE screening funnel, re-derived through the agent loop")
    print("=" * 78)

    dataset = ml_htvs.load(root)
    source = ml_htvs.SOURCE
    print(f"\nSource        : {source.tool} (audit paper_id {source.paper_id}) {source.citation}")
    print(f"  repository  : {source.repository} @ {source.commit[:8]}")
    print(f"  licence     : {source.licence}")
    print(f"  data root   : {dataset.root}")
    print(f"  candidates  : {len(dataset)}")
    print("  checksums verified against the pinned commit.")

    coordinator = build_coordinator(dataset)
    print("\nAgents (each returns published values, none computes anything):")
    for name, action in PIPELINE:
        module = coordinator.modules[name]
        print(f"  {module.agent:<20} {name}.{action}")
    print(f"  {'ProcessCoordinator':<20} field handoff "
          f"{'->'.join(list(coordinator.handoff_edges.items())[0])}")

    print("\nGates (thresholds are literals in the published notebooks):")
    for gate in ML_HTVS_CASCADE:
        for criterion in gate.criteria:
            print(f"  [{gate.name:<21}] {criterion.metric_name} "
                  f"{criterion.operator} {criterion.threshold} {criterion.unit}")
        print(f"  {'':<23} <- {gate.source}")

    # ----- Drive every candidate through the OPEV loop -----
    survivors: dict[str, list[str]] = {stage: [] for stage in _STAGE_OF_GATE.values()}
    rejected_at: dict[str, list[str]] = {stage: [] for stage in _STAGE_OF_GATE.values()}
    evaluations = {gate.name: 0 for gate in ML_HTVS_CASCADE}
    decisions: list[dict] = []
    structures_found = 0

    for entry in dataset.candidates:
        coordinator.create_candidate(
            candidate_id=entry.candidate_id,
            system_name=entry.pretty_formula or entry.composition,
            components=[ComponentSpec(role="electrolyte", composition=entry.composition)],
            test_conditions={"screening_source": source.key},
        )
        # A composition is a fixed object: there is no reformulation to retry, so
        # one pass. The feedback path is exercised by Case B.
        outcome = coordinator.run(
            entry.candidate_id,
            objective="screen a substitution-generated lithium solid electrolyte",
            max_iterations=1,
        )

        metrics = entry.metrics()
        for step in outcome["iterations"][0]["outcome"]["steps"]:
            if step.get("kind") != "gate":
                continue
            gate_name = step["gate"]
            stage = _STAGE_OF_GATE.get(gate_name)
            if stage is None:
                continue
            evaluations[gate_name] += 1
            gate = next(g for g in ML_HTVS_CASCADE if g.name == gate_name)
            decisions.extend(criterion_rows(
                gate, metrics,
                candidate_id=entry.candidate_id,
                composition=entry.composition,
                stage=stage,
            ))
            (survivors if step["passed"] else rejected_at)[stage].append(entry.composition)

        record = coordinator.mkb.get(entry.candidate_id)
        structure_stage = record.get_latest_metrics("MechanismSimulator")
        structures_found += int(bool(structure_stage.get("structure_data_published")))

    coordinator.mkb.sync()

    # ----- Compare against the published survivor lists -----
    print("\n" + "-" * 78)
    print("FUNNEL  (re-derived vs the survivor CSVs the authors released)")
    print("-" * 78)
    print(f"{'stage':<22}{'evaluated':>10}{'survived':>10}{'published':>11}{'set match':>11}")

    entering = len(dataset)
    mismatches: list[str] = []
    funnel: list[dict] = []
    for gate in ML_HTVS_CASCADE:
        stage = _STAGE_OF_GATE[gate.name]
        got = set(survivors[stage])
        want = set(dataset.published_survivors[stage])
        identical = got == want
        if not identical:
            mismatches.append(
                f"{stage}: {len(got - want)} extra, {len(want - got)} missing"
            )
        print(f"{stage:<22}{entering:>10}{len(got):>10}{len(want):>11}"
              f"{('yes' if identical else 'NO'):>11}")
        funnel.append({
            "stage": stage,
            "gate": gate.name,
            "entering": entering,
            "survived": len(got),
            "published_survivors": len(want),
            "rejected": len(rejected_at[stage]),
            "set_identical_to_published": identical,
            "source": gate.source,
        })
        entering = len(got)

    final = survivors["electrochemical"]
    print(f"\n  438 -> {funnel[0]['survived']} -> {funnel[1]['survived']} -> {len(final)}")

    # ----- What the staged ordering saved -----
    possible = len(dataset) * len(ML_HTVS_CASCADE)
    actual = sum(evaluations.values())
    print("\n" + "-" * 78)
    print("FAIL-FAST ACCOUNTING")
    print("-" * 78)
    for gate in ML_HTVS_CASCADE:
        skipped = len(dataset) - evaluations[gate.name]
        print(f"  {gate.name:<22} evaluated {evaluations[gate.name]:>4} of {len(dataset)}"
              f"   ({skipped} never reached it)")
    print(f"  {'total':<22} {actual} of {possible} gate evaluations "
          f"({possible - actual} avoided, {100 * (possible - actual) / possible:.1f}%)")
    print("  The published release contains exactly "
          f"{sum(1 for c in dataset.candidates if c.roost_band_gap is not None)} band gaps and "
          f"{sum(1 for c in dataset.candidates if c.upper_bound is not None)} stability windows,")
    print("  not 438 of each: the upstream pipeline stopped paying for a property once a")
    print("  cheaper filter had already rejected the candidate.")

    # ----- Where the published record stops -----
    print("\n" + "-" * 78)
    print("NEXT STAGE  (what the release contains for the survivors)")
    print("-" * 78)
    print(f"  {structures_found} of the {len(final)} survivors have enumerated structures in")
    print("  06_2_drop_duplicate.csv; for the rest the released record stops at the")
    print("  shortlist. The agent reports the absence rather than supplying a value.")

    # ----- Structured negative evidence -----
    print("\n" + "-" * 78)
    print("STRUCTURED NEGATIVE EVIDENCE (what the MKB retained about the rejects)")
    print("-" * 78)
    by_gate: dict[str, int] = {}
    for record in coordinator.mkb.archive.values():
        for failure in record.failure_log:
            by_gate[failure.gate_name] = by_gate.get(failure.gate_name, 0) + 1
    for gate_name, count in by_gate.items():
        print(f"  {gate_name:<22} {count:>4} rejections logged with reason and conditions")
    example = next(iter(coordinator.mkb.archive.values()))
    failure = example.failure_log[0]
    print(f"\n  example  : {example.candidate_id}")
    print(f"    gate   : {failure.gate_name} (phase {failure.phase.value})")
    print(f"    reason : {failure.failure_mechanism[:150]}")

    snapshot = coordinator.mkb.snapshot()
    print(f"\n  MKB      : {snapshot}")
    print(f"  {snapshot['active']} candidates remain in Core at phase "
          f"'{Phase.SIMULATION.value}' -- the shortlist the authors sent to structure")
    print("  generation and DFT; the rest are archived with their rejection reason.")

    # ----- What actually ran -----
    trace = coordinator.agent_trace
    agents_run = sorted({s["agent"] for s in trace if s["kind"] == "agent"})
    handoffs = sum(1 for s in trace if s["kind"] == "handoff")
    feedback = sum(1 for s in trace if s["kind"] == "feedback")
    print("\n" + "-" * 78)
    print("AGENT TRACE")
    print("-" * 78)
    print(f"  {len(trace)} recorded trace steps: "
          f"{sum(1 for s in trace if s['kind'] == 'agent')} agent invocations, "
          f"{handoffs} field handoffs, {sum(1 for s in trace if s['kind'] == 'gate')} gate decisions, "
          f"{feedback} feedback triggers")
    print(f"  agents invoked      : {', '.join(agents_run)}")
    print(f"  plan source         : {coordinator.plan_source} "
          f"(deterministic={coordinator.deterministic})")

    # ----- Verdict -----
    print("\n" + "=" * 78)
    if mismatches:
        print("MISMATCH against the published survivor lists:")
        for line in mismatches:
            print(f"  {line}")
        print("=" * 78)
    else:
        print("Every stage reproduced the published survivor list exactly (composition sets,")
        print("not just counts). No value in this case was produced by this framework.")
        print("=" * 78)

    return {
        "source": source.provenance(dataset.root),
        "n_candidates": len(dataset),
        "funnel": funnel,
        "decisions": decisions,
        "final_shortlist": sorted(final),
        "gate_evaluations": evaluations,
        "evaluations_possible": possible,
        "evaluations_actual": actual,
        "rejections_by_gate": by_gate,
        "structures_published": structures_found,
        "mkb": snapshot,
        "coordinator": coordinator,
        "agent_trace": trace,
        "agents_invoked": agents_run,
        "handoffs": handoffs,
        "feedback_events": feedback,
        "reproduced": not mismatches,
        "mismatches": mismatches,
    }


def export_ml_htvs(outdir: Path) -> tuple[dict, list[str]]:
    """Run the ML-HTVS screening replay and write the structured evidence files."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        case = run_ml_htvs_screening()
    console = buf.getvalue()
    with open(outdir / "console_ml_htvs_screening.txt", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(console)

    write_json(outdir / "run_record_ml_htvs.json", {
        "schema": "battery_workflow.run_record/3",
        "case": "ml_htvs_screening",
        "generated_utc": None,
        "environment": environment(),
        "objective": "Re-derive the published ML-HTVS-Li-SE screening funnel through "
                     "gates instantiated from that project's own thresholds",
        "source": case["source"],
        "funnel": case["funnel"],
        "gate_evaluations": case["gate_evaluations"],
        "evaluations_possible": case["evaluations_possible"],
        "evaluations_actual": case["evaluations_actual"],
        "criterion_rows": len(case["decisions"]),
        "rejections_by_gate": case["rejections_by_gate"],
        "mkb_snapshot": case["mkb"],
        "final_shortlist": case["final_shortlist"],
        "reproduced_published_survivors": case["reproduced"],
        "agents_invoked": case["agents_invoked"],
        "agent_steps": len(case["agent_trace"]),
        "trace_steps": len(case["agent_trace"]),
        "field_handoffs": case["handoffs"],
        "feedback_events": case["feedback_events"],
        "survivors_with_published_structures": case["structures_published"],
        "provenance": {
            "class": "published",
            "metric_origin": "ROOST formation-energy, hull and band-gap predictions and "
                             "pymatgen stability windows released by "
                             f"{case['source']['citation']} (doi:{case['source']['doi']}); "
                             "read from the pinned commit, not recomputed here",
            "threshold_origin": "literals in the upstream notebooks, quoted per criterion "
                                "with notebook, cell and line in threshold_source",
            "expected_output_origin": "the upstream survivor CSVs; the check is set "
                                      "identity of compositions, not just counts",
            "execution": "every candidate is dispatched through Coordinator.run(); the "
                         "agents serve published rows and the gates are applied by the "
                         "framework, but no upstream program is executed here",
            "validation_scope": "deterministic regression replay of released predictions, "
                                "thresholds and expected survivor sets; not an independent "
                                "validation of ROOST or pymatgen",
            "evidence_scope": "staged-gate plumbing, set identity and accounting only",
            "redistribution": "upstream declares no licence; data is fetched by the "
                              "reader and verified against files_sha256, never vendored",
        },
    })

    written = [
        write_csv(outdir / "table_htvs_funnel.csv", case["funnel"]),
        write_csv(outdir / "table_htvs_gate_decisions.csv", case["decisions"]),
        write_csv(outdir / "table_htvs_agent_trace.csv", case["agent_trace"]),
        write_csv(outdir / "table_htvs_shortlist.csv",
                  [{"rank": i, "composition": c}
                   for i, c in enumerate(case["final_shortlist"], start=1)]),
    ]
    return case, [p.name for p in written]


def main() -> int:
    outcome = run_ml_htvs_screening()
    return 0 if outcome["reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
