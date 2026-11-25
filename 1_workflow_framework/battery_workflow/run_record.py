"""
Serialize a completed workflow run into auditable evidence.

The coordinator keeps every stage result, gate decision, and failure record in
memory only, so a finished run leaves nothing anyone can inspect. This
module turns that in-memory state into a machine-readable run record plus flat
tables, which is the same "reusable research output" the audit asks of the
tools it scores.

Gate rows are reconstructed by re-evaluating each criterion against the metrics
that were recorded at the time, so the exported verdicts are the verdicts the
run actually produced rather than a separate calculation. Every row carries the
criterion's ``source``, so a reader can tell at a glance whether a threshold came
from a published tool, was derived from published data, or is synthetic.

``build_record`` takes its provenance block from the caller rather than assuming
one: the same serializer covers the published-data cases and the synthetic
control-flow smoke test, and those must never be described the same way.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .candidate_record import CandidateRecord, Phase
from .validation_gates import (
    CELL_GATE,
    INTERFACE_GATE,
    MATERIAL_GATE,
    GateCriterion,
    ValidationGate,
)

GATE_BY_MODULE: dict[str, ValidationGate] = {
    "PropertyScreener": MATERIAL_GATE,
    "MechanismSimulator": INTERFACE_GATE,
    "ExperimentFeedback": CELL_GATE,
}

GATE_LABEL = {
    Phase.SCREENING: "material",
    Phase.INTERFACE_VALIDATION: "interface",
    Phase.CELL_VALIDATION: "cell",
}


def _plain(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _variation(values: Any) -> float | None:
    if not hasattr(values, "__len__") or len(values) < 2:
        return None
    return (max(values) - min(values)) / max(abs(min(values)), 1e-9)


def _observed(value: Any, criterion: GateCriterion) -> tuple[Any, str]:
    """Return the scalar actually compared and a human-readable rendering."""
    if criterion.operator == "stable_within":
        var = _variation(value)
        return var, "n/a" if var is None else f"{var * 100:.1f}% spread over {len(value)} points"
    return value, "n/a" if value is None else f"{value}"


def _attainment(value: Any, criterion: GateCriterion) -> float | None:
    """How far the observation clears its threshold; >= 1.0 means the criterion passed.

    Larger-is-better criteria report observed/threshold, smaller-is-better ones
    report threshold/observed, so a single column is comparable across metrics.
    """
    thr = criterion.threshold
    try:
        if criterion.operator == ">=":
            return float(value) / float(thr) if thr else None
        if criterion.operator == "<=":
            return float(thr) / float(value) if value else None
        if criterion.operator == "stable_within":
            var = _variation(value)
            return float(thr) / var if var else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return None


def _rounds(candidate: CandidateRecord) -> list[list[Any]]:
    """Split the flat stage log into rounds; each round opens with KnowledgeMiner."""
    rounds: list[list[Any]] = []
    for result in candidate.stage_results:
        if result.module_name == "KnowledgeMiner" or not rounds:
            rounds.append([])
        rounds[-1].append(result)
    return rounds


def trace_rows(candidate: CandidateRecord) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    step = 0
    for round_no, results in enumerate(_rounds(candidate), start=1):
        for result in results:
            step += 1
            gate = GATE_BY_MODULE.get(result.module_name)
            gate_passed = ""
            if gate is not None:
                gate_passed = "PASS" if gate.evaluate(result.metrics)[0] else "FAIL"
            rows.append({
                "step": step,
                "round": round_no,
                "module": result.module_name,
                "phase_at_entry": result.phase.value,
                "module_passed": result.passed,
                "selected_tool": result.metrics.get("selected_tool", ""),
                "decision_source": result.metrics.get("decision_source")
                                   or result.metrics.get("backend", ""),
                "gate_applied": GATE_LABEL.get(gate.phase, "") if gate else "",
                "gate_verdict": gate_passed,
                "n_metrics": len(result.metrics),
                "n_artifacts": len(result.artifacts),
                "timestamp": result.timestamp,
            })
    return rows


def gate_rows(candidate: CandidateRecord) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for round_no, results in enumerate(_rounds(candidate), start=1):
        for result in results:
            gate = GATE_BY_MODULE.get(result.module_name)
            if gate is None:
                continue
            for criterion in gate.criteria:
                raw = result.metrics.get(criterion.metric_name)
                observed, rendered = _observed(raw, criterion)
                passed = gate._check(raw, criterion) if raw is not None else False
                ratio = _attainment(raw, criterion)
                rows.append({
                    "round": round_no,
                    "gate": GATE_LABEL.get(gate.phase, gate.phase.value),
                    "gate_name": gate.name,
                    "metric": criterion.metric_name,
                    "operator": criterion.operator,
                    "threshold": criterion.threshold,
                    "unit": criterion.unit,
                    "observed_raw": raw if not isinstance(raw, list) else ";".join(map(str, raw)),
                    "observed_compared": observed,
                    "observed_rendered": rendered,
                    "attainment_ratio": None if ratio is None else round(ratio, 4),
                    "verdict": "PASS" if passed else "FAIL",
                    "description": criterion.description,
                    "threshold_source": criterion.source,
                })
    return rows


def criterion_rows(
    gate: ValidationGate,
    metrics: dict[str, Any],
    **extra: Any,
) -> list[dict[str, Any]]:
    """One row per criterion for a single gate evaluation.

    The published-data cases evaluate one gate against hundreds of candidates
    rather than one candidate against a stage ladder, so they build their tables
    from this instead of from ``gate_rows``. ``extra`` carries whatever identifies
    the candidate in that case (composition, cell id, round number).
    """
    rows: list[dict[str, Any]] = []
    for criterion in gate.criteria:
        raw = metrics.get(criterion.metric_name)
        observed, rendered = _observed(raw, criterion)
        passed = gate._check(raw, criterion) if raw is not None else False
        ratio = _attainment(raw, criterion)
        rows.append({
            **extra,
            "gate_name": gate.name,
            "gate_phase": gate.phase.value,
            "metric": criterion.metric_name,
            "operator": criterion.operator,
            "threshold": criterion.threshold,
            "unit": criterion.unit,
            "observed": raw if not isinstance(raw, list) else ";".join(map(str, raw)),
            "observed_rendered": rendered,
            "attainment_ratio": None if ratio is None else round(ratio, 4),
            "verdict": "PASS" if passed else "FAIL",
            "threshold_source": criterion.source,
        })
    return rows


def tool_rows(candidate: CandidateRecord, registry: Any = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for round_no, results in enumerate(_rounds(candidate), start=1):
        for result in results:
            tool = result.metrics.get("selected_tool")
            if not tool:
                continue
            crl = rrl = repo = ""
            if registry is not None:
                match = next((t for t in registry.all_tools() if t.tool == tool), None)
                if match is not None:
                    crl, rrl, repo = match.crl, match.rrl, match.github_url
            rows.append({
                "round": round_no,
                "module": result.module_name,
                "selected_tool": tool,
                "crl": crl,
                "rrl": rrl,
                "repository": repo,
                "decision_source": result.metrics.get("decision_source", ""),
            })
    return rows


#: Provenance block for the synthetic control-flow smoke test. Stated in the
#: bluntest terms available, because the audit's most serious finding was a
#: constructed value being described as if it came from a published case.
SYNTHETIC_PROVENANCE: dict[str, str] = {
    "class": "synthetic",
    "metric_origin": "constructed in synthetic_controlflow_fixture.py to trigger a "
                     "specific pass/fail sequence; not measured, not computed, not "
                     "predicted, and not usable as a result",
    "threshold_origin": "conventional engineering targets hard-coded in "
                        "validation_gates.py; not taken from any published tool",
    "evidence_scope": "no -- control flow only",
}


#: Wall-clock fields. They record when a run happened, which is the one thing
#: about it that cannot be reproduced, so the exporter strips them and the
#: generation time is stated once in the manifest instead. Ordering survives in
#: the ``step`` column.
VOLATILE_KEYS = frozenset({"timestamp", "created_at", "updated_at", "generated_utc"})


def strip_volatile(obj: Any) -> Any:
    """Recursively drop wall-clock fields so an unchanged run exports identical bytes."""
    if isinstance(obj, dict):
        return {k: strip_volatile(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [strip_volatile(v) for v in obj]
    return obj


def environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def build_record(
    candidate: CandidateRecord,
    mkb_snapshot: dict[str, Any],
    objective: str,
    llm_backend: str,
    knowledge_backend: str,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "battery_workflow.run_record/2",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            **environment(),
            "llm_backend": llm_backend,
            "knowledge_backend": knowledge_backend,
        },
        "objective": objective,
        "candidate": _plain(candidate),
        "mkb_snapshot": mkb_snapshot,
        "trace": trace_rows(candidate),
        "gate_decisions": gate_rows(candidate),
        "tool_selection": tool_rows(candidate),
        "provenance": dict(provenance or SYNTHETIC_PROVENANCE),
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    lineterminator: str = "\r\n",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = strip_volatile(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames: list[str] = []
    for row in rows:                       # trace rows differ in shape by step kind
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            restval="",
            lineterminator=lineterminator,
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(strip_volatile(payload), ensure_ascii=False, indent=2))
    return path
