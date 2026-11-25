"""
Schema-validated coordination decisions for the Orchestrator and the agents.

When an LLM is live, it genuinely drives four coordination decisions:
  - plan_pipeline      : order the agents into a screening -> validation pipeline
  - select_tool        : choose the most task-appropriate audited tool
  - translate_fields   : the cross-scale semantic meaning of a field handoff
  - attribute_failure  : interface side, mechanism, and next-round constraints

Every LLM response is parsed as JSON and validated. On any failure (no key, bad
JSON, out-of-set choice) a TASK-RELEVANT DETERMINISTIC policy is used instead, so
the no-key path is fully reproducible while the with-key path is genuinely
agentic and may differ. The LLM NEVER produces physical metric values -- those
come from the registry / case fixtures.
"""

from __future__ import annotations

import json
import re
from typing import Any


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #
def _parse_json(text: str) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start, depth = None, 0
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    start = None
    return None


def _ask_json(llm: Any, system: str, user: str) -> tuple[dict | None, str, dict]:
    """Return (parsed_json, source, raw_call). source in {deterministic, llm, llm-unparsed, error}."""
    if llm is None or not getattr(llm, "is_live", False):
        return None, "deterministic", {}
    res = llm.complete(system, user)
    raw = llm.last_call() if hasattr(llm, "last_call") else {}
    if not getattr(res, "is_live", False):
        return None, res.backend, raw
    data = _parse_json(res.text)
    return (data, "llm", raw) if data else (None, "llm-unparsed", raw)


# --------------------------------------------------------------------------- #
# Task-relevant scoring (deterministic baseline)
# --------------------------------------------------------------------------- #
_STOP = {
    "the", "a", "an", "of", "for", "and", "to", "in", "on", "with", "at", "by",
    "is", "are", "be", "as", "or", "from", "into", "via", "per", "its", "their",
    "this", "that", "these", "those", "system", "tool", "tools", "battery",
    "materials", "material", "model", "models", "data", "using", "based",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _STOP and len(t) > 1}


def _tool_text(tool: Any) -> str:
    return " ".join(str(getattr(tool, a, "")) for a in ("tool", "title", "main_input", "main_output"))


def score_tool(task_text: str, tool: Any) -> tuple[int, int, int]:
    """(keyword overlap with the task, CRL, RRL) -- higher is better."""
    overlap = len(_tokens(task_text) & _tokens(_tool_text(tool)))
    return (overlap, int(getattr(tool, "crl", 0)), int(getattr(tool, "rrl", 0)))


def select_tool(agent: str, task_text: str, system_name: str, tools: list, llm: Any) -> tuple[Any, dict]:
    """Pick the most task-appropriate audited tool for an agent."""
    if not tools:
        return None, {"source": "none", "reason": "no audited tool mapped to this agent (the 0/68 gap)"}

    full_task = f"{task_text} {system_name}"
    ranked = sorted(tools, key=lambda t: score_tool(full_task, t), reverse=True)
    det = ranked[0]
    det_overlap = score_tool(full_task, det)[0]

    by_id = {str(getattr(t, "paper_id", "")): t for t in tools}
    system = (
        f"You are the {agent} agent in a battery R&D workflow. Choose the single most "
        f"task-appropriate tool from the candidates. Respond with STRICT JSON only: "
        f'{{"paper_id": "<id>", "reason": "<one sentence>"}}. Do not invent ids or numbers.'
    )
    cand = "\n".join(
        f"- paper_id={getattr(t,'paper_id','')} name={getattr(t,'tool','')} "
        f"CRL={getattr(t,'crl',0)} RRL={getattr(t,'rrl',0)} "
        f"io=({getattr(t,'main_input','')} -> {getattr(t,'main_output','')})"
        for t in tools
    )
    user = f"Task: {task_text}\nCandidate system: {system_name}\nCandidates:\n{cand}"
    data, source, raw = _ask_json(llm, system, user)

    if data and str(data.get("paper_id")) in by_id:
        chosen = by_id[str(data.get("paper_id"))]
        return chosen, {
            "source": "llm",
            "reason": str(data.get("reason", ""))[:400],
            "deterministic_pick": det.tool,
            "llm_call": raw,
        }
    return det, {
        "source": "deterministic",
        "reason": f"task-relevant score (keyword overlap={det_overlap}) then CRL/RRL among {len(tools)} candidates",
        "llm_call": raw,
    }


# --------------------------------------------------------------------------- #
# Pipeline planning
# --------------------------------------------------------------------------- #
_ACTION = {
    "KnowledgeMiner": "retrieve_design_clues",
    "FormulationDesigner": "recommend_formulations",
    "PropertyScreener": "predict_material_properties",
    "MechanismSimulator": "run_interface_simulation",
    "ExperimentFeedback": "evaluate_cell_performance",
}
_DEFAULT_ORDER = ["KnowledgeMiner", "FormulationDesigner", "PropertyScreener",
                  "MechanismSimulator", "ExperimentFeedback"]
_GATED = ["PropertyScreener", "MechanismSimulator", "ExperimentFeedback"]


def plan_pipeline(
    objective: str,
    available: set[str],
    llm: Any,
    default_pipeline: list[tuple[str, str]] | None = None,
    gated: list[str] | None = None,
) -> tuple[list[tuple[str, str]], str]:
    """Order the available agents into a pipeline of (module, action) pairs.

    ``default_pipeline`` lets a caller supply a case-specific order and the
    actions to invoke, which is how the published-data cases run agents whose
    action names differ from the generic ones. The LLM may still reorder within
    the available set, but only if the gated stages it does contain stay in
    their relative order -- the guarantee that makes a reordered plan comparable
    to the deterministic one.
    """
    pipeline = default_pipeline or [(m, _ACTION[m]) for m in _DEFAULT_ORDER]
    det = [(m, a) for m, a in pipeline if m in available]
    actions = dict(det)
    must_hold = [m for m in (gated if gated is not None else _GATED) if m in actions]

    system = (
        "You are the Orchestrator. Order the available agents into a screening->validation "
        'pipeline. Respond with STRICT JSON only: {"order": ["AgentName", ...]} using only the '
        f"provided agent names. These stages must all be present and in this relative order: "
        f"{must_hold}."
    )
    user = f"Objective: {objective}\nAvailable agents: {sorted(available)}\nDefault order: {[m for m, _ in det]}"
    data, source, _ = _ask_json(llm, system, user)
    if data and isinstance(data.get("order"), list):
        order = [m for m in data["order"] if m in actions]
        if set(must_hold) <= set(order) and [m for m in order if m in must_hold] == must_hold:
            return [(m, actions[m]) for m in order], "llm"
    return det, "deterministic"


# --------------------------------------------------------------------------- #
# Field translation (cross-scale handoff)
# --------------------------------------------------------------------------- #
def translate_fields(source: str, target: str, source_metrics: dict, mapping: dict, llm: Any) -> tuple[dict, str, str]:
    """Apply the deterministic field protocol; the LLM supplies the cross-scale meaning."""
    fields = {tgt: source_metrics[src] for tgt, src in mapping.items() if src in source_metrics}
    note, src = "", "deterministic"
    if fields and llm is not None and getattr(llm, "is_live", False):
        system = (
            f"You are the Process Coordinator bridging {source} -> {target}. Given the mapped "
            'numeric fields, respond with STRICT JSON only: {"semantic_note": "<one sentence '
            'cross-scale meaning>"}. Do NOT change or invent any numbers.'
        )
        user = f"Mapped fields ({source}->{target}): {json.dumps(fields, default=str)}"
        data, source_kind, _ = _ask_json(llm, system, user)
        if data and data.get("semantic_note"):
            note, src = str(data["semantic_note"])[:300], "llm"
    return fields, note, src


# --------------------------------------------------------------------------- #
# Failure attribution
# --------------------------------------------------------------------------- #
def _infer_side(reasons: list[str], metrics: dict) -> str:
    text = (" ".join(reasons) + " " + str(metrics.get("attributed_to", "")) + " "
            + str(metrics.get("failure_mode", ""))).lower()
    if "cei" in text or "cathode" in text:
        return "cathode"
    if "sei" in text or "anode" in text:
        return "anode"
    return "bulk_electrolyte"


def _default_constraint(side: str) -> str:
    return {
        "cathode": "add cathode-side protection (e.g. fluorinated/inorganic filler) to suppress CEI growth",
        "anode": "improve anode interface stability (e.g. dendrite-suppressing layer)",
        "bulk_electrolyte": "raise bulk ionic transport / electrochemical-window margin",
    }.get(side, "tighten the failed acceptance criterion")


def attribute_failure(failure_reasons: list[str], metrics: dict, llm: Any) -> dict:
    side = _infer_side(failure_reasons, metrics)
    mechanism = "; ".join(failure_reasons)
    attributed = metrics.get("attributed_to") or metrics.get("failure_mode")
    if attributed:
        mechanism += f" | attributed_to={attributed}"
    result = {
        "interface_side": side,
        "mechanism": mechanism,
        "next_constraints": [_default_constraint(side)],
        "source": "deterministic",
    }
    if llm is not None and getattr(llm, "is_live", False):
        system = (
            "You are the failure-attribution coordinator. Given gate failure reasons and metrics, "
            'respond with STRICT JSON only: {"interface_side": "cathode|anode|bulk_electrolyte", '
            '"mechanism": "<short>", "next_constraints": ["<actionable constraint>"]}. Do NOT invent numbers.'
        )
        scalar = {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}
        user = f"Failure reasons: {failure_reasons}\nMetrics: {json.dumps(scalar, default=str)}"
        data, _, _ = _ask_json(llm, system, user)
        if data and data.get("interface_side") in ("cathode", "anode", "bulk_electrolyte"):
            result = {
                "interface_side": data["interface_side"],
                "mechanism": str(data.get("mechanism", mechanism))[:400],
                "next_constraints": [str(c)[:200] for c in (data.get("next_constraints") or [])][:5]
                or [_default_constraint(data["interface_side"])],
                "source": "llm",
            }
    return result
