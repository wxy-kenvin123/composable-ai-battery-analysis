"""
Formulation Designer module — generates or recommends multi-component
battery material formulations.

Analogous to ChemAgents' ExperimentDesigner which generates standardized
chemical experiment procedures from templates, this module produces
candidate formulations with full component specifications for downstream
screening and simulation.
"""

from __future__ import annotations

from typing import Any

from .base_module import BaseModule
from ..candidate_record import CandidateRecord, StageResult, Phase, ComponentSpec
from ..synthetic_controlflow_fixture import REMEDIES, VARIANTS, normalize


class FormulationDesigner(BaseModule):
    """
    Generates candidate formulations for multi-component battery systems.

    In production, this would integrate with:
      - Combinatorial design tools
      - Bayesian optimization for composition space exploration
      - Constraint satisfaction (charge balance, stability windows)
    """

    def __init__(self):
        super().__init__(
            module_name="FormulationDesigner",
            system_prompt=(
                "You are a battery formulation design agent. Generate "
                "candidate compositions for the target system, respecting "
                "electrochemical constraints and known failure modes."
            ),
            agent="FormulationDesigner",
        )

    def get_supported_actions(self) -> list[str]:
        return ["recommend_formulations", "refine_after_failure"]

    def _apply_formulation(self, candidate: CandidateRecord, electrolyte: str) -> str:
        """Set the candidate's components to a given electrolyte; record it as tried."""
        candidate.components = [
            ComponentSpec(role="cathode", composition="NMC811 (loading 3.5 mAh/cm^2)"),
            ComponentSpec(role="electrolyte", composition=electrolyte),
            ComponentSpec(role="anode", composition="Li metal"),
        ]
        key = normalize(electrolyte)
        if key not in candidate.tried_formulations:
            candidate.tried_formulations.append(key)
        return key

    def recommend_formulations(
        self,
        parameters: dict[str, Any],
        candidate: CandidateRecord,
    ) -> StageResult:
        """
        On the first round, set the baseline formulation (preferring a KnowledgeMiner
        clue that maps to the baseline). On later rounds, carry over whatever
        ``refine_after_failure`` chose, so the loop progresses rather than resetting.
        """
        knowledge = candidate.test_conditions.get("_knowledge", {})
        clue_formulations = [f.get("composition") for f in knowledge.get("formulations", []) if f.get("composition")]
        constraints = candidate.test_conditions.get("_constraints", [])

        reasoning = self.select_tool_and_reason(
            task_desc=parameters.get("query") or f"recommend formulations for {candidate.system_name}",
            candidate=candidate,
        )

        if not candidate.tried_formulations:
            baseline = next((c for c in clue_formulations if normalize(c) == "peo_litfsi_baseline"), VARIANTS[0])
            chosen_key = self._apply_formulation(candidate, baseline)
            status = "initial"
        else:
            chosen_key = normalize(self.electrolyte_composition(candidate))
            status = "carried_over"

        metrics = {
            "chosen_formulation_key": chosen_key,
            "status": status,
            "from_knowledge_clues": bool(clue_formulations),
            "active_constraints": len(constraints),
            "tried_count": len(candidate.tried_formulations),
        }
        metrics.update(reasoning)
        return self._make_result(
            phase=Phase.SCREENING,
            passed=True,
            metrics=metrics,
            artifacts={
                "electrolyte": self.electrolyte_composition(candidate),
                "rationale": reasoning["rationale"],
                "llm_call": self._last_llm_call_json,
            },
        )

    @staticmethod
    def _diagnosed_remedy(candidate: CandidateRecord) -> str | None:
        """
        Read the structured negative evidence and name the remedy it calls for.

        The most recent failure is consulted before the accumulated constraint list,
        so a later cathode-side failure is not masked by an earlier bulk-electrolyte
        one still sitting in the constraints.
        """
        texts: list[str] = []
        if candidate.failure_log:
            last = candidate.failure_log[-1]
            texts.append((last.failure_mechanism or "").lower())
            if last.interface_side is not None:
                texts.append(getattr(last.interface_side, "value", str(last.interface_side)).lower())
        texts.append(" ".join(candidate.test_conditions.get("_constraints", [])).lower())

        for text in texts:
            if "cei" in text or "cathode" in text:
                return "cathode_cei"
            if "window" in text or "oxidat" in text:
                return "oxidative_stability"
        return None

    def refine_after_failure(
        self,
        parameters: dict[str, Any],
        candidate: CandidateRecord,
    ) -> StageResult:
        """
        Pick the next UNTRIED formulation that remedies the diagnosed failure, and
        apply it. Falls back to the next untried variant in the escalation chain when
        no variant claims that remedy. Refuses to resubmit an already-tried formulation.
        """
        constraints = candidate.test_conditions.get("_constraints", [])
        remedy = self._diagnosed_remedy(candidate)

        untried = [v for v in VARIANTS if normalize(v) not in candidate.tried_formulations]
        next_comp = next(
            (v for v in untried if REMEDIES.get(normalize(v)) == remedy),
            untried[0] if untried else None,
        )

        if next_comp is None:
            return self._make_result(
                phase=Phase.FEEDBACK,
                passed=False,
                metrics={"refined": False, "reason": "formulation space exhausted",
                         "tried": list(candidate.tried_formulations)},
            )

        key = self._apply_formulation(candidate, next_comp)
        return self._make_result(
            phase=Phase.SCREENING,
            passed=True,
            metrics={"refined": True, "refined_to_key": key, "refined_to": next_comp,
                     "diagnosed_remedy": remedy or "none",
                     "remedy_matched": REMEDIES.get(key) == remedy,
                     "constraints_applied": len(constraints)},
            artifacts={"refined_to": next_comp},
        )
