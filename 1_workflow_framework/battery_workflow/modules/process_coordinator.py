"""
Process Coordinator module — manages inter-module scheduling,
field passing, and state synchronization.

This is the module that fills the "zero coverage" gap identified
in the 68-tool audit. It handles:
  - Task dependency resolution
  - Intermediate quantity passing between adjacent scales
  - Candidate state transitions and priority management
"""

from __future__ import annotations

from typing import Any

from .base_module import BaseModule
from ..candidate_record import CandidateRecord, StageResult, Phase
from ..coordination import translate_fields


class ProcessCoordinator(BaseModule):
    """
    Manages the scheduling and data flow between other modules.

    This module is responsible for:
      1. Resolving task dependencies
      2. Translating output fields from one module into input fields
         expected by the next (semantic bridging)
      3. Triggering validation gates at appropriate points
      4. Managing the feedback loop when candidates fail
    """

    def __init__(self, field_mappings: dict[str, dict[str, str]] | None = None):
        super().__init__(
            module_name="ProcessCoordinator",
            system_prompt=(
                "You are a process coordination agent. Manage the flow of "
                "battery candidates through the R&D pipeline, ensuring "
                "proper data handoff between modules and enforcing "
                "stage-gate progression rules."
            ),
            agent="ProcessCoordinator",
        )
        # A case supplies its own mappings when its agents carry the published
        # tool's field names rather than the synthetic fixture's.
        self.field_mappings = field_mappings or self._default_field_mappings()

    def get_supported_actions(self) -> list[str]:
        return ["resolve_handoff", "trigger_feedback_loop", "update_priorities"]

    def resolve_handoff(
        self,
        parameters: dict[str, Any],
        candidate: CandidateRecord,
    ) -> StageResult:
        """
        Resolve the data handoff between two adjacent modules.
        Maps output fields from source module to input fields expected
        by the target module.
        """
        source = parameters.get("source_module", "")
        target = parameters.get("target_module", "")
        mapping_key = f"{source}->{target}"
        mapping = self.field_mappings.get(mapping_key, {})

        source_metrics = candidate.get_latest_metrics(source)
        # Deterministic field protocol produces the values; a live LLM adds the
        # cross-scale semantic meaning. The transformed fields are WRITTEN ONTO the
        # candidate so the downstream module actually consumes them.
        fields, note, src = translate_fields(source, target, source_metrics, mapping, self.llm)
        handoff = candidate.test_conditions.setdefault("_handoff", {})
        handoff[target] = {"from": source, "fields": fields, "semantic_note": note, "source": src}

        return self._make_result(
            phase=candidate.current_phase,
            passed=len(fields) > 0,
            metrics={"mapped_fields": len(fields), "mapping": mapping_key, "translation_source": src},
            artifacts={"transformed_data": str(fields), "semantic_note": note},
        )

    def trigger_feedback_loop(
        self,
        parameters: dict[str, Any],
        candidate: CandidateRecord,
    ) -> StageResult:
        """
        Initiate a feedback loop: structure the failure, update priorities,
        and prepare constraints for the next design iteration.
        """
        constraints = []
        for failure in candidate.failure_log:
            constraints.append({
                "avoid_mechanism": failure.failure_mechanism,
                "interface_side": failure.interface_side.value if failure.interface_side else None,
                "penalize_similar": True,
            })

        return self._make_result(
            phase=Phase.FEEDBACK,
            passed=True,
            metrics={
                "feedback_constraints": len(constraints),
                "priority_adjusted": candidate.priority_score,
            },
        )

    def update_priorities(
        self,
        parameters: dict[str, Any],
        candidate: CandidateRecord,
    ) -> StageResult:
        """Update candidate priority based on accumulated evidence."""
        # Boost candidates that pass more gates
        passed_stages = sum(1 for r in candidate.stage_results if r.passed)
        candidate.priority_score = 1.0 + 0.2 * passed_stages

        return self._make_result(
            phase=candidate.current_phase,
            passed=True,
            metrics={"new_priority": candidate.priority_score},
        )

    @staticmethod
    def _default_field_mappings() -> dict[str, dict[str, str]]:
        """
        Define the semantic field mappings between adjacent modules.
        This is the 'lightweight field protocol' mentioned in the paper.
        """
        return {
            "PropertyScreener->MechanismSimulator": {
                "initial_conductivity": "ionic_conductivity_S_cm",
                "stability_window": "electrochemical_window_V",
                "glass_transition": "glass_transition_temp_C",
            },
            "MechanismSimulator->ExperimentFeedback": {
                "predicted_diffusion": "li_diffusion_coefficient_cm2_s",
                "interface_stability": "oxidation_onset_V",
                "predicted_impedance_trend": "eis_impedance_50cycles_ohm_cm2",
            },
            "ExperimentFeedback->FormulationDesigner": {
                "failure_mechanism": "attributed_to",
                "failure_cycle": "failure_cycle",
                "performance_gap": "capacity_retention_80pct_cycles",
            },
        }
