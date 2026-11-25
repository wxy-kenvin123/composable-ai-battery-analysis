"""
Property Screener module — performs material-level fast screening.

This module wraps fast-prediction tools (e.g., HPG, ML-LNMO-additives)
to evaluate material-level properties before committing to expensive
simulation or experiments.
"""

from __future__ import annotations

from typing import Any

from .base_module import BaseModule
from ..candidate_record import CandidateRecord, StageResult, Phase
from ..synthetic_controlflow_fixture import get_fixture, provenance_note


class PropertyScreener(BaseModule):
    """
    Fast material-level property prediction and screening.

    In production, this would wrap tools like:
      - HPG (CRL=3): polymer electrolyte property prediction
      - ML-LNMO-additives (CRL=3): additive screening
      - Custom QSPR models for ionic conductivity
    """

    def __init__(self):
        super().__init__(
            module_name="PropertyScreener",
            system_prompt=(
                "You are a material property screening agent (HT Predictor). "
                "Select an audited fast-prediction tool and evaluate the candidate "
                "against material-level acceptance criteria."
            ),
            agent="HTPredictor",
        )

    def get_supported_actions(self) -> list[str]:
        return ["predict_material_properties"]

    def predict_material_properties(
        self,
        parameters: dict[str, Any],
        candidate: CandidateRecord,
    ) -> StageResult:
        """
        Material-level screening for the candidate electrolyte. The LLM selects/justifies
        a real HTPredictor tool; the gate metrics below are fixed case values for the
        PEO/Li/NMC811 reference system (NOT LLM-generated), evaluated by MATERIAL_GATE.
        """
        reasoning = self.select_tool_and_reason(
            task_desc=f"fast material-level property screening for {candidate.system_name}",
            candidate=candidate,
        )

        # Material-level metrics come from the cited, illustrative case fixture for
        # this candidate's electrolyte (NOT from the LLM, NOT computed here).
        fx = get_fixture(self.electrolyte_composition(candidate))
        predictions = dict(fx["material"])
        predictions["_metrics_source"] = provenance_note(fx)
        passed = all([
            predictions.get("ionic_conductivity_S_cm", 0) >= 1e-4,
            predictions.get("electrochemical_window_V", 0) >= 4.5,
            predictions.get("thermal_stability_C", 0) >= 150,
        ])

        predictions.update(reasoning)
        return self._make_result(
            phase=Phase.SCREENING,
            passed=passed,
            metrics=predictions,
            artifacts={"rationale": reasoning["rationale"], "llm_call": self._last_llm_call_json},
        )
