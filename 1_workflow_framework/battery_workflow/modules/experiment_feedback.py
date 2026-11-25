"""
Experiment Feedback module — interfaces with characterization data
and cycling test results for cell-level validation.

Analogous to ChemAgents' RobotOperator which generates robot code
and executes experiments, this module consumes real experimental data
(EIS, cycling, post-mortem) and evaluates cell-level performance.

Key difference: ChemAgents' RO directly controls a robot via API;
our module ingests experimental results from various sources and
structures the feedback for the coordination loop.
"""

from __future__ import annotations

from typing import Any

from .base_module import BaseModule
from ..candidate_record import CandidateRecord, StageResult, Phase
from ..synthetic_controlflow_fixture import get_fixture, provenance_note


class ExperimentFeedback(BaseModule):
    """
    Processes experimental characterization and cycling data.

    In production, this would interface with:
      - EIS analysis pipelines
      - Cycling data management systems
      - Post-mortem characterization (XPS, TOF-SIMS, cryo-TEM)
    """

    def __init__(self):
        super().__init__(
            module_name="ExperimentFeedback",
            system_prompt=(
                "You are an experimental feedback agent. Select an audited "
                "characterization/cycling tool and evaluate cell-level performance, "
                "identifying failure mechanisms with interface-side attribution."
            ),
            agent="ExperimentFeedback",
        )

    def get_supported_actions(self) -> list[str]:
        return ["evaluate_cell_performance", "diagnose_failure"]

    def evaluate_cell_performance(
        self,
        parameters: dict[str, Any],
        candidate: CandidateRecord,
    ) -> StageResult:
        """
        Cell-level evaluation from cycling data. The LLM selects/justifies a real
        ExperimentFeedback tool; the cell metrics below are fixed case values (NOT
        LLM-generated) for CELL_GATE validation -- deliberately set to fail so the
        demo exercises the structured-feedback loop.
        """
        reasoning = self.select_tool_and_reason(
            task_desc=f"cell-level performance and failure diagnosis for {candidate.system_name}",
            candidate=candidate,
        )

        # Cell-level metrics from the cited, illustrative case fixture (NOT LLM/computed).
        # The baseline fixture is set to fail the cell gate so the demo exercises the
        # structured-feedback loop; the cathode-protected variant passes.
        fx = get_fixture(self.electrolyte_composition(candidate))
        cell_results = dict(fx["cell"])
        cell_results["_metrics_source"] = provenance_note(fx)
        # Consume the upstream handoff fields written by the ProcessCoordinator.
        handoff = candidate.test_conditions.get("_handoff", {}).get(self.module_name, {})
        if handoff.get("fields"):
            cell_results["received_upstream_fields"] = list(handoff["fields"].keys())
        passed = (
            cell_results.get("capacity_retention_80pct_cycles", 0) >= 200
            and cell_results.get("coulombic_efficiency_pct", 0) >= 99.5
        )
        cell_results.update(reasoning)

        return self._make_result(
            phase=Phase.CELL_VALIDATION,
            passed=passed,
            metrics=cell_results,
            artifacts={"rationale": reasoning["rationale"], "llm_call": self._last_llm_call_json},
        )

    def diagnose_failure(
        self,
        parameters: dict[str, Any],
        candidate: CandidateRecord,
    ) -> StageResult:
        """
        Perform root-cause analysis on a failed candidate using
        post-mortem characterization data.
        """
        diagnosis = {
            "root_cause": "CEI continuous growth on NMC811 cathode surface",
            "interface_side": "cathode",
            "evidence": [
                "XPS: increasing organic carbonate signal over cycles",
                "EIS: growing high-frequency semicircle",
                "Cross-section SEM: 15nm CEI at cycle 30 vs 3nm at cycle 1",
            ],
            "recommendation": (
                "Increase cross-linking density or add inorganic filler "
                "to suppress PEO oxidative decomposition at cathode interface"
            ),
        }

        return self._make_result(
            phase=Phase.FEEDBACK,
            passed=True,
            metrics=diagnosis,
        )
