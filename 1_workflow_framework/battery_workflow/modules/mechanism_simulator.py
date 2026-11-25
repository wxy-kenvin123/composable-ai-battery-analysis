"""
Mechanism Simulator module — provides atomic/interface-level mechanistic
understanding through simulation.

Wraps tools like ai2-kit (CRL=3) and Allegro (CRL=3) for MD simulation,
DFT-informed force fields, and interface reaction analysis.
"""

from __future__ import annotations

from typing import Any

from .base_module import BaseModule
from ..candidate_record import CandidateRecord, StageResult, Phase
from ..synthetic_controlflow_fixture import get_fixture, provenance_note


class MechanismSimulator(BaseModule):
    """
    Atomic and interface-level simulation for mechanistic understanding.

    In production, this would interface with:
      - ai2-kit (CRL=3): ML force field training and MD
      - Allegro (CRL=3): equivariant neural network potentials
      - Interface reaction pathway analysis tools
    """

    def __init__(self):
        super().__init__(
            module_name="MechanismSimulator",
            system_prompt=(
                "You are a mechanism simulation agent. Select an audited MD/DFT/MLFF "
                "tool and assess transport properties and interfacial stability for "
                "battery candidates."
            ),
            agent="MechanismSimulator",
        )

    def get_supported_actions(self) -> list[str]:
        return ["run_interface_simulation", "analyze_transport"]

    def run_interface_simulation(
        self,
        parameters: dict[str, Any],
        candidate: CandidateRecord,
    ) -> StageResult:
        """
        PEO-NMC811 interface assessment at operating voltage. The LLM selects/justifies
        a real MechanismSimulator tool (e.g. ai2-kit, allegro); the interface metrics
        below are fixed case values (NOT LLM-generated) for INTERFACE_GATE evaluation.
        """
        reasoning = self.select_tool_and_reason(
            task_desc=f"MD/interface simulation of the {candidate.system_name} cathode interface at operating voltage",
            candidate=candidate,
        )

        # Interface metrics from the cited, illustrative case fixture (NOT LLM/computed).
        fx = get_fixture(self.electrolyte_composition(candidate))
        interface_results = dict(fx["interface"])
        interface_results["_metrics_source"] = provenance_note(fx)
        # Consume the upstream handoff fields written by the ProcessCoordinator.
        handoff = candidate.test_conditions.get("_handoff", {}).get(self.module_name, {})
        if handoff.get("fields"):
            interface_results["received_upstream_fields"] = list(handoff["fields"].keys())
        interface_results.update(reasoning)

        return self._make_result(
            phase=Phase.INTERFACE_VALIDATION,
            passed=True,
            metrics=interface_results,
            artifacts={
                "rationale": reasoning["rationale"],
                "llm_call": self._last_llm_call_json,
            },
        )

    def analyze_transport(
        self,
        parameters: dict[str, Any],
        candidate: CandidateRecord,
    ) -> StageResult:
        """Analyze Li+ transport pathway and mechanism."""
        return self._make_result(
            phase=Phase.SIMULATION,
            passed=True,
            metrics={
                "migration_barrier_eV": 0.25,
                "transport_mechanism": "segmental_motion",
            },
        )
