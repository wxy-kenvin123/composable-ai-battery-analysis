"""
Shared Candidate Record (CandidateRecord) — the data backbone of the
battery workflow coordination framework.

Analogous to ChemAgents' State (chem_proc/proc.py) which only carries
a message list, this record maintains structured multi-component state
for battery candidates across their full R&D lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Phase(str, Enum):
    SCREENING = "screening"
    SIMULATION = "simulation"
    INTERFACE_VALIDATION = "interface_validation"
    CELL_VALIDATION = "cell_validation"
    FEEDBACK = "feedback"
    COMPLETED = "completed"
    REJECTED = "rejected"


class InterfaceSide(str, Enum):
    CATHODE = "cathode"
    ANODE = "anode"
    BULK_ELECTROLYTE = "bulk_electrolyte"


@dataclass
class ComponentSpec:
    """Specification for a single battery component."""
    role: str  # e.g. "cathode", "electrolyte", "anode"
    composition: str  # e.g. "NMC811", "PEO/LiTFSI (EO:Li=16:1)"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureRecord:
    """Structured negative evidence from a failed validation stage."""
    phase: Phase
    interface_side: InterfaceSide | None
    failure_cycle: int | None = None
    failure_mechanism: str = ""
    conditions: dict[str, Any] = field(default_factory=dict)
    # Which gate rejected the candidate. Several gates can share a phase, so the
    # phase alone does not identify the screen that fired.
    gate_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class StageResult:
    """Result from a single processing stage."""
    module_name: str
    phase: Phase
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CandidateRecord:
    """
    Shared material archive for a battery candidate system.

    Tracks multi-component composition, stage progression, intermediate
    results, and structured failure logs across the entire R&D pipeline.
    """
    candidate_id: str
    system_name: str  # e.g. "PEO/Li/NMC811"
    components: list[ComponentSpec] = field(default_factory=list)
    current_phase: Phase = Phase.SCREENING
    test_conditions: dict[str, Any] = field(default_factory=dict)

    stage_results: list[StageResult] = field(default_factory=list)
    failure_log: list[FailureRecord] = field(default_factory=list)
    priority_score: float = 1.0
    # Normalized fixture keys of formulations already evaluated (dedup for the loop).
    tried_formulations: list[str] = field(default_factory=list)

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def advance_phase(self, next_phase: Phase) -> None:
        self.current_phase = next_phase
        self.updated_at = datetime.now().isoformat()

    def record_result(self, result: StageResult) -> None:
        self.stage_results.append(result)
        self.updated_at = datetime.now().isoformat()

    def record_failure(self, failure: FailureRecord) -> None:
        self.failure_log.append(failure)
        self.current_phase = Phase.REJECTED
        self.priority_score *= 0.3  # penalize similar structures
        self.updated_at = datetime.now().isoformat()

    def get_latest_metrics(self, module_name: str) -> dict[str, Any]:
        for result in reversed(self.stage_results):
            if result.module_name == module_name:
                return result.metrics
        return {}


@dataclass
class MaterialsKnowledgeBase:
    """
    Materials Knowledge Base (MKB) -- the shared store referenced in Fig. 12,
    organized into three layers:

      - Core:    candidates still under active evaluation.
      - Archive: candidates that reached a terminal state (completed / rejected).
      - Recall:  query interface over Core + Archive (past failures, similar
                 candidates) so the coordinator can constrain future searches.

    This is the structural backbone that keeps candidate identity, component
    information, test conditions, and validation results traceable across the
    discovery -> mechanism -> experiment stages.
    """

    core: dict[str, CandidateRecord] = field(default_factory=dict)
    archive: dict[str, CandidateRecord] = field(default_factory=dict)

    # ----- Core -----
    def add(self, candidate: CandidateRecord) -> None:
        self.core[candidate.candidate_id] = candidate

    def get(self, candidate_id: str) -> CandidateRecord | None:
        return self.core.get(candidate_id) or self.archive.get(candidate_id)

    def active(self) -> list[CandidateRecord]:
        return [c for c in self.core.values()
                if c.current_phase not in (Phase.REJECTED, Phase.COMPLETED)]

    def sync(self) -> None:
        """Move terminal candidates from Core to Archive."""
        for cid in list(self.core.keys()):
            c = self.core[cid]
            if c.current_phase in (Phase.REJECTED, Phase.COMPLETED):
                self.archive[cid] = self.core.pop(cid)

    # ----- Recall -----
    def recall_failures(self, interface_side: InterfaceSide | None = None) -> list[FailureRecord]:
        """All structured failure records, optionally filtered by interface side."""
        failures: list[FailureRecord] = []
        for c in list(self.core.values()) + list(self.archive.values()):
            for fr in c.failure_log:
                if interface_side is None or fr.interface_side == interface_side:
                    failures.append(fr)
        return failures

    def recall_similar(self, candidate: CandidateRecord) -> list[CandidateRecord]:
        """Candidates (other than this one) sharing a component composition."""
        own = {(comp.role, comp.composition) for comp in candidate.components}
        similar = []
        for c in list(self.core.values()) + list(self.archive.values()):
            if c.candidate_id == candidate.candidate_id:
                continue
            if any((comp.role, comp.composition) in own for comp in c.components):
                similar.append(c)
        return similar

    def snapshot(self) -> dict[str, Any]:
        return {
            "core": len(self.core),
            "archive": len(self.archive),
            "active": len(self.active()),
            "total_failures": len(self.recall_failures()),
        }
