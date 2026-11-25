"""Oracle access and bookkeeping for one electrolyte search session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import numpy as np

from . import acquisition as acq
from .gates import (
    oedb_data_available_gate,
    oedb_running_best_gate,
    oedb_unseen_gate,
)
from .knowledge import KnowledgeManager, load_pack
from .mechanism import MechanismSimulator, MemoryFeedback
from .policy import (
    ArmConfig,
    CampaignState,
    MechanismStageOutput,
    Proposal,
    knowledge_stage,
    mechanism_stage,
    observed_scalarised,
    plan_stage,
    random_proposal,
)


#: Provenance pointers. Written on every exported row, so they are references
#: rather than paragraphs; the paragraphs live in run_record_case_d.json, which is
#: written once.
_LABEL_REF = "OEDB MCP exportData, unfiltered; doi:10.1038/s41524-026-02093-y"


@dataclass
class RoundOutcome:
    """Everything one round produced, ready to be flattened into a CSV row."""

    row: dict[str, Any]
    knowledge_event: dict[str, Any]
    gate_decisions: list[dict[str, Any]]
    proposal: Proposal | None
    observation: Any


class OEDBCampaign:
    """A single (arm, seed) run over the pinned OEDB Li subspace."""


    OBJECTIVE_MODE = "m1m2m3"
    LATTICE_DIMENSION = 3

    def __init__(self, snapshot: Any, arm_name: str, seed: int) -> None:
        # The snapshot itself is deliberately NOT kept on the campaign. It can
        # mint fresh oracles and evaluators, so retaining it would hand every
        # agent (which holds the campaign) a path to the full label set, so the
        # campaign keeps only the three capability-separated views it actually
        # needs.
        self.contract = snapshot.contract
        self.arm = ArmConfig.from_contract(self.contract, arm_name)
        self.seed = int(seed)

        self.space = snapshot.search_space()
        self._oracle = snapshot.oracle()
        self._evaluator = snapshot.evaluator()

        self.state = CampaignState(
            contract=self.contract,
            space=self.space,
            arm=self.arm,
            seed=self.seed,
            objective_mode=self.OBJECTIVE_MODE,
        )
        self.km = KnowledgeManager(pack=load_pack(Path(__file__).parent / self.contract["rule_pack"]["file"]), enabled=self.arm.km)
        self.mechanism_sim = MechanismSimulator(self.contract, enabled=self.arm.mechanism)
        memory_cfg = self.contract["memory"]
        self.memory = MemoryFeedback(
            k=int(memory_cfg["k"]),
            residual_scale=float(memory_cfg["residual_scale"]),
            cap=float(memory_cfg["cap"]),
            enabled=self.arm.mkb,
        )
        self.lattice = acq.simplex_lattice(
            int(self.contract["acquisition"]["weight_source"]["lattice_divisions"]),
            dimension=self.LATTICE_DIMENSION,
        )
        self.rho = float(self.contract["acquisition"]["rho"])

        self.rounds: list[dict[str, Any]] = []
        self.knowledge_events: list[dict[str, Any]] = []
        self.gate_decisions: list[dict[str, Any]] = []
        self.init_info: dict[str, Any] = {}
        self.protocol_violations = {
            "unseen_gate_failures": 0,
            "evaluator_enabled_early": 0,
            "agent_touched_label": 0,
            "oracle_called_after_gate_failure": 0,
        }
        self._running_best: float | None = None
        self._finished = False

    # ----- capability boundary -------------------------------------------------
    def reveal(self, candidate_id: int) -> Any:
        """The single point in the campaign where a label enters."""
        return self._oracle.reveal(candidate_id)

    @property
    def oracle_stats(self) -> dict[str, Any]:
        return {
            "reveal_count": self._oracle.reveal_count,
            "duplicate_queries": self._oracle.duplicate_queries,
            "missing_queries": self._oracle.missing_queries,
        }

    # ----- stages, called by the monolith and by the agents alike --------------
    def stage_knowledge(self, unseen):
        return knowledge_stage(self.km, self.state, unseen)

    def stage_mechanism(self, knowledge, unseen) -> MechanismStageOutput:
        return mechanism_stage(self.mechanism_sim, self.km, self.state, knowledge, unseen)

    def stage_plan(self, mechanism, unseen) -> Proposal:
        if self.arm.random_only:
            return random_proposal(self.state, unseen)
        return plan_stage(self.state, mechanism, self.memory, unseen, self.lattice)

    # ----- initial design ------------------------------------------------------
    def initialise(self) -> None:
        features = np.asarray(self.space.feature_matrix(), dtype=float)
        n_init = int(self.contract["sessions"]["per_session_budget"]["n_init"])
        indices, info = acq.greedy_maximin(features, n_init, self.seed)
        self.init_info = info
        for position, index in enumerate(indices, start=1):
            candidate_id = self.space.ids[index]
            observation = self.reveal(candidate_id)
            self.state.record(observation)
            self.rounds.append(self._base_row(position, "initial", candidate_id, observation))
        self._refresh_running_best()

    def _refresh_running_best(self) -> None:
        """Best scalarised value so far, under the *next* round's weights.

        Recomputed rather than carried forward because the ParEGO weights change
        every round, so a value from an earlier weighting is not comparable.
        """
        if not self.state.m:
            self._running_best = None
            return
        next_round = max(self.state.round_index + 1, 1)
        weights = acq.weight_for_round(self.lattice, next_round)
        m = np.asarray(self.state.m, dtype=float)
        lower, upper = acq.observed_bounds(m)
        values = acq.augmented_chebyshev(
            acq.min_max_normalise(m, lower, upper), weights, self.rho
        )
        self._running_best = float(values.min())

    def _base_row(
        self, position: int, phase: str, candidate_id: int, observation: Any
    ) -> dict[str, Any]:
        anion, solvent, concentration = self.space.key(candidate_id)
        return {
            "arm": self.arm.name,
            "seed": self.seed,
            "budget_used": position,
            "phase": phase,
            "candidate_id": candidate_id,
            "anion": anion,
            "solvent": solvent,
            "concentration_mol_per_kg": concentration,
            "status": observation.status,
            "m1_neg_log10_sigma": _fmt(observation.m1),
            "m2_neg_log10_d_li": _fmt(observation.m2),
            "m3_log10_eta": _fmt(observation.m3),
            "duplicate": observation.duplicate,
            "source": _LABEL_REF,
        }

    # ----- one optimised round -------------------------------------------------
    def candidate_pool(self) -> tuple[int, ...]:
        """The ids the planner may choose from this round.

        The base campaign's pool is everything its record says is unrevealed.
        Case D-R arms override this: an arm without deduplication ranks every
        candidate, and the 0/68 arm's record simply does not contain the failures
        it never heard about.
        """
        return self.state.unseen_ids()

    def feedback_gates(self) -> list:
        """Gates applied after the oracle agent, in order, for this round."""
        gates = [oedb_data_available_gate()]
        running_best = self.running_best_gate()
        if running_best is not None:
            gates.append(running_best)
        return gates

    def optimised_round(self, round_index: int) -> RoundOutcome:
        """The monolith's round: the same stages, the same gates, no orchestration."""
        self.state.round_index = round_index
        unseen = self.candidate_pool()

        knowledge = self.stage_knowledge(unseen)
        mechanism = self.stage_mechanism(knowledge, unseen)
        proposal = self.stage_plan(mechanism, unseen)

        gate_row = self.check_unseen(round_index, proposal, unseen)
        return self.reveal_and_record(
            round_index, knowledge, mechanism, proposal, [gate_row]
        )

    def check_unseen(self, round_index: int, proposal: Proposal, unseen) -> dict[str, Any]:
        """Protocol gate, applied before anything is revealed.

        Placed here rather than inside the reveal so that the multi-agent arm can
        fire it after its planner and before its oracle agent, which is the same
        order the monolith uses.
        """
        is_unseen = int(proposal.candidate_id in set(unseen))
        gate = oedb_unseen_gate()
        passed, reasons = gate.evaluate({"candidate_is_unseen": is_unseen})
        row = self._gate_row(round_index, gate, passed, reasons, proposal.candidate_id)
        if not passed:
            self.protocol_violations["unseen_gate_failures"] += 1
            raise RuntimeError(
                f"Round {round_index} proposed candidate {proposal.candidate_id}, which is "
                "not an unrevealed member of the universe. Refusing to continue: a "
                "campaign that re-reveals points has no well-defined budget."
            )
        return row

    def running_best_gate(self):
        """The progress gate for the round about to run, or None before any data."""
        if self._running_best is None:
            return None
        return oedb_running_best_gate(self._running_best, self.state.n_usable)

    def reveal_and_record(
        self,
        round_index: int,
        knowledge,
        mechanism,
        proposal: Proposal,
        gates: list[dict[str, Any]] | None = None,
    ) -> RoundOutcome:
        """Reveal the gated proposal and record everything about the round.

        Shared by both arms so that the revealed point and the recorded rows
        cannot drift apart between them.
        """
        budget_used = int(self.contract["sessions"]["per_session_budget"]["n_init"]) + round_index
        gates = list(gates or [])

        observation = self.reveal(proposal.candidate_id)

        data_gate = oedb_data_available_gate()
        ok = int(observation.status == "ok")
        passed_data, reasons = data_gate.evaluate({"label_status_ok": ok})
        gates.append(
            self._gate_row(round_index, data_gate, passed_data, reasons, proposal.candidate_id)
        )

        row = self._base_row(budget_used, "optimised", proposal.candidate_id, observation)
        row.update(proposal.event_row())
        row.update(knowledge.event_row())
        row.update(mechanism.event_row())
        row["round_index"] = round_index

        scalarised = None
        if passed_data:
            weights = np.array(proposal.weights, dtype=float)
            scalarised = observed_scalarised(self.state, weights, self.rho, observation)
            best_gate = self.running_best_gate()
            if best_gate is not None:
                passed_best, reasons = best_gate.evaluate({"scalarised_value": scalarised})
                gates.append(
                    self._gate_row(
                        round_index, best_gate, passed_best, reasons, proposal.candidate_id
                    )
                )
                row["running_best_before"] = round(self._running_best, 9)
                row["improved_running_best"] = passed_best
            if proposal.predicted_scalarised == proposal.predicted_scalarised:
                self.state.scalarised_residuals.append(
                    proposal.predicted_scalarised - scalarised
                )
        row["observed_scalarised"] = _fmt(scalarised)

        self.state.record(observation)
        self._refresh_running_best()

        event = {
            "arm": self.arm.name,
            "seed": self.seed,
            "round_index": round_index,
            "candidate_id": proposal.candidate_id,
        }
        event.update(knowledge.event_row())
        event.update(mechanism.event_row())
        event["mkb_residuals_recorded"] = len(self.state.scalarised_residuals)
        event["noise_inflation"] = proposal.event_row()["noise_inflation"]
        event["source"] = self.contract["contract_tag"]

        self.rounds.append(row)
        self.knowledge_events.append(event)
        self.gate_decisions.extend(gates)
        return RoundOutcome(
            row=row,
            knowledge_event=event,
            gate_decisions=gates,
            proposal=proposal,
            observation=observation,
        )

    def _gate_row(
        self, round_index: int, gate, passed: bool, reasons, candidate_id: int
    ) -> dict[str, Any]:
        return {
            "arm": self.arm.name,
            "seed": self.seed,
            "round_index": round_index,
            "candidate_id": candidate_id,
            "gate": gate.name,
            "verdict": "PASS" if passed else "FAIL",
            "semantics": (
                "optimization_non_improvement, not chemical failure"
                if gate.name == "oedb-running-best"
                else ("data gap, not chemical failure" if gate.name == "oedb-data-available" else "protocol check")
            ),
            "reasons": "; ".join(reasons),
            "threshold_source": gate.source,
            # A short pointer rather than a repeat of threshold_source: the two were
            # identical, and duplicating a per-round provenance string across 17,000
            # rows cost more megabytes than it carried information.
            "source": self.contract["contract_tag"],
        }

    # ----- closing -------------------------------------------------------------
    def finish(self):
        """Freeze the oracle, then unlock the evaluator. Order is the whole point."""
        self._oracle.freeze()
        self._evaluator.enable(self._oracle)
        self._finished = True
        return self._evaluator

    @property
    def finished(self) -> bool:
        return self._finished

    def summary(self) -> dict[str, Any]:
        optimised = [r for r in self.rounds if r["phase"] == "optimised"]
        ties = sum(1 for r in optimised if r.get("acquisition_tie"))
        at_grid_max = sum(
            1 for r in optimised if r.get("gp_m2_lengthscale") == 8.0
        )
        return {
            "arm": self.arm.name,
            "seed": self.seed,
            "budget_used": len(self.rounds),
            "initial_points": int(self.contract["sessions"]["per_session_budget"]["n_init"]),
            "optimised_rounds": len(optimised),
            "usable_observations": self.state.n_usable,
            "data_missing_hits": sum(1 for r in self.rounds if r["status"] == "data_missing"),
            "data_missing_ids_hit": sorted(
                {r["candidate_id"] for r in self.rounds if r["status"] == "data_missing"}
            ),
            "duplicate_selections": self._oracle.duplicate_queries,
            "protocol_violations": sum(self.protocol_violations.values()),
            "protocol_violation_detail": dict(self.protocol_violations),
            "acquisition_ties": ties,
            "rounds_with_m2_lengthscale_at_grid_max": at_grid_max,
            "initial_design": self.init_info,
        }


def _fmt(value: float | None) -> float | str:
    if value is None:
        return ""
    if value != value:
        return ""
    return round(float(value), 9)
