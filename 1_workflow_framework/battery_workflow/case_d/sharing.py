"""Record sharing and module ablation in sequential electrolyte searches."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from ..datasets.oedb import Observation
from . import acquisition as acq
from .batch_policy import plan_batch
from .campaign import OEDBCampaign
from .feasibility import FeasibilityConfig, FeasibilityModel, labels_from_statuses
from .mechanism import d_from_minimize_form
from .metrics import pareto_mask

PACKAGE = Path(__file__).resolve().parent


def load_config(name: str) -> dict[str, Any]:
    """Read one self-contained experiment definition and verify its fixed inputs."""
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("expected a configuration name")
    path = PACKAGE / "configs" / (name + ".json")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg["name"] != name:
        raise ValueError("configuration name differs from its filename")
    for item in (cfg["rule_pack"], cfg["hard_failure_protocol"]):
        actual = hashlib.sha256((PACKAGE / item["file"]).read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {item['file']}")
    # Verify the eight files without exposing their labels to the optimizer.
    from ..datasets.oedb import _source_from_contract

    _source_from_contract(cfg).verify()
    return cfg


def hard_fail_ids(config, protocol):
    spec = next(
        (x for x in config["hard_failure_protocol"]["protocols"] if x["set"] == protocol), None
    )
    if spec is None:
        raise ValueError(f"unknown failure protocol {protocol}")
    path = PACKAGE / config["hard_failure_protocol"]["file"]
    ids = json.loads(path.read_text(encoding="utf-8"))["sets"][protocol]["candidate_ids"]
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()
    if digest != spec["candidate_ids_sha256"] or len(ids) != spec["n_unprocessable"]:
        raise ValueError(f"failure membership differs for {protocol}")
    return frozenset(int(i) for i in ids)


def _subset_mechanism(mechanism, unseen, kept):
    if not mechanism.enabled:
        return mechanism
    keep = np.array([int(cid) in set(int(x) for x in kept) for cid in unseen], dtype=bool)

    extra = {}
    if getattr(mechanism, "tplus_hat_unseen", None) is not None:
        extra["tplus_hat_unseen"] = np.asarray(mechanism.tplus_hat_unseen)[keep]
    return dataclasses.replace(
        mechanism,
        m1_prior_unseen=np.asarray(mechanism.m1_prior_unseen)[keep],
        m2_hat_unseen=np.asarray(mechanism.m2_hat_unseen)[keep],
        m3_hat_unseen=np.asarray(mechanism.m3_hat_unseen)[keep],
        **extra,
    )


def session_own_numbered_pareto(observations):
    """Select the session's measured nondominated conductivity and transference pairs."""
    numbered = [
        o
        for o in observations
        if o.status == "ok"
        and o.m1 is not None
        and o.d_cation is not None
        and o.d_anion is not None
    ]
    if not numbered:
        return []
    points = np.array(
        [
            [float(o.m1), -(float(o.d_cation) / (float(o.d_cation) + float(o.d_anion)))]
            for o in numbered
        ],
        dtype=float,
    )
    return [obs for obs, keep in zip(numbered, pareto_mask(points)) if bool(keep)]


class SharingSession(OEDBCampaign):
    """Search one group using the records available at the start of its session."""

    OBJECTIVE_MODE = "sigma_tplus"
    LATTICE_DIMENSION = 2

    def __init__(
        self,
        snapshot,
        arm_name,
        seed,
        session_index,
        hard_fail_ids,
        inherited_numbered,
        inherited_no_number,
        feasibility,
        pof_enabled,
        void_own_failures,
        weight_selection,
        weight_stride,
        weight_start,
    ):
        super().__init__(snapshot, arm_name, seed)
        self.session_index = int(session_index)
        self.hard_fail_ids = hard_fail_ids
        self.attempt_log = []
        for obs in inherited_numbered:
            self.state.record(obs)
        self.state.observed_ids.extend(int(cid) for cid, _ in inherited_no_number)
        self._inherited_statuses = [
            (int(o.candidate_id), str(o.status)) for o in inherited_numbered
        ] + [(int(cid), str(status)) for cid, status in inherited_no_number if status is not None]
        self.feasibility_cfg = feasibility
        self.pof_enabled = bool(pof_enabled)
        self.void_own_failures = bool(void_own_failures)
        self.weight_selection = str(weight_selection)
        self.weight_stride = int(weight_stride)
        self.weight_start = int(weight_start)
        self.pof_active_batches = 0
        self.batch_reports = []
        self._pof_row = {}
        self._pof_by_id = {}
        self._processability_event = None

    def reveal(self, candidate_id: int) -> Any:
        if int(candidate_id) in self.hard_fail_ids:
            observation = Observation(
                candidate_id=int(candidate_id),
                status="hard_fail",
                m1=None,
                m2=None,
                m3=None,
                d_cation=None,
                d_anion=None,
                concentration=self.space.concentration(int(candidate_id)),
                duplicate=False,
            )
        else:
            observation = super().reveal(candidate_id)
        self.attempt_log.append(observation)
        return observation

    def initialise(self) -> None:

        pool = list(self.state.unseen_ids())
        n_init = int(self.contract["sessions"]["per_session_budget"]["n_init"])
        withheld = 0
        design_pof_active = False
        if self.pof_enabled:
            model = self.fit_feasibility()
            if model.active:
                design_pof_active = True
                pof = model.pof(np.asarray(self.space.feature_matrix(pool), dtype=float))
                mask = pof >= self.feasibility_cfg.pof_veto_threshold
                if int(mask.sum()) >= n_init and not mask.all():
                    withheld = int((~mask).sum())
                    pool = [cid for cid, keep in zip(pool, mask) if keep]
        features = np.asarray(self.space.feature_matrix(pool), dtype=float)
        design_seed = self.seed * 10 + self.session_index
        indices, info = acq.greedy_maximin(features, n_init, design_seed)
        info = dict(info)
        info["design_seed"] = design_seed
        info["pool_size"] = len(pool) + withheld
        info["pool_size_after_feasibility"] = len(pool)
        info["design_withheld"] = withheld
        info["design_pof_active"] = design_pof_active
        self.init_info = info

        for position, index in enumerate(indices, start=1):
            candidate_id = int(pool[index])
            observation = self.reveal(candidate_id)
            self.state.record(observation)
            row = self._base_row(position, "initial", candidate_id, observation)
            row["pof_withheld"] = withheld
            row["pof_active"] = design_pof_active
            self.rounds.append(row)
        self._refresh_running_best()

    def training_statuses(self) -> tuple[list[int], list[str]]:
        pairs = list(self._inherited_statuses)
        for observation in self.attempt_log:
            if self.void_own_failures and str(observation.status) == "hard_fail":
                continue
            pairs.append((int(observation.candidate_id), str(observation.status)))
        return [cid for cid, _ in pairs], [status for _, status in pairs]

    def fit_feasibility(self) -> FeasibilityModel:
        model = FeasibilityModel(self.feasibility_cfg)
        ids, statuses = self.training_statuses()
        keep, y = labels_from_statuses(statuses)
        kept_ids = [cid for cid, k in zip(ids, keep) if k]
        if kept_ids:
            X = np.asarray(self.space.feature_matrix(kept_ids), dtype=float)
            model.fit(X, y)
        return model

    def _feasible_pool(self, mechanism, unseen, q: int):
        ei_scale = None
        self._pof_by_id = {}
        kept = tuple(int(cid) for cid in unseen)
        kept_mechanism = mechanism
        self._pof_row = {
            "pof_active": False,
            "pof_n_positives": "",
            "pof_n_negatives": "",
            "pof_iterations": "",
            "pof_converged": "",
            "pof_withheld": "",
        }
        if not self.pof_enabled:
            return kept, kept_mechanism, ei_scale
        model = self.fit_feasibility()
        self._pof_row = model.event_row()
        self._pof_row["pof_withheld"] = 0
        if not model.active:
            return kept, kept_mechanism, ei_scale
        self.pof_active_batches += 1
        x_unseen = np.asarray(self.space.feature_matrix(unseen), dtype=float)
        pof = model.pof(x_unseen)
        self._pof_by_id = {int(cid): float(p) for cid, p in zip(unseen, pof)}
        mask = pof >= self.feasibility_cfg.pof_veto_threshold
        if mask.any() and not mask.all() and int(mask.sum()) >= q:
            kept = tuple(cid for cid, keep in zip(unseen, mask) if keep)
            kept_mechanism = _subset_mechanism(mechanism, unseen, kept)
            self._pof_row["pof_withheld"] = int((~mask).sum())
            pof = pof[mask]
        ei_scale = np.clip(pof, self.feasibility_cfg.pof_floor, 1.0)
        return kept, kept_mechanism, ei_scale

    def optimised_batch(self, batch_index: int, q: int) -> list[Any]:
        """Plan ``q`` selections from the information in hand, then reveal them."""
        unseen = self.candidate_pool()
        knowledge = self.stage_knowledge(unseen)
        mechanism = self.stage_mechanism(knowledge, unseen)
        kept, kept_mechanism, ei_scale = self._feasibility_for_batch(mechanism, unseen, q)
        first_round = (int(batch_index) - 1) * q + 1
        self.state.round_index = first_round
        proposals, diagnostics = plan_batch(
            self.state,
            kept_mechanism,
            self.memory,
            kept,
            self.lattice,
            q=q,
            stride=self.weight_stride,
            start=self.weight_start,
            ei_scale=ei_scale,
            weight_selection=self.weight_selection,
        )
        cells = []
        outcomes = []
        for within, proposal in enumerate(proposals, start=1):
            round_index = (int(batch_index) - 1) * q + within
            self.state.round_index = round_index
            # Planned ids are unseen at batch start; later reveals in this
            # batch must not make an earlier pick look like a repeat.
            gate_row = self.check_unseen(round_index, proposal, unseen)
            outcome = self.reveal_and_record(
                round_index, knowledge, mechanism, proposal, [gate_row]
            )
            row = outcome.row
            row.update(self._pof_row)
            selected = self._pof_by_id.get(int(proposal.candidate_id))
            row["pof_selected"] = "" if selected is None else round(selected, 9)
            row["batch_index"] = int(batch_index)
            row["within_batch_index"] = int(within)
            row["weight_selection"] = self.weight_selection
            anion, solvent, _ = self.space.key(int(proposal.candidate_id))
            cells.append((anion, solvent))
            outcomes.append(outcome)
        diversity = len(set(cells))
        for outcome in outcomes:
            outcome.row["batch_diversity"] = diversity
        self.batch_reports.append(
            {
                "batch_index": int(batch_index),
                "q": q,
                "pof_active": bool(self._pof_row.get("pof_active")),
                "batch_diversity": diversity,
                "candidate_ids": [int(p.candidate_id) for p in proposals],
                "diagnostics": diagnostics,
            }
        )
        return outcomes

    def _feasibility_for_batch(self, mechanism, unseen, q):
        kept, kept_mechanism, ei_scale = self._feasible_pool(mechanism, unseen, q)
        channel = self.contract["knowledge_channel"]
        if channel == "prior":
            return kept, kept_mechanism, ei_scale
        anions = [self.space.anion(i) for i in kept]
        concentrations = [self.space.concentration(i) for i in kept]
        if channel == "processability_se_ei_scale":
            m2 = np.asarray(kept_mechanism.m2_hat_unseen, dtype=float)
            if len(m2) != len(kept):
                keep = np.array(
                    [int(cid) in set(int(x) for x in kept) for cid in unseen], dtype=bool
                )
                m2 = m2[keep] if len(m2) == len(unseen) else np.zeros(len(kept), dtype=float)
            knowledge = self.km.evaluate(
                anions, concentrations, d_cation_hat=d_from_minimize_form(m2)
            )
            self._processability_event = knowledge
        elif channel == "processability_ei_scale":
            knowledge = self.km.evaluate(anions, concentrations)
        else:
            raise ValueError(f"unknown knowledge channel {channel}")
        pi = np.asarray(knowledge.processability, dtype=float)
        if ei_scale is None and np.allclose(pi, 1.0):
            return kept, kept_mechanism, None
        if ei_scale is None:
            return kept, kept_mechanism, pi
        return kept, kept_mechanism, np.asarray(ei_scale, dtype=float) * pi

    def reveal_and_record(self, round_index, knowledge, mechanism, proposal, gates=None):
        if self._processability_event is not None:
            knowledge = self._processability_event
        outcome = super().reveal_and_record(round_index, knowledge, mechanism, proposal, gates)
        outcome.row.update(self._pof_row)
        selected = self._pof_by_id.get(int(proposal.candidate_id))
        outcome.row["pof_selected"] = "" if selected is None else round(selected, 9)
        return outcome


class SharingCampaign:
    """Run all sessions in order and account for records transferred across them."""

    def __init__(self, snapshot, arm_name, seed, protocol="p80", pof_enabled=True):
        self._snapshot_ref = snapshot
        self.contract = snapshot.contract
        self.arm_cfg = next((a for a in self.contract["arms"] if a["name"] == arm_name), None)
        if self.arm_cfg is None:
            raise ValueError(f"unknown sharing condition {arm_name}")
        self.arm_name = arm_name
        self.seed = int(seed)
        self.protocol = protocol
        self.hard_fail_ids = hard_fail_ids(self.contract, protocol)
        self.transfer = self.arm_cfg["transfer"]
        self.coordinator_on = self.arm_cfg["coordinator"]
        self.feasibility_cfg = FeasibilityConfig.from_contract(self.contract)
        self.pof_enabled = bool(pof_enabled)
        self.void_own_failures = self.arm_cfg["void_own_failures"]
        self.records_failures_for_successors = self.arm_cfg["records_failures_for_successors"]
        self.success_transfer = self.arm_cfg["success_transfer"]
        self.failure_regime = self.arm_cfg["failure_regime"]
        block = self.contract["sessions"]
        budget = block["per_session_budget"]
        self.n_sessions = int(block["n_sessions"])
        self.n_init = int(budget["n_init"])
        self.n_optimised_batches = int(budget["n_optimised_batches"])
        self.batch_size = int(budget["batch_size"])
        self.per_session_total = int(budget["total"])
        self.campaign_total = int(block["campaign_total_budget"])
        if self.n_init + self.n_optimised_batches * self.batch_size != self.per_session_total:
            raise ValueError("session budget does not add up")
        if self.per_session_total * self.n_sessions != self.campaign_total:
            raise ValueError("campaign budget does not add up")
        batch = self.contract["acquisition"]["batch"]
        self.weight_selection = batch["weight_selection"]
        self.weight_stride = int(batch["weight_stride"])
        self.weight_start = int(batch["weight_start"])
        self._attempted_ever: set[int] = set()
        self._numbered_store: dict[int, Observation] = {}
        self._no_number_status: dict[int, str | None] = {}
        self._session_attempts: list[tuple[int, Observation]] = []

        self.counters = {
            "cross_session_duplicate_attempts": 0,
            "repeated_hard_fail_attempts": 0,
            "hard_fail_attempts_total": 0,
            "hard_fail_first_attempts": 0,
            "data_missing_attempts": 0,
            "numbered_attempts": 0,
            "within_session_duplicates": 0,
            "hard_fail_first_attempts_later": 0,
            "hard_fail_first_attempts_later_design": 0,
            "hard_fail_first_attempts_later_optimised": 0,
            "pof_active_batches": 0,
        }
        self.per_session: list[dict[str, Any]] = []
        self.rounds: list[dict[str, Any]] = []
        self.knowledge_events: list[dict[str, Any]] = []
        self.gate_decisions: list[dict[str, Any]] = []
        self.agent_trace: list[dict[str, Any]] = []
        self.init_infos: list[dict[str, Any]] = []
        self.batch_reports: list[dict[str, Any]] = []
        self.protocol_violations_total = 0
        self._finished = False

        self.weight_schedule = acq.weight_schedule_report_stride(
            acq.simplex_lattice(
                int(self.contract["acquisition"]["weight_source"]["lattice_divisions"]),
                dimension=SharingSession.LATTICE_DIMENSION,
            ),
            n_batches=self.n_sessions * self.n_optimised_batches,
            q=self.batch_size,
            stride=self.weight_stride if self.weight_selection == "stride" else 1,
            start=self.weight_start,
        )

    def rule_labels(self):
        values = {
            "knowledge_pack": self.contract["rule_pack"]["file"],
            "knowledge_channel": self.contract["knowledge_channel"],
            "knowledge_combination": self.contract["knowledge_combination"],
        }
        return {k: values[k] for k in self.contract["report_rule_fields"]}

    def _inheritance(self) -> tuple[list[Observation], list[tuple[int, str | None]]]:
        if self.transfer == "none":
            return [], []
        if self.transfer == "numbers_only":
            extras: list[tuple[int, str | None]] = []
            if self.records_failures_for_successors:
                extras = [
                    (cid, status)
                    for cid, status in sorted(self._no_number_status.items())
                    if status is not None
                ]
            return list(self._numbered_store.values()), extras
        if self.transfer == "full_learn":
            if self.records_failures_for_successors:
                no_number = [
                    (cid, status) for cid, status in sorted(self._no_number_status.items())
                ]
            else:
                no_number = [(cid, None) for cid in sorted(self._no_number_status)]
            return list(self._numbered_store.values()), no_number
        raise ValueError(f"unknown transfer policy {self.transfer!r}")

    def _account_session(self, session_index: int, log: list[Observation]) -> None:
        session_seen: set[int] = set()
        for observation in log:
            cid = int(observation.candidate_id)
            if cid in session_seen:
                self.counters["within_session_duplicates"] += 1
            elif cid in self._attempted_ever:
                self.counters["cross_session_duplicate_attempts"] += 1
                if cid in self.hard_fail_ids:
                    self.counters["repeated_hard_fail_attempts"] += 1
            session_seen.add(cid)
            self._session_attempts.append((session_index, observation))
            if observation.status == "hard_fail":
                self.counters["hard_fail_attempts_total"] += 1
                if cid not in self._attempted_ever:
                    self.counters["hard_fail_first_attempts"] += 1
            elif observation.status == "data_missing":
                self.counters["data_missing_attempts"] += 1
            elif observation.status == "ok":
                self.counters["numbered_attempts"] += 1

    def _fold_session(self, session_attempts: list[Observation]) -> int:
        if self.success_transfer == "pareto":
            numbered_to_store = session_own_numbered_pareto(session_attempts)
        elif self.success_transfer == "all":
            numbered_to_store = [obs for obs in session_attempts if obs.status == "ok"]
        else:
            raise ValueError(f"unknown success_transfer {self.success_transfer!r}")

        for obs in session_attempts:
            cid = int(obs.candidate_id)
            self._attempted_ever.add(cid)
            if obs.status == "ok":
                continue
            if self.records_failures_for_successors:
                self._no_number_status.setdefault(cid, str(obs.status))
            elif self.transfer == "full_learn":
                self._no_number_status.setdefault(cid, None)
        for obs in numbered_to_store:
            self._numbered_store.setdefault(int(obs.candidate_id), obs)
        return len(numbered_to_store)

    def run(self) -> None:
        for session_index in range(1, self.n_sessions + 1):
            inherited_numbered, inherited_no_number = self._inheritance()
            session = SharingSession(
                self._snapshot_ref,
                self.arm_name,
                self.seed,
                session_index,
                self.hard_fail_ids,
                inherited_numbered,
                inherited_no_number,
                self.feasibility_cfg,
                pof_enabled=self.pof_enabled,
                void_own_failures=self.void_own_failures,
                weight_selection=self.weight_selection,
                weight_stride=self.weight_stride,
                weight_start=self.weight_start,
            )
            session.initialise()
            for batch_index in range(1, self.n_optimised_batches + 1):
                session.optimised_batch(batch_index, self.batch_size)
            if self.coordinator_on:
                for report in session.batch_reports:
                    self.agent_trace.append(
                        {
                            "kind": "batch",
                            "action": "plan_batch",
                            "session": session_index,
                            "batch_index": report["batch_index"],
                            "candidate_ids": list(report["candidate_ids"]),
                            "source": self.contract["contract_tag"],
                        }
                    )

            self._account_session(session_index, session.attempt_log)
            self.counters["pof_active_batches"] += int(session.pof_active_batches)

            first_hf_session = 0
            first_hf_design = 0
            first_hf_optimised = 0
            for row in session.rounds:
                cid = int(row["candidate_id"])
                if row["status"] == "hard_fail" and cid not in self._attempted_ever:
                    first_hf_session += 1
                    if row["phase"] == "initial":
                        first_hf_design += 1
                    else:
                        first_hf_optimised += 1
            if session_index >= 2:
                self.counters["hard_fail_first_attempts_later"] += first_hf_session
                self.counters["hard_fail_first_attempts_later_design"] += first_hf_design
                self.counters["hard_fail_first_attempts_later_optimised"] += first_hf_optimised

            offset = (session_index - 1) * self.per_session_total
            earlier = self._attempted_before_session(session_index)
            for row in session.rounds:
                row = dict(row)
                row["protocol"] = self.protocol
                row["session"] = session_index
                row["session_budget_used"] = row["budget_used"]
                row["budget_used"] = offset + int(row["budget_used"])
                row["cross_session_duplicate"] = int(row["candidate_id"]) in earlier
                row["void_own_failures"] = self.void_own_failures
                row["records_failures_for_successors"] = self.records_failures_for_successors
                row["success_transfer"] = self.success_transfer
                row["failure_regime"] = self.failure_regime
                row["world"] = self.contract["name"]
                row["session_contract"] = self.contract["contract_tag"]
                row["objectives"] = SharingSession.OBJECTIVE_MODE
                row.update(self.rule_labels())
                self.rounds.append(row)
            for event in session.knowledge_events:
                event = dict(event)
                event["protocol"] = self.protocol
                event["session"] = session_index
                self.knowledge_events.append(event)
            for gate in session.gate_decisions:
                gate = dict(gate)
                gate["protocol"] = self.protocol
                gate["session"] = session_index
                self.gate_decisions.append(gate)
            self.init_infos.append(session.init_info)
            for report in session.batch_reports:
                report = dict(report)
                report["session"] = session_index
                self.batch_reports.append(report)
            self.protocol_violations_total += sum(session.protocol_violations.values())

            session_attempts = [obs for s, obs in self._session_attempts if s == session_index]
            folded_numbered = self._fold_session(session_attempts)
            self.per_session.append(
                {
                    "session": session_index,
                    "attempts": len(session_attempts),
                    "numbered": sum(1 for o in session_attempts if o.status == "ok"),
                    "hard_fail": sum(1 for o in session_attempts if o.status == "hard_fail"),
                    "data_missing": sum(1 for o in session_attempts if o.status == "data_missing"),
                    "cross_session_duplicates": sum(
                        1 for o in session_attempts if int(o.candidate_id) in earlier
                    ),
                    "hard_fail_first": first_hf_session,
                    "hard_fail_first_design": first_hf_design,
                    "hard_fail_first_optimised": first_hf_optimised,
                    "pof_active_batches": int(session.pof_active_batches),
                    "inherited_numbered": len(inherited_numbered),
                    "folded_numbered": folded_numbered,
                    "mean_batch_diversity": (
                        float(np.mean([r["batch_diversity"] for r in session.batch_reports]))
                        if session.batch_reports
                        else ""
                    ),
                }
            )

        later = int(self.counters["hard_fail_first_attempts_later"])
        design = int(self.counters["hard_fail_first_attempts_later_design"])
        optimised = int(self.counters["hard_fail_first_attempts_later_optimised"])
        if later != design + optimised:
            raise RuntimeError(f"later swamp identity failed: {later} != {design} + {optimised}")
        if len(self.rounds) != self.campaign_total:
            raise RuntimeError(
                f"budget ledger failed: {len(self.rounds)} evaluations, expected {self.campaign_total}"
            )
        self._finished = True

    def _attempted_before_session(self, session_index: int) -> set[int]:
        return {int(o.candidate_id) for s, o in self._session_attempts if s < session_index}

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def arm(self):
        return SimpleNamespace(name=self.arm_name)

    def unique_cells_attempted(self) -> int:
        return len({(row["anion"], row["solvent"]) for row in self.rounds})

    def summary(self) -> dict[str, Any]:
        counters = dict(self.counters)
        return {
            "arm": self.arm_name,
            "seed": self.seed,
            "protocol": self.protocol,
            "transfer": self.transfer,
            "success_transfer": self.success_transfer,
            "pof_enabled": self.pof_enabled,
            "world": self.contract["name"],
            "void_own_failures": self.void_own_failures,
            "records_failures_for_successors": self.records_failures_for_successors,
            "failure_regime": self.failure_regime,
            "n_sessions": self.n_sessions,
            "per_session_budget": self.per_session_total,
            "batch_size": self.batch_size,
            "weight_stride": self.weight_stride,
            "weight_start": self.weight_start,
            "weight_selection": self.weight_selection,
            "budget_used": len(self.rounds),
            "wasted_fraction": round(
                counters["cross_session_duplicate_attempts"] / self.campaign_total, 9
            ),
            **counters,
            "per_session": list(self.per_session),
            "protocol_violations": self.protocol_violations_total,
            "unique_candidates_attempted": len(self._attempted_ever),
            "unique_cells_attempted": self.unique_cells_attempted(),
            "mean_batch_diversity": (
                float(np.mean([r["batch_diversity"] for r in self.batch_reports]))
                if self.batch_reports
                else ""
            ),
            "weight_schedule_report": dict(self.weight_schedule),
            "source": self.contract["contract_tag"],
            "objectives": SharingSession.OBJECTIVE_MODE,
            **self.rule_labels(),
        }
