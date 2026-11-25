"""
Battery Workflow Coordinator -- the orchestration hub (Fig. 12 "Orchestrator").

Runs an explicit Observe-Plan-Execute-Validate (OPEV) loop over a shared
Materials Knowledge Base (MKB):

  - Observe : read the candidate's state and the MKB snapshot.
  - Plan    : decompose the objective into staged subtasks.
  - Execute : dispatch each subtask to its domain agent; between adjacent
              stages the ProcessCoordinator translates output fields into the
              next stage's inputs (the lightweight field protocol).
  - Validate: apply the stage's gate; on failure, write structured negative
              evidence, trigger the feedback loop (ProcessCoordinator +
              FormulationDesigner.refine_after_failure), and iterate.

Key differences from ChemAgents:
  - Operates on multi-component candidate systems (not single materials).
  - Maintains a three-layer MKB (Core/Archive/Recall) with cross-component state.
  - Enforces three-stage validation gates between modules.
  - Tracks structured negative evidence for genuine feedback iterations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .candidate_record import (
    CandidateRecord,
    Phase,
    ComponentSpec,
    MaterialsKnowledgeBase,
)
from .validation_gates import (
    MATERIAL_GATE,
    INTERFACE_GATE,
    CELL_GATE,
    run_validation,
)
from . import coordination
from .llm_client import get_llm_client


@dataclass
class TaskSpec:
    """A subtask to be dispatched to a module."""
    module_name: str
    action: str
    parameters: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)


# Default linear pipeline (the Plan step). Gates are applied after the
# starred stages by ``_gate_after``.
_DEFAULT_PIPELINE = [
    ("KnowledgeMiner", "retrieve_design_clues"),
    ("FormulationDesigner", "recommend_formulations"),
    ("PropertyScreener", "predict_material_properties"),    # -> MATERIAL_GATE
    ("MechanismSimulator", "run_interface_simulation"),     # -> INTERFACE_GATE
    ("ExperimentFeedback", "evaluate_cell_performance"),    # -> CELL_GATE
]

# Field-handoff edges handled by the ProcessCoordinator between adjacent stages.
_HANDOFF_EDGES = {
    "PropertyScreener": "MechanismSimulator",
    "MechanismSimulator": "ExperimentFeedback",
    "ExperimentFeedback": "FormulationDesigner",
}

# Gate applied after each stage of the default pipeline.
_DEFAULT_GATES = {
    "PropertyScreener": MATERIAL_GATE,
    "MechanismSimulator": INTERFACE_GATE,
    "ExperimentFeedback": CELL_GATE,
}


class Coordinator:
    """Central orchestrator for battery multi-component R&D workflows.

    The default configuration is the generic five-agent pipeline with the three
    synthetic gates, which is what the control-flow smoke test exercises. The
    published-data cases pass their own ``pipeline``, ``gate_map`` and
    ``handoff_edges`` so that the same OPEV driver runs agents backed by an
    audited tool's released data instead of by the synthetic fixture.

    ``deterministic=True`` detaches the coordinator *and every module registered
    afterwards* from the LLM client. The published cases set it so that a
    configured API key cannot change a gate sequence, which would otherwise make
    their reported counts depend on the local environment.
    """

    def __init__(
        self,
        pipeline: list[tuple[str, str]] | None = None,
        gate_map: dict[str, list] | None = None,
        handoff_edges: dict[str, str] | None = None,
        deterministic: bool = False,
    ):
        self.modules: dict[str, Any] = {}
        self.mkb = MaterialsKnowledgeBase()
        self.task_queue: list[TaskSpec] = []
        self.deterministic = deterministic
        self.llm = None if deterministic else get_llm_client()
        self.pipeline = list(pipeline) if pipeline else None
        self.gate_map = {k: list(v) for k, v in gate_map.items()} if gate_map is not None else None
        self.handoff_edges = dict(handoff_edges) if handoff_edges is not None else dict(_HANDOFF_EDGES)
        self.plan_source = "deterministic"
        # Every agent invocation, handoff and gate decision, in execution order.
        self.agent_trace: list[dict] = []
        # Backward-compatible alias: some callers used ``self.candidates``.
        self.candidates = self.mkb.core

    def register_module(self, name: str, module: Any) -> None:
        if self.deterministic:
            module.llm = None
        self.modules[name] = module

    def create_candidate(
        self,
        candidate_id: str,
        system_name: str,
        components: list[ComponentSpec],
        test_conditions: dict[str, Any] | None = None,
    ) -> CandidateRecord:
        record = CandidateRecord(
            candidate_id=candidate_id,
            system_name=system_name,
            components=components,
            test_conditions=test_conditions or {},
        )
        self.mkb.add(record)
        return record

    # ----- Plan -----
    def plan_workflow(self, objective: str, candidate_id: str) -> list[TaskSpec]:
        # The LLM (when live) may reorder the pipeline; the deterministic default
        # is the reproducible baseline. Both are validated to contain the gated stages.
        pipeline, self.plan_source = coordination.plan_pipeline(
            objective, set(self.modules), self.llm, default_pipeline=self.pipeline
        )
        if not pipeline:  # registry/modules missing -> fall back to the literal default
            default = self.pipeline or _DEFAULT_PIPELINE
            pipeline = [(m, a) for m, a in default if m in self.modules]
        tasks: list[TaskSpec] = []
        prev = None
        for module_name, action in pipeline:
            params = {"candidate_id": candidate_id}
            if module_name == "KnowledgeMiner":
                params["query"] = objective
            tasks.append(
                TaskSpec(
                    module_name=module_name,
                    action=action,
                    parameters=params,
                    depends_on=[prev] if prev else [],
                )
            )
            prev = module_name
        self.task_queue = tasks
        return tasks

    # ----- OPEV driver -----
    def run(self, candidate_id: str, objective: str, max_iterations: int = 3) -> dict:
        """Execute the full Observe-Plan-Execute-Validate loop with feedback."""
        candidate = self.mkb.get(candidate_id)
        if candidate is None:
            raise KeyError(f"unknown candidate '{candidate_id}'")

        history: list[dict] = []
        for iteration in range(1, max_iterations + 1):
            observation = self._observe(candidate)
            tasks = self.plan_workflow(objective, candidate_id)
            outcome = self._execute_and_validate(candidate, tasks)
            history.append(
                {"iteration": iteration, "observe": observation, "outcome": outcome}
            )
            if candidate.current_phase == Phase.COMPLETED:
                break
            if candidate.current_phase == Phase.REJECTED:
                if iteration < max_iterations:
                    self._feedback(candidate)  # revives candidate for next round
                else:
                    break

        self.mkb.sync()
        return {
            "candidate": candidate,
            "iterations": history,
            "mkb": self.mkb.snapshot(),
        }

    # ----- Observe -----
    def _observe(self, candidate: CandidateRecord) -> dict:
        return {
            "phase": candidate.current_phase.value,
            "priority": round(candidate.priority_score, 3),
            "failures_so_far": len(candidate.failure_log),
            "mkb": self.mkb.snapshot(),
        }

    # ----- Execute + Validate -----
    def _execute_and_validate(self, candidate: CandidateRecord, tasks: list[TaskSpec]) -> dict:
        steps: list[dict] = []
        pc = self.modules.get("ProcessCoordinator")
        prev_module: str | None = None
        rejected = False

        for task in tasks:
            if candidate.current_phase == Phase.REJECTED:
                break

            module = self.modules.get(task.module_name)
            if module is None:
                raise RuntimeError(f"Module '{task.module_name}' not registered")

            # ProcessCoordinator field handoff from the previous stage (if mapped).
            if pc is not None and prev_module and self.handoff_edges.get(prev_module) == task.module_name:
                handoff = pc.execute(
                    "resolve_handoff",
                    {"source_module": prev_module, "target_module": task.module_name},
                    candidate,
                )
                candidate.record_result(handoff)
                step = {"stage": f"handoff:{prev_module}->{task.module_name}",
                        "kind": "handoff",
                        "agent": "ProcessCoordinator",
                        "action": "resolve_handoff",
                        "mapped_fields": handoff.metrics.get("mapped_fields", 0)}
                steps.append(step)
                self._trace(candidate, step)

            result = module.execute(task.action, task.parameters, candidate)
            candidate.record_result(result)
            step = {
                "stage": task.module_name,
                "kind": "agent",
                "agent": getattr(module, "agent", task.module_name),
                "action": task.action,
                "passed": result.passed,
                "selected_tool": result.metrics.get("selected_tool"),
                "decision_source": result.metrics.get("decision_source"),
                "backend": result.metrics.get("backend"),
                "metrics_source": result.metrics.get("_metrics_source"),
                "execution": result.metrics.get("execution"),
            }
            steps.append(step)
            self._trace(candidate, step)

            for gate in self._gates_after(task.module_name):
                passed = run_validation(candidate, gate, result.metrics, self.llm)
                step = {"stage": f"gate:{gate.name or gate.phase.value}",
                        "kind": "gate",
                        "gate": gate.name or gate.phase.value,
                        "after_agent": task.module_name,
                        "passed": passed,
                        "phase": candidate.current_phase.value}
                steps.append(step)
                self._trace(candidate, step)
                if not passed:
                    if pc is not None:
                        feedback = pc.execute("trigger_feedback_loop", {}, candidate)
                        candidate.record_result(feedback)
                        step = {
                            "stage": "feedback:trigger_feedback_loop",
                            "kind": "feedback",
                            "agent": getattr(pc, "agent", "ProcessCoordinator"),
                            "action": "trigger_feedback_loop",
                            "passed": feedback.passed,
                            "failures_on_record": len(candidate.failure_log),
                            "constraints_on_record": len(
                                candidate.test_conditions.get("_constraints", [])
                            ),
                        }
                        steps.append(step)
                        self._trace(candidate, step)
                    rejected = True
                    break
            if rejected:
                break

            prev_module = task.module_name

        return {"final_phase": candidate.current_phase.value, "steps": steps}

    def _trace(self, candidate: CandidateRecord, step: dict) -> None:
        self.agent_trace.append({"candidate_id": candidate.candidate_id, **step})

    # ----- Feedback (revive for the next iteration) -----
    def _feedback(self, candidate: CandidateRecord) -> None:
        pc = self.modules.get("ProcessCoordinator")
        if pc is not None:
            self._invoke_feedback(candidate, pc, "ProcessCoordinator", "update_priorities")
        fd = self.modules.get("FormulationDesigner")
        if fd is not None and "refine_after_failure" in fd.get_supported_actions():
            self._invoke_feedback(candidate, fd, "FormulationDesigner", "refine_after_failure")
        # Revive: a rejected candidate re-enters screening with its failure log intact,
        # so the next round is constrained by the structured negative evidence.
        candidate.advance_phase(Phase.SCREENING)

    def _invoke_feedback(self, candidate: CandidateRecord, module: Any, name: str, action: str) -> None:
        """Run a feedback-path action and put it in the trace.

        The feedback path runs outside ``_execute_and_validate``, so without this
        the step that turns a rejection into the next proposal would be the one
        part of the loop with no record of having happened.
        """
        result = module.execute(action, {}, candidate)
        candidate.record_result(result)
        self._trace(candidate, {
            "stage": f"feedback:{action}",
            "kind": "feedback",
            "agent": getattr(module, "agent", name),
            "action": action,
            "passed": result.passed,
            "failures_on_record": len(candidate.failure_log),
            "constraints_on_record": len(candidate.test_conditions.get("_constraints", [])),
        })

    def _gates_after(self, module_name: str) -> list:
        """Gates applied after a stage, in order; evaluation stops at the first failure.

        A list rather than a single gate because a published screen can be a
        cascade: the three ML-HTVS filters are all material-level and all follow
        the same agent.
        """
        if self.gate_map is not None:
            return list(self.gate_map.get(module_name, []))
        gate = _DEFAULT_GATES.get(module_name)
        return [gate] if gate is not None else []

    def _gate_after(self, module_name: str):
        """First gate after a stage, or None (kept for callers predating cascades)."""
        gates = self._gates_after(module_name)
        return gates[0] if gates else None

    # ----- queries -----
    def get_active_candidates(self) -> list[CandidateRecord]:
        return self.mkb.active()

    def get_ranked_candidates(self) -> list[CandidateRecord]:
        return sorted(self.mkb.active(), key=lambda c: c.priority_score, reverse=True)
