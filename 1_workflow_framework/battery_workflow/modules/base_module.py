"""
Base module class for the battery workflow agents.

Each module operates on a structured ``CandidateRecord`` (not a flat message
list) and returns a typed ``StageResult``. The base class provides a shared,
data-grounded reasoning helper:

  ``select_tool_and_reason()`` queries the real 68-tool registry for the tools
  mapped to this agent (by their main input -> main output signature) and selects
  the most TASK-RELEVANT one via the coordination layer: a deterministic
  task-aware score (keyword overlap, then CRL/RRL) provides the reproducible
  baseline, while a live LLM may override the choice within the candidate set
  (schema-validated). The selecting LLM call is captured for traceability and
  written by each module into ``StageResult.artifacts["llm_call"]``.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from ..candidate_record import CandidateRecord, StageResult, Phase
from ..coordination import select_tool as coord_select_tool

try:
    from ..tool_registry import ToolRegistry
except Exception:  # pragma: no cover
    ToolRegistry = None  # type: ignore

try:
    from ..llm_client import get_llm_client
except Exception:  # pragma: no cover
    get_llm_client = None  # type: ignore


_REGISTRY = None


def _registry():
    """Lazy shared ToolRegistry; None if the census CSVs are unavailable."""
    global _REGISTRY
    if _REGISTRY is None:
        if ToolRegistry is None:
            _REGISTRY = False
        else:
            try:
                _REGISTRY = ToolRegistry.load()
            except Exception:
                _REGISTRY = False
    return _REGISTRY or None


class BaseModule(ABC):
    """Abstract base class for battery workflow modules (agents)."""

    def __init__(self, module_name: str, system_prompt: str = "", agent: str | None = None):
        self.module_name = module_name
        self.system_prompt = system_prompt
        # ``agent`` is the canonical registry/Fig.12 role name (e.g. PropertyScreener -> "HTPredictor").
        self.agent = agent or module_name
        self.llm = get_llm_client() if get_llm_client else None
        # JSON of the most recent selecting LLM call (for StageResult.artifacts["llm_call"]).
        self._last_llm_call_json = ""

    def execute(
        self,
        action: str,
        parameters: dict[str, Any],
        candidate: CandidateRecord,
    ) -> StageResult:
        method = getattr(self, action, None)
        if method is None:
            raise NotImplementedError(
                f"{self.module_name} does not support action '{action}'"
            )
        return method(parameters, candidate)

    @abstractmethod
    def get_supported_actions(self) -> list[str]:
        ...

    # ----- task-relevant tool selection via the coordination layer -----
    def select_tool_and_reason(self, task_desc: str, candidate: CandidateRecord) -> dict:
        """
        Select the most task-relevant audited tool for this agent. Deterministic
        baseline = task-keyword overlap then CRL/RRL; a live LLM may override
        within the candidate set (schema-validated). Returns metric-friendly keys;
        the selecting LLM call is stashed for the module to write into artifacts.
        """
        reg = _registry()
        tools = reg.tools_for_agent(self.agent) if reg else []
        selected, decision = coord_select_tool(self.agent, task_desc, candidate.system_name, tools, self.llm)

        call = decision.get("llm_call") or {}
        self._last_llm_call_json = json.dumps(call, ensure_ascii=False) if call else ""

        return {
            "agent": self.agent,
            "selected_tool": selected.tool if selected else None,
            "selected_repo": selected.github_url if selected else None,
            "selected_crl": selected.crl if selected else None,
            "n_candidates": len(tools),
            "candidates": [t.tool for t in tools[:8]],
            "decision_source": decision.get("source"),
            "rationale": decision.get("reason", ""),
        }

    @staticmethod
    def electrolyte_composition(candidate: CandidateRecord) -> str:
        """Composition string of the candidate's electrolyte component (for fixtures)."""
        for c in candidate.components:
            if c.role == "electrolyte":
                return c.composition
        return ""

    def _make_result(
        self,
        phase: Phase,
        passed: bool,
        metrics: dict[str, Any],
        artifacts: dict[str, str] | None = None,
    ) -> StageResult:
        return StageResult(
            module_name=self.module_name,
            phase=phase,
            passed=passed,
            metrics=metrics,
            artifacts=artifacts or {},
        )
