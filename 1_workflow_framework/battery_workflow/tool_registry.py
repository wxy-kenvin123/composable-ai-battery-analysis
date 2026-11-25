"""
Tool Registry — data-grounded backbone that connects the workflow framework
to the *real* 68-tool RRL/CRL audit.

Two CSVs are joined on ``paper_id``:
  - ``data/census/registry_tools.csv``  : the audited 68 downloadable tools
    (repository link, R1-R7, D1-D3, aggregate RRL/CRL, original cluster).
  - ``data/census/tool_agent_mapping.csv`` : each tool's assignment to one of
    the six R&D agents, classified by its **main input -> main output** signature.

This replaces the previous hard-coded placeholder tool names: every agent now
"carries" the real tools of the audit, so the demo and the SI
table draw from the same source of truth.

Stdlib-only; no external dependencies.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# --- Canonical agents and their input -> output signatures (Fig. 12 / SI S6.2) ---
# The signature is the classification rule: a tool is assigned to the agent whose
# (main_input -> main_output) it matches.
AGENT_IO: dict[str, dict[str, str]] = {
    "KnowledgeMiner": {
        "input": "literature / databases (text, abstracts, knowledge graphs)",
        "output": "structured design clues (formulations, property targets, failure modes)",
    },
    "FormulationDesigner": {
        "input": "target + constraints (+ design clues)",
        "output": "candidate multi-component formulations / compositions (generation, optimization, recommendation)",
    },
    "HTPredictor": {
        "input": "composition / structure descriptors",
        "output": "predicted material-level scalar property (fast surrogate, screening)",
    },
    "MechanismSimulator": {
        "input": "atomic structure / MD configuration",
        "output": "forces, energies, diffusion, migration barriers, interface stability (MD/DFT/MLFF)",
    },
    "ExperimentFeedback": {
        "input": "experimental / characterization / cycling data (images, EIS, capacity)",
        "output": "electrode/cell-level metrics + failure diagnosis",
    },
    "ProcessCoordinator": {
        "input": "outputs (fields) of the other agents",
        "output": "orchestration / field translation (no scientific output) -- the audited gap (0 tools)",
    },
}

# Canonical display order (matches Fig. 12 left-to-right pipeline).
AGENT_ORDER: list[str] = [
    "KnowledgeMiner",
    "FormulationDesigner",
    "HTPredictor",
    "MechanismSimulator",
    "ExperimentFeedback",
    "ProcessCoordinator",
]


@dataclass(frozen=True)
class ToolEntry:
    """A single audited tool, joined across the scores and agent-mapping tables."""

    paper_id: str
    tool: str
    agent: str
    cluster: str            # original eight-cluster label
    title: str
    github_url: str
    rrl: int                # 0-5
    crl: int                # 0-3
    d1: bool                # parameter/task configurability
    d2: bool                # data interoperability
    d3: bool                # interface invocability
    main_input: str = ""
    main_output: str = ""
    rationale: str = ""

    @property
    def chainable(self) -> bool:
        """CRL == 3: all three integration cues detected (priority packaging candidate)."""
        return self.crl >= 3


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return default


def _default_census_dir() -> Path:
    """Locate ``data/census`` relative to the repository, with env override."""
    override = os.environ.get("BATTAGENT_CENSUS_DIR")
    if override:
        return Path(override)
    # battery_workflow/tool_registry.py -> repo root is two levels up.
    return Path(__file__).resolve().parent.parent / "data" / "census"


class ToolRegistry:
    """
    Loads and joins the audited-tool tables, then exposes per-agent queries.

    Usage::

        reg = ToolRegistry.load()
        for t in reg.tools_for_agent("MechanismSimulator"):
            print(t.tool, t.crl, t.github_url)
    """

    SCORES_FILE = "registry_tools.csv"
    MAPPING_FILE = "tool_agent_mapping.csv"

    def __init__(self, entries: list[ToolEntry]):
        self._entries = entries
        self._by_id = {e.paper_id: e for e in entries}

    # ----- construction -----
    @classmethod
    def load(cls, census_dir: str | Path | None = None) -> "ToolRegistry":
        base = Path(census_dir) if census_dir else _default_census_dir()
        scores_path = base / cls.SCORES_FILE
        mapping_path = base / cls.MAPPING_FILE

        if not scores_path.exists():
            raise FileNotFoundError(
                f"RRL/CRL scores not found at {scores_path}. "
                f"Set BATTAGENT_CENSUS_DIR to the directory containing "
                f"{cls.SCORES_FILE} and {cls.MAPPING_FILE}."
            )
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"Agent mapping not found at {mapping_path}. "
                f"Generate it first (see data/census/tool_agent_mapping.csv)."
            )

        # agent mapping: paper_id -> {agent, main_input, main_output, rationale}
        mapping: dict[str, dict[str, str]] = {}
        with open(mapping_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                mapping[row["paper_id"].strip()] = row

        entries: list[ToolEntry] = []
        with open(scores_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                pid = row["paper_id"].strip()
                m = mapping.get(pid, {})
                entries.append(
                    ToolEntry(
                        paper_id=pid,
                        tool=(m.get("tool") or _tool_name_from_url(row.get("github_url", "")) or pid),
                        agent=m.get("agent", "Unassigned"),
                        cluster=row.get("cluster", ""),
                        title=row.get("title", ""),
                        github_url=row.get("github_url", ""),
                        rrl=_to_int(row.get("RRL")),
                        crl=_to_int(row.get("CRL")),
                        d1=_to_bool(row.get("D1")),
                        d2=_to_bool(row.get("D2")),
                        d3=_to_bool(row.get("D3")),
                        main_input=m.get("main_input", ""),
                        main_output=m.get("main_output", ""),
                        rationale=m.get("rationale", ""),
                    )
                )
        return cls(entries)

    # ----- queries -----
    def all_tools(self) -> list[ToolEntry]:
        return list(self._entries)

    def get(self, paper_id: str) -> ToolEntry | None:
        return self._by_id.get(str(paper_id))

    def agents(self) -> list[str]:
        present = {e.agent for e in self._entries}
        ordered = [a for a in AGENT_ORDER if a in present or a == "ProcessCoordinator"]
        extra = sorted(present - set(AGENT_ORDER))
        return ordered + extra

    def tools_for_agent(self, agent: str) -> list[ToolEntry]:
        """All audited tools assigned to ``agent`` (CRL desc, then RRL desc)."""
        tools = [e for e in self._entries if e.agent == agent]
        return sorted(tools, key=lambda e: (e.crl, e.rrl), reverse=True)

    def best_tools(self, agent: str, k: int = 3) -> list[ToolEntry]:
        """Top-k packaging-ready tools for an agent (highest CRL/RRL first)."""
        return self.tools_for_agent(agent)[:k]

    def agent_counts(self) -> dict[str, int]:
        counts = {a: 0 for a in AGENT_ORDER}
        for e in self._entries:
            counts[e.agent] = counts.get(e.agent, 0) + 1
        return counts

    def crl3_counts(self) -> dict[str, int]:
        counts = {a: 0 for a in AGENT_ORDER}
        for e in self._entries:
            if e.chainable:
                counts[e.agent] = counts.get(e.agent, 0) + 1
        return counts

    def summary_rows(self) -> list[tuple[str, int, int, str]]:
        """(agent, n_tools, n_crl3, io_signature) in canonical order -- for SI S6.2."""
        counts = self.agent_counts()
        crl3 = self.crl3_counts()
        rows = []
        for agent in AGENT_ORDER:
            io = AGENT_IO[agent]
            rows.append(
                (
                    agent,
                    counts.get(agent, 0),
                    crl3.get(agent, 0),
                    f"{io['input']} -> {io['output']}",
                )
            )
        return rows


def _tool_name_from_url(url: str) -> str:
    if not url:
        return ""
    return url.rstrip("/").split("/")[-1].replace(".git", "")


def _format_summary(reg: ToolRegistry) -> str:
    lines = ["Agent                  n  CRL=3  signature"]
    for agent, n, c3, io in reg.summary_rows():
        lines.append(f"{agent:<22}{n:>2}  {c3:>4}   {io[:60]}")
    lines.append(f"{'TOTAL':<22}{sum(reg.agent_counts().values()):>2}")
    return "\n".join(lines)


if __name__ == "__main__":
    reg = ToolRegistry.load()
    print(_format_summary(reg))
    print()
    for agent in AGENT_ORDER:
        tools = reg.tools_for_agent(agent)
        if not tools:
            print(f"[{agent}] (no tools -- the audited gap)" if agent == "ProcessCoordinator" else f"[{agent}] (none)")
            continue
        names = ", ".join(f"{t.tool}(CRL{t.crl})" for t in tools)
        print(f"[{agent}] {len(tools)}: {names}")
