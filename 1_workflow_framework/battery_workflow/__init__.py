"""
Battery Workflow Coordination Framework
========================================

A conceptual proof-of-concept for coordinating battery AI tools into
composable R&D workflows.

Architecture (three layers):
  - Coordinator:        Orchestrates multi-stage candidate evaluation
  - Modules (×6):       Domain-specific R&D function units
  - CandidateRecord:    Shared material archive for cross-tool traceability

Design inspired by ChemAgents (Song et al., 2025) Agent-as-Tool pattern,
adapted for battery-specific requirements:
  - Multi-component candidate tracking (cathode + electrolyte + anode + interface)
  - Three-stage progressive validation gates (material → interface → cell)
  - Structured negative evidence and feedback loops
"""

__version__ = "0.2.0"


# Explicit correspondence between the code class names and the module names used in
# the figures. The registry/AGENT_IO use the canonical role keys in the right column.
#
#   code class          figure role             registry agent key
NAME_MAP: dict[str, dict[str, str]] = {
    "KnowledgeMiner":      {"figure": "Knowledge Miner",      "agent": "KnowledgeMiner"},
    "FormulationDesigner": {"figure": "Materials Designer",   "agent": "FormulationDesigner"},
    "PropertyScreener":    {"figure": "HT Predictor",         "agent": "HTPredictor"},
    "MechanismSimulator":  {"figure": "Mechanism Simulator",  "agent": "MechanismSimulator"},
    "ExperimentFeedback":  {"figure": "Experiment Planner",   "agent": "ExperimentFeedback"},
    "ProcessCoordinator":  {"figure": "Process Coordinator",  "agent": "ProcessCoordinator"},
    "Coordinator":         {"figure": "Orchestrator",         "agent": "-"},
}


def figure_role(class_name: str) -> str:
    """Return the Fig. 12 role name for a given code class name."""
    return NAME_MAP.get(class_name, {}).get("figure", class_name)
