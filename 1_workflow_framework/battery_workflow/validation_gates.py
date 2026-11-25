"""
Progressive validation gates for battery candidates.

Analogous to ChemAgents' ED_EXPERTRULE_PROMPT (expert-rule reflection in a single
pass), this module implements multi-stage go/no-go criteria that progressively
tighten as a candidate gets more expensive to evaluate.

Every criterion carries a ``source`` string, because where a threshold comes from
determines what a gate decision is worth. Two provenance classes appear here:

  published   the threshold is a literal in an audited tool's own released code,
              quoted with notebook, cell, line and DOI. Gates built this way let
              the framework re-derive a published screening result instead of
              restating it -- see the ML_HTVS_* gates.
  synthetic   conventional engineering targets used only by the control-flow
              smoke test. They are not results. See the
              MATERIAL_GATE / INTERFACE_GATE / CELL_GATE block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .candidate_record import CandidateRecord, Phase, FailureRecord, InterfaceSide
from . import coordination


@dataclass
class GateCriterion:
    """A single pass/fail criterion within a validation gate."""
    metric_name: str
    operator: str  # ">", ">=", "<", "<=", "in_range", "stable_within"
    threshold: Any
    unit: str = ""
    description: str = ""
    #: Where the threshold comes from -- file, cell/line, and citation where the
    #: criterion was taken from a published tool. Exported with every decision.
    source: str = ""


@dataclass
class ValidationGate:
    """A collection of criteria that must all pass for stage advancement."""
    phase: Phase
    criteria: list[GateCriterion] = field(default_factory=list)
    #: Short identifier, needed when several gates share a phase (the three
    #: ML-HTVS screens are all material-level).
    name: str = ""
    #: Phase a candidate enters on success. ``None`` uses the default ladder.
    next_phase: Phase | None = None
    source: str = ""

    def evaluate(self, metrics: dict[str, Any]) -> tuple[bool, list[str]]:
        failures = []
        for criterion in self.criteria:
            value = metrics.get(criterion.metric_name)
            if value is None:
                failures.append(
                    f"{criterion.metric_name}: not available"
                )
                continue
            if not self._check(value, criterion):
                failures.append(
                    f"{criterion.metric_name}: {value} {criterion.unit} "
                    f"fails {criterion.operator} {criterion.threshold}"
                )
        return len(failures) == 0, failures

    def _check(self, value: Any, criterion: GateCriterion) -> bool:
        op = criterion.operator
        thr = criterion.threshold
        if op == ">=":
            return value >= thr
        elif op == ">":
            return value > thr
        elif op == "<=":
            return value <= thr
        elif op == "<":
            return value < thr
        elif op == "in_range":
            return thr[0] <= value <= thr[1]
        elif op == "stable_within":
            # value should be a sequence; check variation < threshold
            if not hasattr(value, "__len__") or len(value) < 2:
                return False
            variation = (max(value) - min(value)) / max(abs(min(value)), 1e-9)
            return variation <= thr
        return False


# --------------------------------------------------------------------------- #
# Published: the ML-HTVS-Li-SE screening funnel
#
# Kim, Mok, Kim & Back released the notebooks that produced their solid-electrolyte
# shortlist. Each of the three filters below is a literal from those notebooks, so
# a candidate's verdict here is the verdict the published pipeline gave it. All
# three are material-level, which is why they share Phase.SCREENING and are
# distinguished by ``name``; the shortlist that clears all three is what the
# authors then sent to structure generation and DFT.
# --------------------------------------------------------------------------- #

_HTVS_REPO = "SeoinBack/ML-HTVS-Li-SE @ c13da8e9"
_HTVS_CITE = "[@kim2023accelerating] doi:10.1021/acsami.3c10798"


def _htvs_source(notebook: str, cell: int, line: int) -> str:
    return f"{_HTVS_REPO} {notebook} cell[{cell}] line {line}; {_HTVS_CITE}"


ML_HTVS_THERMODYNAMIC_GATE = ValidationGate(
    name="htvs-thermodynamic",
    phase=Phase.SCREENING,
    next_phase=Phase.SCREENING,          # two further material-level screens follow
    source=_htvs_source("02_1_e_above_hull_roost.ipynb", 30, 1),
    criteria=[
        GateCriterion(
            metric_name="roost_ensemble_mean",
            operator="<",
            threshold=0.1,
            unit="eV/atom",
            description="ROOST-predicted formation energy: the compound must be formable",
            source=_htvs_source("02_1_e_above_hull_roost.ipynb", 30, 1),
        ),
        GateCriterion(
            metric_name="e_hull_from_roost",
            operator="<",
            threshold=0.2,
            unit="eV/atom",
            description="Energy above the convex hull: the compound must be synthesisably close to stable",
            source=_htvs_source("02_1_e_above_hull_roost.ipynb", 30, 1),
        ),
    ],
)

ML_HTVS_ELECTRONIC_GATE = ValidationGate(
    name="htvs-electronic",
    phase=Phase.SCREENING,
    next_phase=Phase.SCREENING,
    source=_htvs_source("02_2_band_gap_roost.ipynb", 8, 1),
    criteria=[
        GateCriterion(
            metric_name="roost_band_gap",
            operator=">",
            threshold=0.5,
            unit="eV",
            description="ROOST-predicted band gap: an electrolyte must not conduct electrons",
            source=_htvs_source("02_2_band_gap_roost.ipynb", 8, 1),
        ),
    ],
)

ML_HTVS_ELECTROCHEMICAL_GATE = ValidationGate(
    name="htvs-electrochemical",
    phase=Phase.SCREENING,
    next_phase=Phase.SIMULATION,         # survivors go to structure generation + DFT
    source=_htvs_source("02_3_stability_window.ipynb", 7, 1),
    criteria=[
        GateCriterion(
            metric_name="upper_bound",
            operator=">",
            threshold=1,
            unit="V",
            description="Anodic limit of the electrochemical stability window",
            source=_htvs_source("02_3_stability_window.ipynb", 7, 1),
        ),
        GateCriterion(
            metric_name="total_range",
            operator=">",
            threshold=0.001,
            unit="V",
            description="Width of the electrochemical stability window",
            source=_htvs_source("02_3_stability_window.ipynb", 7, 1),
        ),
    ],
)

#: The published funnel in order. Each stage's survivors are the next stage's input.
ML_HTVS_CASCADE: list[ValidationGate] = [
    ML_HTVS_THERMODYNAMIC_GATE,
    ML_HTVS_ELECTRONIC_GATE,
    ML_HTVS_ELECTROCHEMICAL_GATE,
]


# --------------------------------------------------------------------------- #
# Synthetic: control-flow smoke test only
#
# These thresholds are conventional engineering targets for a 4.5 V NMC811 solid
# polymer cell. They are NOT taken from any published tool and they are not
# results; they exist so the OPEV loop, the feedback path and the
# negative-evidence machinery can be exercised deterministically in unit tests
# against synthetic_controlflow_fixture.py.
# --------------------------------------------------------------------------- #

_SYNTHETIC_SOURCE = (
    "conventional engineering target for a 4.5 V NMC811 solid polymer cell; "
    "control-flow smoke test only, not from a published tool and not a "
    "result"
)

MATERIAL_GATE = ValidationGate(
    name="synthetic-material",
    phase=Phase.SCREENING,
    source=_SYNTHETIC_SOURCE,
    criteria=[
        GateCriterion(
            metric_name="ionic_conductivity_S_cm",
            operator=">=",
            threshold=1e-4,
            unit="S/cm",
            description="Minimum ionic conductivity at the candidate's stated operating temperature",
            source=_SYNTHETIC_SOURCE,
        ),
        GateCriterion(
            metric_name="electrochemical_window_V",
            operator=">=",
            threshold=4.5,
            unit="V",
            description="Oxidative stability window vs Li/Li+, set by the 4.5 V NMC811 cutoff",
            source=_SYNTHETIC_SOURCE,
        ),
        GateCriterion(
            metric_name="thermal_stability_C",
            operator=">=",
            threshold=150,
            unit="°C",
            description="Onset decomposition temperature",
            source=_SYNTHETIC_SOURCE,
        ),
    ],
)

INTERFACE_GATE = ValidationGate(
    name="synthetic-interface",
    phase=Phase.INTERFACE_VALIDATION,
    source=_SYNTHETIC_SOURCE,
    criteria=[
        GateCriterion(
            metric_name="eis_impedance_50cycles_ohm_cm2",
            operator="stable_within",
            threshold=0.20,
            unit="relative variation",
            description="EIS interfacial impedance stability over 50 cycles",
            source=_SYNTHETIC_SOURCE,
        ),
        GateCriterion(
            metric_name="cei_thickness_growth_nm_per_cycle",
            operator="<=",
            threshold=0.5,
            unit="nm/cycle",
            description="CEI thickness growth rate on cathode side",
            source=_SYNTHETIC_SOURCE,
        ),
    ],
)

CELL_GATE = ValidationGate(
    name="synthetic-cell",
    phase=Phase.CELL_VALIDATION,
    source=_SYNTHETIC_SOURCE,
    criteria=[
        GateCriterion(
            metric_name="capacity_retention_80pct_cycles",
            operator=">=",
            threshold=200,
            unit="cycles",
            description="Cycles to reach 80% SOH at 4.5V cutoff",
            source=_SYNTHETIC_SOURCE,
        ),
        GateCriterion(
            metric_name="coulombic_efficiency_pct",
            operator=">=",
            threshold=99.5,
            unit="%",
            description="Average coulombic efficiency over first 50 cycles",
            source=_SYNTHETIC_SOURCE,
        ),
    ],
)


_SIDE_MAP = {
    "cathode": InterfaceSide.CATHODE,
    "anode": InterfaceSide.ANODE,
    "bulk_electrolyte": InterfaceSide.BULK_ELECTROLYTE,
}

_DEFAULT_LADDER = {
    Phase.SCREENING: Phase.SIMULATION,
    Phase.INTERFACE_VALIDATION: Phase.CELL_VALIDATION,
    Phase.CELL_VALIDATION: Phase.COMPLETED,
}


def run_validation(
    candidate: CandidateRecord,
    gate: ValidationGate,
    metrics: dict[str, Any],
    llm: Any = None,
) -> bool:
    """
    Evaluate a candidate against a validation gate. On failure, the coordination
    layer attributes the failure (interface side, mechanism, next constraints);
    the constraints accumulate on the candidate to drive the next refinement.
    """
    passed, failure_reasons = gate.evaluate(metrics)

    if passed:
        candidate.advance_phase(
            gate.next_phase or _DEFAULT_LADDER.get(gate.phase, Phase.COMPLETED)
        )
    else:
        attribution = coordination.attribute_failure(failure_reasons, metrics, llm)
        side = _SIDE_MAP.get(attribution.get("interface_side"), InterfaceSide.BULK_ELECTROLYTE)
        candidate.record_failure(FailureRecord(
            phase=gate.phase,
            interface_side=side,
            failure_cycle=metrics.get("failure_cycle"),
            failure_mechanism=attribution.get("mechanism", "; ".join(failure_reasons)),
            conditions={k: v for k, v in candidate.test_conditions.items() if not k.startswith("_")},
            gate_name=gate.name,
        ))
        # Accumulate actionable constraints for the next formulation round.
        cons = candidate.test_conditions.setdefault("_constraints", [])
        for c in attribution.get("next_constraints", []):
            if c not in cons:
                cons.append(c)

    return passed
