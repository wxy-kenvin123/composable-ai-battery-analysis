"""Checks applied before and after each electrolyte evaluation."""

from __future__ import annotations

from ..candidate_record import Phase
from ..validation_gates import GateCriterion, ValidationGate

_OEDB_CITE = "[@nakamura2026oedb] doi:10.1038/s41524-026-02093-y"
_CONTRACT = "electrolyte search contract"


def oedb_unseen_gate() -> ValidationGate:
    source = (
        f"protocol check defined in {_CONTRACT}: the planner must return a member of "
        "the 1,872-point Li universe that has not been revealed. Not a scientific "
        "criterion."
    )
    return ValidationGate(
        name="oedb-candidate-unseen",
        phase=Phase.SCREENING,
        next_phase=Phase.SIMULATION,
        source=source,
        criteria=[
            GateCriterion(
                metric_name="candidate_is_unseen",
                operator=">=",
                threshold=1,
                unit="boolean",
                description=(
                    "Proposed candidate is in the pinned universe and has not been "
                    "revealed in this campaign"
                ),
                source=source,
            )
        ],
    )


def oedb_data_available_gate() -> ValidationGate:
    source = (
        f"Missing conductivity is handled by the policy in "
        f"{_CONTRACT}: ids 3010, 3011, 3370, 3428 and 3503 carry no ionic conductivity in "
        f"the official export ({_OEDB_CITE}). Selecting one is a data gap, not a chemical "
        "result."
    )
    return ValidationGate(
        name="oedb-data-available",
        phase=Phase.SIMULATION,
        next_phase=Phase.CELL_VALIDATION,
        source=source,
        criteria=[
            GateCriterion(
                metric_name="label_status_ok",
                operator=">=",
                threshold=1,
                unit="boolean",
                description=(
                    "The revealed row has an ionic-conductivity label, so all three "
                    "objectives are defined"
                ),
                source=source,
            )
        ],
    )


def oedb_material_conductivity_gate(threshold_mS_cm: float) -> ValidationGate:
    """A material-level bar at an ionic conductivity of 1e-4 S/cm.

    OEDB reports mS/cm, so the same number arrives here as 0.1. Failing it is a
    screening rejection at the material level. The formulation's labels are real
    and stay real; the candidate is below the bar, which is not a chemical
    failure.
    """
    source = (
        f"material-level acceptance at ionic conductivity >= 1e-4 S/cm = "
        f"{threshold_mS_cm} mS/cm"
    )
    return ValidationGate(
        name="oedb-material-conductivity",
        phase=Phase.CELL_VALIDATION,
        next_phase=Phase.CELL_VALIDATION,
        source=source,
        criteria=[
            GateCriterion(
                metric_name="ionic_conductivity_mS_cm",
                operator=">=",
                threshold=threshold_mS_cm,
                unit="mS/cm",
                description=(
                    "Revealed formulation meets the material-level "
                    "conductivity bar; a miss is a screening rejection, not a "
                    "chemical failure"
                ),
                source=source,
            )
        ],
    )


def oedb_running_best_gate(prior_best: float, n_prior: int) -> ValidationGate:
    """Did this round improve on the best scalarised value seen before it?"""
    threshold = round(float(prior_best), 9)
    # Kept terse on purpose: this string is written once per round of every run, so
    # boilerplate here becomes megabytes in the exported table. What varies per row
    # -- the threshold and how many observations produced it -- is what it carries.
    source = (
        f"min(scalarised) over the {n_prior} preceding usable observations = "
        f"{threshold:.9f}; no later round or unrevealed label used"
    )
    return ValidationGate(
        name="oedb-running-best",
        phase=Phase.CELL_VALIDATION,
        next_phase=Phase.COMPLETED,
        source=source,
        criteria=[
            GateCriterion(
                metric_name="scalarised_value",
                operator="<",
                threshold=threshold,
                unit="augmented Chebyshev, minimised",
                description=(
                    "Revealed point improves on the best scalarised value among prior "
                    "rounds; an optimisation-progress audit, not physical acceptance"
                ),
                source=source,
            )
        ],
    )
