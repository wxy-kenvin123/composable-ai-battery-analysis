"""Mechanism priors, surrogate fitting and acquisition decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from . import acquisition as acq
from .knowledge import KnowledgeManager, KnowledgeOutput
from .mechanism import (
    MechanismSimulator,
    MechanismState,
    MemoryFeedback,
    d_from_minimize_form,
    sigma_from_minimize_form,
)
from .surrogate import GPFit, fit_gp

MIX_WITH_OBSERVED_MEAN = "mix_with_observed_mean"
SCALE_NE_CONDUCTIVITY = "scale_ne_conductivity"
_KNOWLEDGE_COMBINATIONS = frozenset({MIX_WITH_OBSERVED_MEAN, SCALE_NE_CONDUCTIVITY})


def resolve_knowledge_combination(
    contract: dict[str, Any], manager: KnowledgeManager
) -> str:


    raw = contract.get("knowledge_combination")
    if not raw:
        design = (getattr(manager, "pack", None) or {}).get(
            "acquisition_influence_design"
        ) or {}
        raw = design.get("knowledge_combination")
    value = str(raw or MIX_WITH_OBSERVED_MEAN)
    if value not in _KNOWLEDGE_COMBINATIONS:
        known = ", ".join(sorted(_KNOWLEDGE_COMBINATIONS))
        raise ValueError(f"unknown knowledge_combination {value!r}; expected one of {known}")
    return value


@dataclass
class ArmConfig:
    """Which of the switches an arm has on."""

    name: str
    km: bool
    mkb: bool
    mechanism: bool
    coordinator: bool
    random_only: bool = False

    @classmethod
    def from_contract(cls, contract: dict[str, Any], name: str) -> "ArmConfig":
        for arm in contract["arms"]:
            if arm["name"] == name:
                return cls(
                    name=name,
                    km=bool(arm["km"]),
                    mkb=bool(arm["mkb"]),
                    mechanism=bool(arm["mechanism"]),
                    coordinator=bool(arm["coordinator"]),
                    random_only=(name == "random"),
                )
        known = ", ".join(a["name"] for a in contract["arms"])
        raise KeyError(f"unknown arm {name!r}; the contract defines: {known}")


@dataclass
class CampaignState:
    """Everything the decision core is allowed to see.

    Note what is absent: there is no reference to the oracle, the evaluator, or
    any label the oracle has not already handed over. A stage that wanted to peek
    would have nothing to peek at.
    """

    contract: dict[str, Any]
    space: Any                      # OEDBSearchSpace
    arm: ArmConfig
    seed: int
    observed_ids: list[int] = field(default_factory=list)
    #: usable observations only; data_missing points never land here
    m: list[tuple[float, ...]] = field(default_factory=list)
    usable_ids: list[int] = field(default_factory=list)
    diffusivity_history: list[dict[str, float]] = field(default_factory=list)
    scalarised_residuals: list[float] = field(default_factory=list)
    round_index: int = 0            # 1-based over optimised rounds


    objective_mode: str = "m1m2m3"

    def record(self, observation: Any) -> None:
        self.observed_ids.append(observation.candidate_id)
        if not observation.usable:
            return
        self.usable_ids.append(observation.candidate_id)
        if self.objective_mode == "sigma_tplus":
            total = observation.d_cation + observation.d_anion
            self.m.append((observation.m1, -(observation.d_cation / total)))
        elif self.objective_mode == "m1m2m3":
            self.m.append((observation.m1, observation.m2, observation.m3))
        else:
            raise ValueError(f"unknown objective_mode {self.objective_mode!r}")
        self.diffusivity_history.append(
            {
                "sigma": sigma_from_minimize_form(observation.m1),
                "d_cation": observation.d_cation,
                "d_anion": observation.d_anion,
                "concentration": observation.concentration,
            }
        )

    def observed_m2(self) -> np.ndarray:
        """``-log10(D_Li)`` for the usable observations, in their recorded order.

        In ``m1m2m3`` mode this is exactly ``m[:, 1]``; a test pins that. It is a
        separate accessor because in ``sigma_tplus`` mode column 1 is the
        transference number, and the mechanism prior still needs the
        diffusivity it was always given.
        """
        return np.array(
            [-math.log10(row["d_cation"]) for row in self.diffusivity_history],
            dtype=float,
        )

    @property
    def n_usable(self) -> int:
        return len(self.m)

    def unseen_ids(self) -> tuple[int, ...]:
        return self.space.unseen_ids(self.observed_ids)

    def observed_features(self) -> np.ndarray:
        return np.asarray(self.space.feature_matrix(self.usable_ids), dtype=float)

    def observed_concentrations(self) -> np.ndarray:
        return np.array(
            [self.space.concentration(i) for i in self.usable_ids], dtype=float
        )

    def observed_anions(self) -> list[str]:
        return [self.space.anion(i) for i in self.usable_ids]


@dataclass(frozen=True)
class MechanismStageOutput:
    state: MechanismState
    m1_prior_unseen: np.ndarray
    m1_prior_observed: np.ndarray
    m2_hat_unseen: np.ndarray
    m3_hat_unseen: np.ndarray
    gp_m2: GPFit | None
    gp_m3: GPFit | None
    enabled: bool


    tplus_hat_unseen: np.ndarray | None = None
    gp_tplus: GPFit | None = None

    def event_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"mechanism_enabled": self.enabled}
        row.update(self.state.event_row())
        if self.gp_m2 is not None and self.gp_m3 is not None:
            row["gp_m2_lengthscale"] = self.gp_m2.lengthscale
            row["gp_m3_lengthscale"] = self.gp_m3.lengthscale
        if self.gp_tplus is not None:
            row["gp_tplus_lengthscale"] = self.gp_tplus.lengthscale
        return row


@dataclass(frozen=True)
class Proposal:
    candidate_id: int
    expected_improvement: float
    tied: bool
    n_tied: int
    predicted_scalarised: float
    predicted_variance: float
    incumbent: float

    weights: tuple[float, ...]
    noise_inflation: float
    gp: GPFit | None
    decision_source: str

    def event_row(self) -> dict[str, Any]:
        return {
            "selected_id": self.candidate_id,
            "expected_improvement": _round(self.expected_improvement, 12),
            "acquisition_tie": self.tied,
            "acquisition_n_tied": self.n_tied,
            "predicted_scalarised": _round(self.predicted_scalarised, 9),
            "predicted_variance": _round(self.predicted_variance, 12),
            "incumbent_scalarised": _round(self.incumbent, 9),
            "parego_w1": _round(self.weights[0], 9),
            "parego_w2": _round(self.weights[1], 9),
            # Two-objective runs leave the third column blank rather than
            # inventing a zero weight that was never in the lattice.
            "parego_w3": _round(self.weights[2], 9) if len(self.weights) > 2 else "",
            "noise_inflation": _round(self.noise_inflation, 9),
            "decision_source": self.decision_source,
        }


def _round(value: float, digits: int) -> float | str:
    """Round for export, leaving the random arm's NaNs legible rather than 'nan'."""
    if value != value:  # NaN
        return ""
    return round(float(value), digits)


def knowledge_stage(
    manager: KnowledgeManager, state: CampaignState, unseen: Sequence[int]
) -> KnowledgeOutput:
    anions = [state.space.anion(i) for i in unseen]
    concentrations = [state.space.concentration(i) for i in unseen]
    return manager.evaluate(anions, concentrations)


def mechanism_stage(
    simulator: MechanismSimulator,
    manager: KnowledgeManager,
    state: CampaignState,
    knowledge: KnowledgeOutput,
    unseen: Sequence[int],
) -> MechanismStageOutput:
    contract = state.contract
    grid = contract["surrogate"]["grid"]
    jitter = contract["surrogate"]["cholesky_jitter"]

    m = np.asarray(state.m, dtype=float)
    m1_bar = float(m[:, 0].mean())
    x_obs = state.observed_features()
    x_unseen = np.asarray(state.space.feature_matrix(unseen), dtype=float)

    if not simulator.enabled:
        return MechanismStageOutput(
            state=simulator.state_from_history([]),
            m1_prior_unseen=np.zeros(len(unseen), dtype=float),
            m1_prior_observed=np.zeros(len(x_obs), dtype=float),
            m2_hat_unseen=np.zeros(len(unseen), dtype=float),
            m3_hat_unseen=np.zeros(len(unseen), dtype=float),
            gp_m2=None,
            gp_m3=None,
            enabled=False,
        )

    two_objective = getattr(state, "objective_mode", "m1m2m3") == "sigma_tplus"
    gp_tplus = None
    tplus_unseen = None
    if two_objective:
        # Column 1 is -t_plus here, so the diffusivity the mechanism prior needs
        # comes from the revealed d_cation rather than from the objective matrix.
        gp_m2 = fit_gp(x_obs, state.observed_m2(), grid, jitter)
        gp_m3 = None
        gp_tplus = fit_gp(x_obs, m[:, 1], grid, jitter)
        m2_unseen, _ = gp_m2.predict_residual(x_unseen)
        tplus_unseen, _ = gp_tplus.predict_residual(x_unseen)
        m3_unseen = np.zeros(len(unseen), dtype=float)
    else:
        # Call order below is the Case D order and is left exactly as it was.
        gp_m2 = fit_gp(x_obs, m[:, 1], grid, jitter)
        gp_m3 = fit_gp(x_obs, m[:, 2], grid, jitter)
        m2_unseen, _ = gp_m2.predict_residual(x_unseen)
        m3_unseen, _ = gp_m3.predict_residual(x_unseen)
    m2_observed, _ = gp_m2.predict_residual(x_obs)

    # Fitted to revealed rounds only; the campaign calls this before the oracle.
    mech_state = simulator.state_from_history(state.diffusivity_history)

    conc_unseen = np.array([state.space.concentration(i) for i in unseen], dtype=float)
    conc_observed = state.observed_concentrations()

    m1_mech_unseen = simulator.prior_m1(
        mech_state, conc_unseen, d_from_minimize_form(m2_unseen), m1_bar
    )
    m1_mech_observed = simulator.prior_m1(
        mech_state, conc_observed, d_from_minimize_form(m2_observed), m1_bar
    )

    lam_unseen = knowledge.scale
    lam_observed = manager.evaluate(state.observed_anions(), conc_observed).scale

    combination = resolve_knowledge_combination(contract, manager)
    if combination == SCALE_NE_CONDUCTIVITY:
        m1_prior_unseen = simulator.prior_m1(
            mech_state,
            conc_unseen,
            d_from_minimize_form(m2_unseen),
            m1_bar,
            conductivity_scale=lam_unseen,
        )
        m1_prior_observed = simulator.prior_m1(
            mech_state,
            conc_observed,
            d_from_minimize_form(m2_observed),
            m1_bar,
            conductivity_scale=lam_observed,
        )
    else:
        m1_prior_unseen = lam_unseen * m1_mech_unseen + (1.0 - lam_unseen) * m1_bar
        m1_prior_observed = lam_observed * m1_mech_observed + (1.0 - lam_observed) * m1_bar

    return MechanismStageOutput(
        state=mech_state,
        m1_prior_unseen=m1_prior_unseen,
        m1_prior_observed=m1_prior_observed,
        m2_hat_unseen=m2_unseen,
        m3_hat_unseen=m3_unseen,
        gp_m2=gp_m2,
        gp_m3=gp_m3,
        enabled=True,
        tplus_hat_unseen=tplus_unseen,
        gp_tplus=gp_tplus,
    )


def prior_columns(
    state: Any, mechanism: MechanismStageOutput, m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The mechanism-informed prior, stacked to match the objective columns.

    Shared by :func:`plan_stage` and ``batch_policy.plan_one`` so the two cannot
    drift apart; the three-objective branch is the original expression.
    """
    if getattr(state, "objective_mode", "m1m2m3") == "sigma_tplus":
        if mechanism.tplus_hat_unseen is None:
            raise RuntimeError(
                "objective_mode='sigma_tplus' needs the t_plus surrogate; "
                "mechanism_stage returned tplus_hat_unseen=None"
            )
        prior_obs = np.column_stack([mechanism.m1_prior_observed, m[:, 1]])
        prior_unseen = np.column_stack(
            [mechanism.m1_prior_unseen, mechanism.tplus_hat_unseen]
        )
        return prior_obs, prior_unseen
    prior_obs = np.column_stack([mechanism.m1_prior_observed, m[:, 1], m[:, 2]])
    prior_unseen = np.column_stack(
        [mechanism.m1_prior_unseen, mechanism.m2_hat_unseen, mechanism.m3_hat_unseen]
    )
    return prior_obs, prior_unseen


def plan_stage(
    state: CampaignState,
    mechanism: MechanismStageOutput,
    memory: MemoryFeedback,
    unseen: Sequence[int],
    lattice: np.ndarray,
    ei_scale: np.ndarray | None = None,
) -> Proposal:
    contract = state.contract
    acq_cfg = contract["acquisition"]
    grid = contract["surrogate"]["grid"]
    jitter = contract["surrogate"]["cholesky_jitter"]
    rho = float(acq_cfg["rho"])

    weights = acq.weight_for_round(lattice, state.round_index)
    m = np.asarray(state.m, dtype=float)
    lower, upper = acq.observed_bounds(m)

    s_obs = acq.augmented_chebyshev(
        acq.min_max_normalise(m, lower, upper), weights, rho
    )

    if mechanism.enabled:
        prior_obs, prior_unseen = prior_columns(state, mechanism, m)
        mu0_obs = acq.augmented_chebyshev(
            acq.min_max_normalise(prior_obs, lower, upper), weights, rho
        )
        mu0_unseen = acq.augmented_chebyshev(
            acq.min_max_normalise(prior_unseen, lower, upper), weights, rho
        )
    else:
        # A zero prior leaves fit_gp to standardise the raw scalarised values,
        # which is precisely the constant observed-mean GP the contract specifies
        # for parego_plain.
        mu0_obs = np.zeros(len(s_obs), dtype=float)
        mu0_unseen = np.zeros(len(unseen), dtype=float)

    inflation, _ = memory.inflation(state.scalarised_residuals)

    x_obs = state.observed_features()
    x_unseen = np.asarray(state.space.feature_matrix(unseen), dtype=float)
    gp = fit_gp(x_obs, s_obs - mu0_obs, grid, jitter, noise_inflation=inflation)
    residual_mean, residual_var = gp.predict_residual(x_unseen)
    mean = mu0_unseen + residual_mean

    incumbent = float(s_obs.min())
    ei = acq.expected_improvement(
        mean, residual_var, incumbent, float(acq_cfg["expected_improvement"]["xi"])
    )
    if ei_scale is not None:
        # Case D-R's failure-to-constraint feedback enters here and nowhere else:
        # a multiplicative attenuation of expected improvement, never a hard veto.
        # The first Case D never passes this argument, so its behaviour is
        # untouched byte for byte.
        ei = ei * np.asarray(ei_scale, dtype=float)
    pick = acq.argmax_first(ei)

    return Proposal(
        candidate_id=int(unseen[pick.index]),
        expected_improvement=pick.value,
        tied=pick.tied,
        n_tied=pick.n_tied,
        predicted_scalarised=float(mean[pick.index]),
        predicted_variance=float(residual_var[pick.index]),
        incumbent=incumbent,
        weights=tuple(float(w) for w in weights),
        noise_inflation=inflation,
        gp=gp,
        decision_source="parego_ei",
    )


def random_proposal(state: CampaignState, unseen: Sequence[int]) -> Proposal:
    """Uniform choice, with the stream derived from (seed, round) like every arm."""
    rng = np.random.default_rng([state.seed, state.round_index])
    index = int(rng.integers(0, len(unseen)))
    nan = float("nan")
    return Proposal(
        candidate_id=int(unseen[index]),
        expected_improvement=nan,
        tied=False,
        n_tied=1,
        predicted_scalarised=nan,
        predicted_variance=nan,
        incumbent=nan,
        weights=(nan, nan, nan),
        noise_inflation=1.0,
        gp=None,
        decision_source="uniform_random",
    )


def observed_scalarised(
    state: CampaignState, weights: np.ndarray, rho: float, observation: Any
) -> float:
    """Scalarised value of a just-revealed point under this round's weights.

    Used for the MKB residual. The bounds come from the observations that existed
    before this one, so the number is comparable with what the planner predicted.
    """
    m = np.asarray(state.m, dtype=float)
    lower, upper = acq.observed_bounds(m)
    if getattr(state, "objective_mode", "m1m2m3") == "sigma_tplus":
        total = observation.d_cation + observation.d_anion
        row = np.array([[observation.m1, -(observation.d_cation / total)]], dtype=float)
    else:
        row = np.array([[observation.m1, observation.m2, observation.m3]], dtype=float)
    return float(
        acq.augmented_chebyshev(
            acq.min_max_normalise(row, lower, upper), weights, rho
        )[0]
    )
