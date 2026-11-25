"""
Nernst-Einstein prediction with a correction fitted only to what has been seen.

The mechanism agent answers one question: given the diffusivities the surrogate
currently believes a candidate has, what conductivity would Nernst-Einstein
predict? That prediction becomes the prior mean of the conductivity objective,
so physics enters the acquisition function as a number rather than as a comment.

Two notes that the naming is built around.

**The correction is not a Haven ratio.** Nernst-Einstein takes molar
concentration; OEDB reports molality, and converting between them needs the
molar mass of each lithium salt, which the export does not contain. Rather than
import a molar-mass table nobody in this repository has verified, the molality
is taken as mol/L at 1 kg/L, the SI result is converted to the mS/cm of the
OEDB conductivity, and the remaining conversion is absorbed into the
empirically fitted factor. That factor is therefore the product of a true Haven
ratio and a unit conversion, which is why it is called ``haven_like``
everywhere and why no Haven ratio is reported as a result.

**The correction uses only the past.** The value applied in round t is fitted to
rounds 1..t-1. The campaign snapshots it before calling the oracle, and a test
recomputes it independently from the prefix of the observation log. This is the
single most load-bearing causality property in Case D, so it is enforced in two
places rather than one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class MechanismState:
    """The correction and ratio in force for one round, fitted to earlier rounds."""

    haven_like: float
    log10_haven_like: float
    n_observations: int
    anion_to_cation_ratio: float
    shrinkage_weight: float

    def event_row(self) -> dict[str, Any]:
        return {
            "haven_like_correction": round(self.haven_like, 9),
            "haven_like_n_obs": self.n_observations,
            "haven_like_shrinkage_weight": round(self.shrinkage_weight, 9),
            "anion_to_cation_diffusivity_ratio": round(self.anion_to_cation_ratio, 9),
        }


class MechanismSimulator:
    """Nernst-Einstein with an empirically corrected scale."""

    def __init__(self, contract: dict[str, Any], enabled: bool = True) -> None:
        block = contract["mechanism"]
        constants = block["constants"]
        self.enabled = enabled
        self.faraday = float(constants["faraday_C_per_mol"])
        self.gas_constant = float(constants["gas_constant_J_per_mol_K"])
        self.temperature = float(constants["temperature_K"])
        correction = block["haven_like_correction"]
        self.n0 = float(correction["n0"])
        self.prefactor = (self.faraday * self.faraday) / (
            self.gas_constant * self.temperature
        )

    def raw_sigma(
        self,
        concentration: np.ndarray,
        d_cation: np.ndarray,
        d_anion: np.ndarray,
    ) -> np.ndarray:
        """sigma_NE in mS/cm, the unit of the OEDB conductivity.

        The molality (mol/kg) is taken as mol/L at 1 kg/L and converted to mol/m3
        (x1000); with diffusivities in m2/s the product is in S/m, converted to
        mS/cm (x10). The density difference is left to the fitted correction.
        """
        concentration = np.asarray(concentration, dtype=float)
        return 1.0e4 * self.prefactor * concentration * (
            np.asarray(d_cation, dtype=float) + np.asarray(d_anion, dtype=float)
        )

    def state_from_history(
        self, history: Sequence[dict[str, float]]
    ) -> MechanismState:
        """Fit the correction and the diffusivity ratio to past observations only.

        ``history`` carries one entry per usable revealed point, each with the
        observed conductivity, cation and anion diffusivities and concentration.
        An empty history yields the prior: correction 1.0, ratio 1.0.
        """
        usable = [
            h
            for h in history
            if h.get("sigma") is not None
            and h.get("d_cation")
            and h.get("d_anion")
            and h.get("concentration")
        ]
        n = len(usable)
        if n == 0:
            return MechanismState(
                haven_like=1.0,
                log10_haven_like=0.0,
                n_observations=0,
                anion_to_cation_ratio=1.0,
                shrinkage_weight=0.0,
            )

        ratios = np.array([h["d_anion"] / h["d_cation"] for h in usable], dtype=float)
        ratio = float(np.median(ratios))

        raw = self.raw_sigma(
            np.array([h["concentration"] for h in usable]),
            np.array([h["d_cation"] for h in usable]),
            np.array([h["d_anion"] for h in usable]),
        )
        observed = np.array([h["sigma"] for h in usable], dtype=float)
        log_ratio = np.log10(raw / observed)
        weight = n / (n + self.n0)
        shrunk = weight * float(log_ratio.mean())
        return MechanismState(
            haven_like=float(10.0 ** shrunk),
            log10_haven_like=shrunk,
            n_observations=n,
            anion_to_cation_ratio=ratio,
            shrinkage_weight=weight,
        )

    def prior_m1(
        self,
        state: MechanismState,
        concentration: np.ndarray,
        d_cation_hat: np.ndarray,
        fallback: float,
        conductivity_scale: np.ndarray | None = None,
    ) -> np.ndarray:
        """Prior for -log10(sigma) from predicted cation diffusivity.

        ``d_cation_hat`` is the surrogate's posterior mean for the candidate's
        Li diffusivity, so the mechanism never needs an unrevealed label. The
        anion diffusivity is taken as that value times the running observed
        ratio.

        ``conductivity_scale`` is optional. When set, it multiplies the
        corrected Nernst-Einstein conductivity before the logarithm
        (``σ_used = (σ_NE / haven_like) * λ``). The default ``None`` leaves
        the historical prior unchanged.
        """
        if not self.enabled:
            return np.full(len(np.atleast_1d(concentration)), float(fallback))
        d_cation_hat = np.asarray(d_cation_hat, dtype=float)
        d_anion_hat = d_cation_hat * state.anion_to_cation_ratio
        raw = self.raw_sigma(concentration, d_cation_hat, d_anion_hat)
        corrected = raw / state.haven_like
        if conductivity_scale is not None:
            corrected = corrected * np.asarray(conductivity_scale, dtype=float)
        # A non-positive prediction cannot be logged; fall back to the observed
        # mean for that candidate rather than clipping to an arbitrary floor.
        out = np.full(raw.shape, float(fallback))
        live = corrected > 0.0
        out[live] = -np.log10(corrected[live])
        return out


def d_from_minimize_form(m2: np.ndarray) -> np.ndarray:
    """Invert m2 = -log10(D_Li) back to linear diffusivity."""
    return np.power(10.0, -np.asarray(m2, dtype=float))


def sigma_from_minimize_form(m1: float) -> float:
    return float(10.0 ** (-m1))


@dataclass
class MemoryFeedback:
    """MKB-derived noise inflation, from the residuals of earlier rounds."""

    k: int
    residual_scale: float
    cap: float
    enabled: bool = True

    def inflation(self, residuals: Sequence[float]) -> tuple[float, dict[str, Any]]:
        if not self.enabled or not residuals:
            return 1.0, {
                "mkb_enabled": self.enabled,
                "mkb_residuals_used": 0,
                "mkb_noise_inflation": 1.0,
            }
        window = list(residuals)[-self.k :]
        magnitude = float(np.mean(np.abs(np.asarray(window, dtype=float))))
        value = min(1.0 + magnitude / self.residual_scale, self.cap)
        return value, {
            "mkb_enabled": True,
            "mkb_residuals_used": len(window),
            "mkb_mean_abs_residual": round(magnitude, 9),
            "mkb_noise_inflation": round(value, 9),
        }
