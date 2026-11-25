"""Literature rules for conductivity priors and predicted processability."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

PACK_PATH = Path(__file__).resolve().parent / "rules" / "rules_prior_mix.json"


def load_pack(path: Path | None = None) -> dict[str, Any]:
    target = Path(path) if path else PACK_PATH
    if not target.is_file():
        raise FileNotFoundError(
            f"The frozen knowledge pack is missing at {target}. Case D will not run "
            "with an improvised pack: the configuration fixes the rules."
        )
    with open(target, encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class KnowledgeOutput:
    """Per-candidate multipliers plus the audit trail that produced them."""

    scale: np.ndarray
    rules_fired: tuple[str, ...]
    enabled: bool
    n_attenuated: int
    min_scale: float
    mean_scale: float
    processability: np.ndarray = field(default_factory=lambda: np.ones(0, dtype=float))
    n_processability_attenuated: int = 0
    min_processability: float = 1.0
    mean_processability: float = 1.0

    def event_row(self) -> dict[str, Any]:
        return {
            "km_enabled": self.enabled,
            "km_rules_fired": ";".join(self.rules_fired) if self.rules_fired else "",
            "km_candidates_attenuated": self.n_attenuated,
            "km_min_prior_shift_scale": round(self.min_scale, 9),
            "km_mean_prior_shift_scale": round(self.mean_scale, 9),
            "km_candidates_processability_attenuated": self.n_processability_attenuated,
            "km_min_processability_scale": round(self.min_processability, 9),
            "km_mean_processability_scale": round(self.mean_processability, 9),
        }


class KnowledgeManager:
    """Applies the frozen pack to a candidate set."""

    def __init__(self, pack: dict[str, Any] | None = None, enabled: bool = True) -> None:
        self.pack = pack or load_pack()
        self.enabled = enabled
        self._rules = {rule["rule_id"]: rule for rule in self.pack["rules"]}

    def _rule_applies(self, rule_id: str) -> bool:
        """Only rules marked ``affects_acquisition`` enter the multiplier.

        The original pack leaves R2 and R3 on, so Case D is unchanged. A later
        pack can retire a rule without deleting the citation by flipping that
        flag. A missing rule is a no-op, not a KeyError.
        """
        rule = self._rules.get(rule_id)
        return bool(rule) and bool(rule.get("affects_acquisition"))

    def _effect_quantity(self, rule_id: str) -> str:
        """Which multiplier a rule writes. Missing quantity keeps the old prior_shift."""
        rule = self._rules.get(rule_id) or {}
        effect = rule.get("executable_effect") or {}
        return str(effect.get("quantity") or "prior_shift_scale")

    def _effect_form(self, rule_id: str) -> str:
        """How a rule is applied. Missing form keeps the historical concentration ramp."""
        rule = self._rules.get(rule_id) or {}
        effect = rule.get("executable_effect") or {}
        return str(effect.get("form") or "linear_ramp")

    def _concentration_rule(self) -> dict[str, Any]:
        return self._rules["KM-R2-stokes-einstein-breaks-at-high-concentration"]

    def _anion_rule(self) -> dict[str, Any]:
        return self._rules["KM-R3-correlation-panel-derived-from-lifsi-subset"]

    def concentration_scale(self, concentration: np.ndarray) -> np.ndarray:
        """Linear ramp between the two concentrations the source actually pins.

        The paper states the relationship holds at 1.0 mol/kg and has broken down
        by 4.0 mol/kg. It says nothing about the shape in between, so a straight
        line between the two stated endpoints is used rather than a sharper
        threshold that would have to be invented.
        """
        effect = self._concentration_rule()["executable_effect"]
        low = float(self._concentration_rule()["applicability_bounds"]["concentration_mol_per_kg"]["trusted_at_or_below"])
        high = float(self._concentration_rule()["applicability_bounds"]["concentration_mol_per_kg"]["documented_breakdown_at_or_above"])
        value_low = float(effect["value_at_or_below_1.0"])
        value_high = float(effect["value_at_or_above_4.0"])
        c = np.asarray(concentration, dtype=float)
        t = np.clip((c - low) / (high - low), 0.0, 1.0)
        return value_low + t * (value_high - value_low)

    def anion_scale(self, anions: Sequence[str]) -> np.ndarray:
        effect = self._anion_rule()["executable_effect"]
        trusted = set(self._anion_rule()["applicability_bounds"]["anion_full_confidence"])
        full = float(effect["value_for_fsi"])
        other = float(effect["value_for_other_anions"])
        return np.array([full if a in trusted else other for a in anions], dtype=float)

    def processability_from_d_hat(self, d_cation_hat: Iterable[float]) -> np.ndarray:


        d = np.asarray(list(d_cation_hat), dtype=float)
        n = int(d.size)
        ones = np.ones(n, dtype=float)
        if not self.enabled or n == 0:
            return ones
        rule_id = "KM-R2-stokes-einstein-breaks-at-high-concentration"
        if not self._rule_applies(rule_id):
            return ones
        if self._effect_form(rule_id) != "pool_relative_stokes_einstein":
            return ones
        if self._effect_quantity(rule_id) != "processability_scale":
            return ones
        effect = self._concentration_rule()["executable_effect"]
        low = float(effect["value_at_lowest_D_in_pool"])
        high = float(effect["value_at_highest_D_in_pool"])
        dmin = float(d.min())
        dmax = float(d.max())
        if dmax == dmin:
            return ones
        u = (d - dmin) / (dmax - dmin)
        return np.clip(low + (high - low) * u, 0.0, 1.0)

    def evaluate(
        self,
        anions: Sequence[str],
        concentrations: Iterable[float],
        d_cation_hat: Iterable[float] | None = None,
    ) -> KnowledgeOutput:
        concentrations = np.asarray(list(concentrations), dtype=float)
        n = len(concentrations)
        ones = np.ones(n, dtype=float)
        if not self.enabled:
            return KnowledgeOutput(
                scale=ones,
                rules_fired=(),
                enabled=False,
                n_attenuated=0,
                min_scale=1.0,
                mean_scale=1.0,
                processability=ones,
                n_processability_attenuated=0,
                min_processability=1.0,
                mean_processability=1.0,
            )

        scale = ones.copy()
        processability = ones.copy()
        fired: list[str] = []
        if self._rule_applies("KM-R1-nernst-einstein-coupling"):
            fired.append("KM-R1-nernst-einstein-coupling")

        if self._rule_applies("KM-R2-stokes-einstein-breaks-at-high-concentration"):
            form = self._effect_form("KM-R2-stokes-einstein-breaks-at-high-concentration")
            if form == "pool_relative_stokes_einstein":
                if d_cation_hat is not None:
                    d_scale = self.processability_from_d_hat(d_cation_hat)
                    if n and len(d_scale) != n:
                        raise ValueError("d_cation_hat length must match the candidate pool")
                    if np.any(d_scale < 1.0):
                        fired.append("KM-R2-stokes-einstein-breaks-at-high-concentration")
                    processability = processability * d_scale
            else:
                c_scale = self.concentration_scale(concentrations)
                if np.any(c_scale < 1.0):
                    fired.append("KM-R2-stokes-einstein-breaks-at-high-concentration")
                if self._effect_quantity("KM-R2-stokes-einstein-breaks-at-high-concentration") == (
                    "processability_scale"
                ):
                    processability = processability * c_scale
                else:
                    scale = scale * c_scale

        if self._rule_applies("KM-R3-correlation-panel-derived-from-lifsi-subset"):
            a_scale = self.anion_scale(anions)
            if np.any(a_scale < 1.0):
                fired.append("KM-R3-correlation-panel-derived-from-lifsi-subset")
            if self._effect_quantity("KM-R3-correlation-panel-derived-from-lifsi-subset") == (
                "processability_scale"
            ):
                processability = processability * a_scale
            else:
                scale = scale * a_scale

        scale = np.clip(scale, 0.0, 1.0)
        processability = np.clip(processability, 0.0, 1.0)
        return KnowledgeOutput(
            scale=scale,
            rules_fired=tuple(fired),
            enabled=True,
            n_attenuated=int(np.count_nonzero(scale < 1.0)),
            min_scale=float(scale.min()) if n else 1.0,
            mean_scale=float(scale.mean()) if n else 1.0,
            processability=processability,
            n_processability_attenuated=int(np.count_nonzero(processability < 1.0)),
            min_processability=float(processability.min()) if n else 1.0,
            mean_processability=float(processability.mean()) if n else 1.0,
        )

    def citation_records(self) -> list[dict[str, Any]]:
        """Every rule with its citekeys and the sentence that supports it."""
        rows = []
        for rule in self.pack["rules"]:
            for citekey in rule["citekeys"]:
                source = next(
                    s for s in self.pack["sources"] if s["citekey"] == citekey
                )
                rows.append(
                    {
                        "rule_id": rule["rule_id"],
                        "citekey": citekey,
                        "title": source["title"],
                        "doi": source["doi"] or "",
                        "doi_absent_reason": source.get("doi_absent_reason", ""),
                        "affects_acquisition": rule["affects_acquisition"],
                        "verbatim_support": rule["verbatim_support"],
                        "support_location": rule["support_location"],
                        "manually_verified": source["manually_verified"],
                        "source": (
                            f"{source['title']} ({source['venue']}, {source['year']})"
                        ),
                    }
                )
        return rows
