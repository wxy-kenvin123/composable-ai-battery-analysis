"""
Synthetic control-flow fixture -- unit tests only, never a scientific result.

    THE NUMBERS IN THIS FILE ARE CONSTRUCTED. They must not appear in any
    exported evidence table or in any statement about what the framework can
    predict. They exist
    for exactly one purpose: to give the OPEV loop, the gates, the feedback path
    and the negative-evidence log a deterministic input so their control flow can
    be tested offline. The published-data screening replay is
    ``examples/ml_htvs_screening.py``; it consumes published data and no values
    from here.

The values were chosen to exercise a specific pass/fail sequence. They are
calibrated against the *textbook range* for each material class, stated per
formulation in ``basis``, so that a reader can see no value is physically absurd
-- but calibration is not measurement, and nothing here was measured or computed.

``context_refs`` lists representative literature for the formulation class. Those
papers are NOT the source of the individual numbers below; they are provided so a
reader can locate the material class, not to imply numeric provenance.

Three formulations form an escalation chain, so the loop exercises a different
gate on each round:

  1. ``peo_litfsi_baseline``     : FAILS the MATERIAL gate. Neat PEO oxidizes well
                                   below a 4.5 V cutoff -- the standard reason PEO
                                   is paired with LiFePO4 rather than NMC811. This
                                   is the cheapest possible place to catch it.
  2. ``peo_llzo_crosslinked``    : cross-linking + a ceramic filler clears the
                                   material and interface gates, but FAILS the CELL
                                   gate: nothing yet forms a stable cathode film.
  3. ``peo_llzo_fec_protected``  : adds a cathode-side film former and PASSES all
                                   three gates.
"""

from __future__ import annotations

ILLUSTRATIVE = (
    "synthetic control-flow value calibrated to the textbook range for this "
    "material class; not measured, not computed, and not usable as a result"
)


_BASELINE = {
    "key": "peo_litfsi_baseline",
    "label": "PEO/LiTFSI (EO:Li=16:1)",
    "context_refs": ["[@li2024development]", "[@wang2023computational]"],
    "note": ILLUSTRATIVE,
    "basis": (
        "Neat PEO/LiTFSI reaches ~1e-4 to 1e-3 S/cm at 60 C but its ether oxygens "
        "oxidize at roughly 3.8-4.0 V vs Li/Li+, so it cannot support a 4.5 V NMC811 "
        "cutoff. Interface and cell values follow from that oxidation."
    ),
    "material": {
        "ionic_conductivity_S_cm": 2.1e-4,
        "electrochemical_window_V": 3.9,
        "thermal_stability_C": 180,
        "li_transference_number": 0.25,
        "glass_transition_temp_C": -45,
    },
    "interface": {
        "li_diffusion_coefficient_cm2_s": 1.5e-7,
        "coordination_number_Li_O": 4.2,
        "oxidation_onset_V": 3.9,
        "eis_impedance_50cycles_ohm_cm2": [48, 62, 78, 95, 112],
        "cei_thickness_growth_nm_per_cycle": 0.9,
        "interface_reaction_free_energy_eV": -1.4,
    },
    "cell": {
        "capacity_retention_80pct_cycles": 45,
        "coulombic_efficiency_pct": 97.8,
        "impedance_rise_at_30cycles_pct": 130,
        "failure_cycle": 20,
        "failure_mode": "continuous_oxidative_decomposition",
        "attributed_to": "peo_oxidation_above_4V",
    },
}

_CROSSLINKED = {
    "key": "peo_llzo_crosslinked",
    "label": "Cross-linked PEO/LiTFSI + 10wt% LLZO",
    "context_refs": ["[@wu2023structure]", "[@tang2018simultaneously]", "[@yang2021anomalously]"],
    "note": ILLUSTRATIVE,
    "basis": (
        "Cross-linking plus a garnet filler is the standard route to raise anodic "
        "stability and decomposition onset; composite membranes are commonly reported "
        "stable to ~4.5-5.0 V by linear sweep voltammetry. Nothing here forms a "
        "cathode film, so CEI growth stays near the interface-gate limit and the cell "
        "still fades."
    ),
    "material": {
        "ionic_conductivity_S_cm": 2.6e-4,
        "electrochemical_window_V": 4.6,
        "thermal_stability_C": 210,
        "li_transference_number": 0.42,
        "glass_transition_temp_C": -38,
    },
    "interface": {
        "li_diffusion_coefficient_cm2_s": 1.7e-7,
        "coordination_number_Li_O": 4.0,
        "oxidation_onset_V": 4.6,
        "eis_impedance_50cycles_ohm_cm2": [52, 55, 57, 58, 57],
        "cei_thickness_growth_nm_per_cycle": 0.45,
        "interface_reaction_free_energy_eV": -0.6,
    },
    "cell": {
        "capacity_retention_80pct_cycles": 165,
        "coulombic_efficiency_pct": 99.3,
        "impedance_rise_at_30cycles_pct": 38,
        "failure_cycle": 40,
        "failure_mode": "impedance_sudden_rise",
        "attributed_to": "cathode_cei_instability",
    },
}

_PROTECTED = {
    "key": "peo_llzo_fec_protected",
    "label": "Cross-linked PEO/LiTFSI + 10wt% LLZO + 2wt% FEC",
    "context_refs": ["[@diddens2022modeling]", "[@shen2025machine]"],
    "note": ILLUSTRATIVE,
    "basis": (
        "A fluorinated additive is the conventional way to form a thin, stable "
        "cathode film, which suppresses CEI growth and impedance rise; the small "
        "thermal-stability penalty relative to the filler-only variant reflects the "
        "additive's lower decomposition onset."
    ),
    "material": {
        "ionic_conductivity_S_cm": 2.8e-4,
        "electrochemical_window_V": 4.7,
        "thermal_stability_C": 205,
        "li_transference_number": 0.45,
        "glass_transition_temp_C": -40,
    },
    "interface": {
        "li_diffusion_coefficient_cm2_s": 1.8e-7,
        "coordination_number_Li_O": 3.9,
        "oxidation_onset_V": 4.7,
        "eis_impedance_50cycles_ohm_cm2": [44, 45, 46, 46, 45],
        "cei_thickness_growth_nm_per_cycle": 0.12,
        "interface_reaction_free_energy_eV": -0.3,
    },
    "cell": {
        "capacity_retention_80pct_cycles": 245,
        "coulombic_efficiency_pct": 99.7,
        "impedance_rise_at_30cycles_pct": 11,
        "failure_cycle": 0,
        "failure_mode": "none",
        "attributed_to": "none",
    },
}

_BY_KEY = {f["key"]: f for f in (_BASELINE, _CROSSLINKED, _PROTECTED)}

# Ordered escalation of electrolyte formulations the FormulationDesigner explores.
VARIANTS = [f["label"] for f in (_BASELINE, _CROSSLINKED, _PROTECTED)]

# The failure each variant is meant to remedy, so refinement can be directed by the
# diagnosis rather than by position in the list.
REMEDIES = {
    "peo_litfsi_baseline": None,
    "peo_llzo_crosslinked": "oxidative_stability",
    "peo_llzo_fec_protected": "cathode_cei",
}


def normalize(composition: str) -> str:
    """Map an electrolyte composition string to a fixture key."""
    s = (composition or "").lower()
    if any(k in s for k in ("fec", "fluor", "film-form", "protected")):
        return "peo_llzo_fec_protected"
    if any(k in s for k in ("llzo", "cross-link", "crosslink", "garnet", "inorganic", "filler")):
        return "peo_llzo_crosslinked"
    return "peo_litfsi_baseline"


def get_fixture(composition: str) -> dict:
    """Return the illustrative fixture for an electrolyte composition."""
    return _BY_KEY[normalize(composition)]


def provenance_note(fixture: dict) -> str:
    """One-line provenance string recorded alongside every fixture-derived metric."""
    return (
        f"synthetic-fixture:{fixture['key']} {' '.join(fixture['context_refs'])} "
        f"({fixture['note']}) basis: {fixture['basis']}"
    )
