"""Pure unit-cost, input-table and parity-gate helpers for record-sharing TEA."""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = ROOT / "data" / "tea_inputs.json"
HOURS_PER_YEAR = 8760.0


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return path


def load_inputs(path: Path = DEFAULT_INPUTS) -> dict:
    with open(require_file(path), encoding="utf-8") as handle:
        return json.load(handle)


def V(section: dict, key: str) -> float:
    """Numeric value of an input entry (``{"value": ...}``) or a bare number."""
    entry = section[key]
    if isinstance(entry, dict):
        return float(entry["value"])
    return float(entry)


def with_overrides(inputs: dict, overrides: dict[str, float]) -> dict:
    """Deep copy of the inputs with ``section.key`` values replaced."""
    new = copy.deepcopy(inputs)
    for dotted, value in overrides.items():
        section, key = dotted.split(".", 1)
        entry = new[section][key]
        if isinstance(entry, dict):
            entry["value"] = value
        else:
            new[section][key] = value
    return new


def input_rows(inputs: dict) -> list[dict]:
    """Flat listing of every documented input (Table S18 source)."""
    rows = []
    for section in ("compute", "laboratory", "labor", "equipment", "record_system", "finance", "reuse", "cell_gate"):
        for key, entry in inputs[section].items():
            if not isinstance(entry, dict) or "value" not in entry:
                continue
            rows.append({
                "section": section,
                "input": key,
                "value": entry["value"],
                "unit": entry.get("unit", ""),
                "evidence_class": entry.get("evidence_class", ""),
                "source": entry.get("source", ""),
                "accessed": entry.get("accessed", ""),
                "note": entry.get("note", ""),
            })
    return rows



def crf(r: float, n: int) -> float:
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def annuity_factor(r: float, L: int) -> float:
    return sum((1 + r) ** (-t) for t in range(1, L + 1))


@dataclass
class UnitCosts:
    c_md: float                     # USD per avoided MD evaluation (selected billing basis)
    c_md_by_basis: dict
    retention_usd_per_eval: float   # optional paid trajectory retention
    gb_per_eval: float
    cpu_hours_per_eval: float
    c_sup: float                    # laboratory supplies and variable utilities per package
    c_lab: float                    # hands-on labor per package at the loaded rate
    c_fail_sup: float               # supplies lost on a repeated intractable attempt
    c_fail_lab: float               # hands-on labor lost on a repeated intractable attempt
    alloc_cycler: float             # capital recovery + maintenance allocation
    alloc_lab: float
    rate: float                     # loaded hourly rate
    hours: float                    # hands-on hours per package
    attempts: float                 # attempted cells per package
    valid_cells: float
    channel_hours: float
    K0: float
    F: float
    cR: float
    breakdown: dict

    @property
    def package_total(self) -> float:
        return self.c_md + self.c_sup + self.c_lab + self.alloc_cycler + self.alloc_lab


def unit_costs(inputs: dict, mapping: str | None = None, rate_override: float | None = None,
               consumable_scale: float = 1.0, hands_on_scale: float = 1.0) -> UnitCosts:
    c, L, lab = inputs["compute"], inputs["laboratory"], inputs["labor"]
    eq, rs, fin = inputs["equipment"], inputs["record_system"], inputs["finance"]

    cpu_per_eval = V(c, "cpu_hours_total_oedb") / V(c, "compositions_oedb")
    fx = V(c, "fx_jpy_per_usd")
    jpy_pt = V(c, "mdx_jpy_per_point")
    node_pts = V(c, "mdx_node_points_per_hour")
    c_md_by = {
        "node_occupancy_physical_core": cpu_per_eval / V(c, "mdx_physical_cores_per_node") * node_pts * jpy_pt / fx,
        "node_occupancy_vcpu": cpu_per_eval / V(c, "mdx_vcpu_per_node") * node_pts * jpy_pt / fx,
        "cpu_pack_vcpu": cpu_per_eval * V(c, "mdx_cpu_pack_points_per_vcpu_hour") * jpy_pt / fx,
    }
    mapping = mapping or c["tariff_mapping_base"]
    if mapping not in c_md_by:
        raise KeyError(f"unknown tariff mapping {mapping}")
    gb = V(c, "trajectory_gb_per_evaluation")
    retention = gb * V(c, "mdx_highspeed_storage_points_per_gb_day") * V(c, "trajectory_retention_days_sensitivity") * jpy_pt / fx

    disc_area_cm2 = math.pi * (V(L, "electrode_disc_diameter_mm") / 20.0) ** 2
    cath_disc = V(L, "cathode_sheet_pack_usd") / V(L, "cathode_sheet_pack_area_cm2") * disc_area_cm2 / V(L, "punching_recovery")
    kit = V(L, "coin_cell_kit_pack_usd") / V(L, "coin_cell_kit_pack_sets")
    valid = V(L, "protocols_per_package") * V(L, "replicates_per_protocol")
    attempts = valid / V(L, "assembly_yield")
    elec = V(L, "electrolyte_mixture_ml") * V(L, "electrolyte_usd_per_ml")
    gas = V(L, "gas_waste_usd_per_package")
    power = V(L, "electricity_kwh_per_attempt") * V(L, "electricity_usd_per_kwh")
    per_attempt = cath_disc + V(L, "anode_disc_usd") + kit + power
    c_sup = (elec + gas + attempts * per_attempt) * consumable_scale
    hours = (V(L, "hands_on_hours_fixed_per_package") + V(L, "hands_on_hours_per_attempt") * attempts) * hands_on_scale
    rate = rate_override if rate_override is not None else (
        V(lab, "median_annual_wage_usd") / V(lab, "paid_hours_per_year") / V(lab, "wage_share_of_compensation"))
    c_lab = hours * rate
    c_fail_sup = (elec + gas) * consumable_scale
    c_fail_lab = V(L, "failed_preparation_hours") * hands_on_scale * rate

    r = V(fin, "discount_rate")
    cyc_capital = V(eq, "cycler_capital_usd")
    cyc_annual = cyc_capital * crf(r, int(V(eq, "cycler_life_years"))) + V(eq, "maintenance_fraction_of_capital") * cyc_capital
    channel_hours_available = V(eq, "cycler_channels") * HOURS_PER_YEAR * V(eq, "cycler_utilization")
    channel_hours = attempts * V(L, "channel_hours_per_attempt")
    alloc_cycler = channel_hours * cyc_annual / channel_hours_available
    lab_capital = V(eq, "shared_lab_capital_usd")
    lab_annual = (lab_capital * crf(r, int(V(eq, "shared_lab_life_years")))
                  + V(eq, "maintenance_fraction_of_capital") * lab_capital + V(eq, "facility_fixed_usd_per_year"))
    lab_hours_available = V(eq, "shared_lab_hours_per_year") * V(eq, "shared_lab_utilization")
    alloc_lab = hours * lab_annual / lab_hours_available

    K0 = V(rs, "implementation_hours") * rate + V(rs, "implementation_other_usd")
    F = V(rs, "annual_maintenance_hours") * rate + V(rs, "annual_digital_opex_usd")
    cR = V(rs, "curation_hours_per_study") * rate

    breakdown = {
        "cpu_hours_per_evaluation": cpu_per_eval,
        "cathode_disc_usd": cath_disc,
        "anode_disc_usd": V(L, "anode_disc_usd"),
        "kit_per_cell_usd": kit,
        "electricity_per_attempt_usd": power,
        "electrolyte_mixture_usd": elec,
        "gas_waste_usd": gas,
        "per_attempt_usd": per_attempt,
        "cycler_annual_usd": cyc_annual,
        "cycler_channel_hours_available": channel_hours_available,
        "shared_lab_annual_usd": lab_annual,
        "shared_lab_hours_available": lab_hours_available,
        "rate_source": ("override" if rate_override is not None else
                        "median wage / paid hours / wage share of compensation"),
    }
    return UnitCosts(
        c_md=c_md_by[mapping], c_md_by_basis=c_md_by, retention_usd_per_eval=retention, gb_per_eval=gb,
        cpu_hours_per_eval=cpu_per_eval, c_sup=c_sup, c_lab=c_lab, c_fail_sup=c_fail_sup, c_fail_lab=c_fail_lab,
        alloc_cycler=alloc_cycler, alloc_lab=alloc_lab, rate=rate, hours=hours, attempts=attempts,
        valid_cells=valid, channel_hours=channel_hours, K0=K0, F=F, cR=cR, breakdown=breakdown,
    )


def lifetime_required(s_e: float, M: float, y: float, y0: float, e: float = 1.0, delta: float = 0.0) -> float:
    """Supplementary Equation 6: relative lifetime at cost parity (R = 1)."""
    return ((1.0 - s_e) + s_e * M + delta) * (y0 / y) / e


def inventory_rows(uc: UnitCosts, inputs: dict) -> list[dict]:
    L = inputs["laboratory"]
    return [
        {"stage": "MD screening", "flow": f"one fresh OEDB-average evaluation ({uc.cpu_hours_per_eval:,.0f} CPU-h, {uc.gb_per_eval:.0f} GB)",
         "treatment": "compute capacity at the public tariff; cash only if a future reservation is reduced (rho)", "evidence": "published x tariff"},
        {"stage": "Electrolyte preparation", "flow": f"{V(L, 'electrolyte_mixture_ml'):g} mL salt + solvents + additives",
         "treatment": "variable consumption; scenario allowance", "evidence": "scenario"},
        {"stage": "Electrode pair", "flow": "coated NMC811 disc and coated graphite disc, current collectors included",
         "treatment": "variable; cathode from a catalogue sheet price, anode allowance", "evidence": "catalogue / scenario"},
        {"stage": "Cell assembly", "flow": f"{uc.valid_cells:.0f} valid cells from {uc.attempts:.2f} attempts; case, spacers, separator, spring",
         "treatment": "variable; kit price; yield applied to every attempt", "evidence": "catalogue / scenario"},
        {"stage": "Formation and cycling", "flow": f"{V(L, 'protocols_per_package'):g} protocols x {V(L, 'replicates_per_protocol'):g} replicates; {V(L, 'channel_hours_per_attempt'):g} channel-h per attempt",
         "treatment": "channel capacity allocation plus incremental electricity", "evidence": "scenario"},
        {"stage": "Hands-on labor", "flow": f"{uc.hours:.2f} h preparation, assembly and review",
         "treatment": "only the eta share becomes avoided cash", "evidence": "published wage anchor x scenario hours"},
        {"stage": "Laboratory operation", "flow": "glove-box gas, consumables, waste; fixed facility",
         "treatment": "variable allowance plus fixed allocation (not cash)", "evidence": "scenario"},
        {"stage": "Equipment", "flow": "cycler; glove box, preparation and characterization equipment",
         "treatment": "capital recovery and maintenance allocation (not cash)", "evidence": "scenario"},
        {"stage": "Record system", "flow": "implementation, curation, hosting, maintenance",
         "treatment": "incremental sharing costs (cash)", "evidence": "scenario hours x published wage anchor"},
        {"stage": "Cell and pack gate", "flow": "yield, delivered energy, lifetime, electrolyte price share",
         "treatment": "analytical parity threshold; absolute cost not estimated", "evidence": "scenario"},
    ]
