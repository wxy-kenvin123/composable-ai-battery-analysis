"""Recompute the record-sharing TEA tables (Supplementary Tables 17-20) behind Fig. 6h,i.

Economic basis:
  * the comparator is best-few sharing (pareto_only); the benefit is the evaluations a study avoids when every
    group passes on its full record instead (full_shared), paired by seed over the 60 seeds of the ladder;
  * lambda is the share of avoided evaluations that would also have been a laboratory validation package;
  * no implementation cost; curation (per study) and record maintenance (per year) are hands-on labour, cash at
    the same share eta as the saved laboratory labour; hosting, software, backups and transfer are cash in full.

    G = rho c_MD (dN_ok + dN_fail) + lambda [dN_ok (c_sup + eta c_lab) + dN_fail (c_fail,sup + eta c_fail,lab)]
    NPV = AF [q (G - c_R) - F],  c_R = eta h_R w,  F = eta h_F w + D

Unit costs, finance and the cell gate come from the unchanged TEA script (scripts/tea_unit_costs.py,
data/tea_inputs.json) and Case 2 counts from ../10_case2_sharing/results/sharing_conditions/seed_summary.csv. Inputs are only read; outputs go to rerun/ (or the folder given); compare them
with output/.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PKG = REPO.parent / "10_case2_sharing" / "results" / "sharing_conditions"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "rerun"
OUT.mkdir(parents=True, exist_ok=True)
COMP, FULL, NUMS = "pareto_only", "full_shared", "success_only"
BOOT_RNG, BOOT_N = 20261027, 10000

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"_repo_{name}", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


m38 = load("tea_unit_costs")
V = m38.V

inputs = m38.load_inputs()
seeds = pd.read_csv(PKG / "seed_summary.csv", encoding="utf-8-sig")
assert set(seeds["protocol"]) == {"p80"}


def arm(a):
    d = seeds[seeds["arm"] == a].set_index("seed").sort_index()
    assert len(d) == 60
    ok_lost = d["cross_session_duplicate_attempts"] - d["repeated_hard_fail_attempts"]
    return ok_lost, d["hard_fail_attempts_total"], d


def avoided(frm, to):
    ok_a, fail_a, _ = arm(frm)
    ok_b, fail_b, _ = arm(to)
    assert ok_a.index.equals(ok_b.index)
    return (ok_a - ok_b).to_numpy(float), (fail_a - fail_b).to_numpy(float)


ok_s, fail_s = avoided(COMP, FULL)
nok_s, nfail_s = avoided(NUMS, FULL)
d_ok, d_fail = float(ok_s.mean()), float(fail_s.mean())

fin, rs = inputs["finance"], inputs["record_system"]
R, L, Q = V(fin, "discount_rate"), int(V(fin, "horizon_years")), V(fin, "studies_per_year")
ETA = V(inputs["labor"], "eta_labor_cash_realization")
RHO = V(inputs["compute"], "rho_compute_cash_realization")
uc = m38.unit_costs(inputs)
AF = m38.annuity_factor(R, L)


@dataclass(frozen=True)
class Model:
    u: object
    eta: float = ETA
    rho: float = RHO
    q: float = Q
    af: float = AF
    cur_h: float = V(rs, "curation_hours_per_study")
    op_h: float = V(rs, "annual_maintenance_hours")
    opex: float = V(rs, "annual_digital_opex_usd")
    c_md: float | None = None

    def comp(self, n_ok, n_fail):
        return self.rho * (self.u.c_md if self.c_md is None else self.c_md) * (n_ok + n_fail)

    def lab(self, n_ok, n_fail):
        u, e = self.u, self.eta
        return n_ok * (u.c_sup + e * u.c_lab) + n_fail * (u.c_fail_sup + e * u.c_fail_lab)

    def cur(self):
        return self.eta * self.cur_h * self.u.rate

    def op(self):
        return self.eta * self.op_h * self.u.rate + self.opex

    def G(self, n_ok, n_fail, lam):
        return self.comp(n_ok, n_fail) + lam * self.lab(n_ok, n_fail)

    def net(self, n_ok, n_fail, lam):
        return self.q * (self.G(n_ok, n_fail, lam) - self.cur()) - self.op()

    def npv(self, n_ok, n_fail, lam):
        return self.af * self.net(n_ok, n_fail, lam)

    def lam_be(self, n_ok, n_fail):
        return (self.cur() + self.op() / self.q - self.comp(n_ok, n_fail)) / self.lab(n_ok, n_fail)

    def q_be(self, n_ok, n_fail, lam):
        margin = self.G(n_ok, n_fail, lam) - self.cur()
        return self.op() / margin if margin > 0 else math.inf


base = Model(uc)
assert math.isclose(base.cur_h * uc.rate, uc.cR) and math.isclose(base.op_h * uc.rate + base.opex, uc.F)

# ---------------------------------------------------------------- seed bootstrap (the draws of Fig. 6)
idx = np.random.default_rng(BOOT_RNG).integers(0, 60, size=(BOOT_N, 60))
bo, bf = ok_s[idx].mean(1), fail_s[idx].mean(1)


def ci(v):
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


be_draw = base.lam_be(bo, bf)
be_ci = ci(be_draw)
# Seed bounds are single joint draws, so the feasible and infeasible counts of the same resampled seeds stay
# together; they are the draws whose break-even share is nearest its 2.5th and 97.5th percentiles.
i_more, i_fewer = int(np.argmin(np.abs(be_draw - be_ci[0]))), int(np.argmin(np.abs(be_draw - be_ci[1])))
N_more = (float(bo[i_more]), float(bf[i_more]))
N_fewer = (float(bo[i_fewer]), float(bf[i_fewer]))

ledger = [("Compute", AF * Q * base.comp(d_ok, d_fail), ci(AF * Q * base.comp(bo, bf))),
          ("Experiments, all", AF * Q * base.lab(d_ok, d_fail), ci(AF * Q * base.lab(bo, bf))),
          ("Curation", -AF * Q * base.cur(), None),
          ("Operation", -AF * base.op(), None),
          ("NPV, no experiments", base.npv(d_ok, d_fail, 0.0), ci(base.npv(bo, bf, 0.0))),
          ("NPV, all experiments", base.npv(d_ok, d_fail, 1.0), ci(base.npv(bo, bf, 1.0)))]

_, _, dpo = arm(COMP)
# Fig. 6f resamples the per-seed totals in its plotting order (total descending, then seed)
waste_best_few = (pd.DataFrame({"seed": dpo.index.to_numpy(), "total": (dpo["cross_session_duplicate_attempts"]
                                + dpo["hard_fail_first_attempts"]).to_numpy(float)})
                  .sort_values(["total", "seed"], ascending=[False, True])["total"].to_numpy(float))
groups = [float(dpo[f"s{k}_cross_session_duplicates"].mean()) for k in (2, 3, 4)]

base_out = {
    "G_lambda0": base.G(d_ok, d_fail, 0.0), "G_lambda1": base.G(d_ok, d_fail, 1.0),
    "net_lambda0": base.net(d_ok, d_fail, 0.0), "net_lambda1": base.net(d_ok, d_fail, 1.0),
    "npv_lambda0": base.npv(d_ok, d_fail, 0.0), "npv_lambda0_ci": ci(base.npv(bo, bf, 0.0)),
    "npv_lambda1": base.npv(d_ok, d_fail, 1.0), "npv_lambda1_ci": ci(base.npv(bo, bf, 1.0)),
    "lambda_break_even": base.lam_be(d_ok, d_fail), "lambda_break_even_ci": be_ci,
    "lambda_break_even_eta0": replace(base, eta=0.0).lam_be(d_ok, d_fail),
    "lambda_break_even_eta1": replace(base, eta=1.0).lam_be(d_ok, d_fail),
    "lambda_break_even_rho0": replace(base, rho=0.0).lam_be(d_ok, d_fail),
    "q_break_even_lambda0": base.q_be(d_ok, d_fail, 0.0), "q_break_even_lambda1": base.q_be(d_ok, d_fail, 1.0),
    "failures_value_per_study_lambda0": base.G(nok_s.mean(), nfail_s.mean(), 0.0),
    "failures_value_per_study_lambda1": base.G(nok_s.mean(), nfail_s.mean(), 1.0),
}

# ---------------------------------------------------------------- scenario table (Supplementary Table S19)
rows = []


def scen(name, change, m=base, n=(d_ok, d_fail)):
    rows.append({"scenario": name, "change": change, "avoided_ok": n[0], "avoided_fail": n[1],
                 "eta": m.eta, "rho": m.rho, "q": m.q, "c_md": m.u.c_md if m.c_md is None else m.c_md,
                 "G_lambda0": m.G(*n, 0.0), "G_lambda1": m.G(*n, 1.0),
                 "net_lambda0": m.net(*n, 0.0), "net_lambda1": m.net(*n, 1.0),
                 "npv_lambda0": m.npv(*n, 0.0), "npv_lambda1": m.npv(*n, 1.0),
                 "lambda_break_even": m.lam_be(*n), "q_break_even_lambda1": m.q_be(*n, 1.0)})


scen("Base scenario", "eta 0.5; rho 1; q 10")
scen("No labour cash (eta = 0)", "eta 0", replace(base, eta=0.0))
scen("All labour cash (eta = 1)", "eta 1", replace(base, eta=1.0))
scen("Compute reservation unchanged (rho = 0)", "rho 0", replace(base, rho=0.0))
scen("Neither labour nor compute cash", "eta 0; rho 0", replace(base, eta=0.0, rho=0.0))
scen("Five studies per year", "q 5", replace(base, q=5.0))
scen("Twenty studies per year", "q 20", replace(base, q=20.0))
scen("Fewer avoided evaluations (seed bound)", "joint draw at the 97.5th percentile of the break-even share", n=N_fewer)
scen("More avoided evaluations (seed bound)", "joint draw at the 2.5th percentile of the break-even share", n=N_more)
for basis in ("node_occupancy_vcpu", "cpu_pack_vcpu"):
    scen(f"Tariff basis: {basis.replace('_', ' ')}", f"c_MD = US${uc.c_md_by_basis[basis]:.2f}",
         replace(base, c_md=uc.c_md_by_basis[basis]))
scen("Paid trajectory retention (30 d, high-speed storage)", f"c_MD + US${uc.retention_usd_per_eval:.2f}",
     replace(base, c_md=uc.c_md + uc.retention_usd_per_eval))
scen_df = pd.DataFrame(rows)

# ---------------------------------------------------------------- one-way sensitivities (Supplementary Table S20)


def model_mod(overrides=None, mapping=None, rate_override=None, consumable_scale=1.0, hands_on_scale=1.0, **kw):
    inp = m38.with_overrides(inputs, overrides or {})
    u = m38.unit_costs(inp, mapping=mapping, rate_override=rate_override,
                       consumable_scale=consumable_scale, hands_on_scale=hands_on_scale)
    f, r_ = inp["finance"], inp["record_system"]
    m = Model(u, af=m38.annuity_factor(V(f, "discount_rate"), int(V(f, "horizon_years"))),
              cur_h=V(r_, "curation_hours_per_study"), op_h=V(r_, "annual_maintenance_hours"),
              opex=V(r_, "annual_digital_opex_usd"))
    return replace(m, **kw)


sens = []


def add(label, lo, hi, lo_lab, hi_lab):
    n_lo, n_hi = lo.pop("N", (d_ok, d_fail)), hi.pop("N", (d_ok, d_fail))
    m_lo, m_hi = model_mod(**lo), model_mod(**hi)
    sens.append({"parameter": label, "low_label": lo_lab, "high_label": hi_lab,
                 "be_low": m_lo.lam_be(*n_lo), "be_high": m_hi.lam_be(*n_hi),
                 "npv1_low": m_lo.npv(*n_lo, 1.0), "npv1_high": m_hi.npv(*n_hi, 1.0)})


add("Tariff basis (c_MD)", {"mapping": "cpu_pack_vcpu"}, {"mapping": "node_occupancy_vcpu"},
    f"CPU pack, US${uc.c_md_by_basis['cpu_pack_vcpu']:.2f}", f"node vCPU, US${uc.c_md_by_basis['node_occupancy_vcpu']:.2f}")
add("Exchange rate (JPY per USD)", {"overrides": {"compute.fx_jpy_per_usd": 140}},
    {"overrides": {"compute.fx_jpy_per_usd": 180}}, "140", "180")
add("Avoided evaluations per study (seed bounds)", {"N": N_fewer}, {"N": N_more},
    f"{sum(N_fewer):.1f}", f"{sum(N_more):.1f}")
add("Labor cash realization (eta)", {"eta": 0.0}, {"eta": 1.0}, "0", "1")
add("Compute cash realization (rho)", {"rho": 0.0}, {"rho": 1.0}, "0", "1")
add("Studies per year (q)", {"q": 5.0}, {"q": 20.0}, "5", "20")
add("Valid cells per package", {"overrides": {"laboratory.replicates_per_protocol": 1.5}},
    {"overrides": {"laboratory.replicates_per_protocol": 6}}, "3", "12")
add("Assembly yield", {"overrides": {"laboratory.assembly_yield": 0.80}},
    {"overrides": {"laboratory.assembly_yield": 0.98}}, "80%", "98%")
add("Consumable and utility prices", {"consumable_scale": 0.5}, {"consumable_scale": 2.0}, "0.5x", "2x")
add("Loaded labor rate (USD per h)", {"rate_override": 35.0}, {"rate_override": 100.0}, "35", "100")
add("Hands-on time", {"hands_on_scale": 0.5}, {"hands_on_scale": 2.0}, "0.5x", "2x")
add("Curation time per study (h)", {"overrides": {"record_system.curation_hours_per_study": 1.0}},
    {"overrides": {"record_system.curation_hours_per_study": 4.0}}, "1", "4")
add("Annual record maintenance (h)", {"overrides": {"record_system.annual_maintenance_hours": 20}},
    {"overrides": {"record_system.annual_maintenance_hours": 100}}, "20", "100")
add("Hosting, software and backups (USD per year)", {"overrides": {"record_system.annual_digital_opex_usd": 600.0}},
    {"overrides": {"record_system.annual_digital_opex_usd": 2400.0}}, "600", "2,400")
sens_df = pd.DataFrame(sens)
sens_df["swing"] = (sens_df["be_high"] - sens_df["be_low"]).abs()
sens_df = sens_df.sort_values("swing", ascending=False, kind="stable")

be0, npv1_0 = base.lam_be(d_ok, d_fail), base.npv(d_ok, d_fail, 1.0)
resource_rows = []
for label, keys, lo_v, hi_v in (
    ("Cycler and shared-lab utilization", ("equipment.cycler_utilization", "equipment.shared_lab_utilization"), (0.25, 0.25), (0.80, 0.80)),
    ("Cycling duration per attempted cell (h)", ("laboratory.channel_hours_per_attempt",), (120,), (480,)),
):
    lo_m, hi_m = model_mod(overrides=dict(zip(keys, lo_v))), model_mod(overrides=dict(zip(keys, hi_v)))
    resource_rows.append({"parameter": label, "low": "/".join(map(str, lo_v)), "high": "/".join(map(str, hi_v)),
                          "package_total_low": lo_m.u.package_total, "package_total_high": hi_m.u.package_total,
                          "break_even_unchanged": all(math.isclose(m.lam_be(d_ok, d_fail), be0) for m in (lo_m, hi_m))
                          and all(math.isclose(m.npv(d_ok, d_fail, 1.0), npv1_0) for m in (lo_m, hi_m))})
finance_rows = []
for label, key, lo_v, hi_v in (("Discount rate", "finance.discount_rate", 0.04, 0.12),
                               ("Horizon (years)", "finance.horizon_years", 2, 5)):
    lo_m, hi_m = model_mod(overrides={key: lo_v}), model_mod(overrides={key: hi_v})
    finance_rows.append({"parameter": label, "low": lo_v, "high": hi_v,
                         "npv1_low": lo_m.npv(d_ok, d_fail, 1.0), "npv1_high": hi_m.npv(d_ok, d_fail, 1.0),
                         "break_even_unchanged": all(math.isclose(m.lam_be(d_ok, d_fail), be0) for m in (lo_m, hi_m))})
assert all(r_["break_even_unchanged"] for r_ in resource_rows + finance_rows)

# ---------------------------------------------------------------- cell gate (independent of the counts)
cg = inputs["cell_gate"]
s_e, y0, M = V(cg, "electrolyte_share_of_cell_cost"), V(cg, "baseline_yield"), V(cg, "price_multiplier_example")
gate = []
for label, kw in (
    ("Price x2; yield 95%; e = 1", dict(s_e=s_e, M=M, y=y0, y0=y0, e=1.0)),
    ("Price x2; yield 90%; e = 1", dict(s_e=s_e, M=M, y=0.90, y0=y0, e=1.0)),
    ("Price x2; yield 90%; e = 0.95", dict(s_e=s_e, M=M, y=0.90, y0=y0, e=0.95)),
    ("Price x2; yield 95%; electrolyte share 2%", dict(s_e=0.02, M=M, y=y0, y0=y0, e=1.0)),
    ("Price x2; yield 95%; electrolyte share 10%", dict(s_e=0.10, M=M, y=y0, y0=y0, e=1.0)),
    ("Price x2; yield 95%; added process cost 5%", dict(s_e=s_e, M=M, y=y0, y0=y0, e=1.0, delta=V(cg, "process_cost_example"))),
):
    gate.append({"case": label, **kw, "lifetime_required": m38.lifetime_required(**kw)})

# ---------------------------------------------------------------- write
inventory = m38.inventory_rows(uc, inputs)
rec = next(r_ for r_ in inventory if r_["stage"] == "Record system")
rec.update(flow="curation per study; hosting, software, backups and maintenance per year",
           treatment="running costs of sharing; labour at the eta share, hosting in full")
uc_out = {k: v for k, v in m38.asdict(uc).items() if k != "breakdown"}
uc_out["package_total"] = uc.package_total
uc_out["cash_realizable_package"] = uc.c_md + uc.c_sup + ETA * uc.c_lab
res = {
    "comparator": COMP, "full": FULL, "numbers_without_failures": NUMS,
    "counts": {"n_seeds": len(ok_s), "avoided_ok": d_ok, "avoided_ok_ci": ci(bo), "avoided_fail": d_fail,
               "avoided_fail_ci": ci(bf), "avoided_total": d_ok + d_fail, "avoided_total_ci": ci(bo + bf),
               "N_fewer": N_fewer, "N_more": N_more,
               "failures_avoided_ok": float(nok_s.mean()), "failures_avoided_fail": float(nfail_s.mean()),
               "failures_avoided_total_ci": ci((nok_s + nfail_s)[idx].mean(1)),
               "uninformative_best_few": [float(waste_best_few.mean())] + ci(waste_best_few[idx].mean(1)),
               "uninformative_by_group": groups},
    "parameters": {"eta": ETA, "rho": RHO, "q": Q, "r": R, "L": L, "annuity_factor": AF},
    "record_system": {"curation_hours": base.cur_h, "maintenance_hours": base.op_h, "digital_opex_usd": base.opex,
                      "labour_rate": uc.rate, "curation_cash": base.cur(), "operation_cash": base.op(),
                      "curation_full_rate": uc.cR, "operation_full_rate": uc.F, "implementation_not_counted": uc.K0},
    "unit_costs": uc_out,
    "unit_cost_breakdown": uc.breakdown,
    "base": base_out,
    "ledger": [{"term": t, "pv_usd": v, "ci95": c} for t, v, c in ledger],
    "scenarios": scen_df.to_dict(orient="records"),
    "sensitivities": sens_df.to_dict(orient="records"),
    "resource_sensitivities": resource_rows,
    "finance_sensitivities": finance_rows,
    "cell_gate": gate,
    "inventory": inventory,
    "input_rows": m38.input_rows(inputs),
}
(OUT / "tea_ladder_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
scen_df.to_csv(OUT / "tea_ladder_scenarios.csv", index=False, encoding="utf-8-sig")
sens_df.to_csv(OUT / "tea_ladder_sensitivities.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(gate).to_csv(OUT / "tea_cell_gate.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(m38.input_rows(inputs)).to_csv(OUT / "tea_inputs_table.csv", index=False, encoding="utf-8-sig")


print(f"avoided {d_ok:.2f} + {d_fail:.2f} = {d_ok + d_fail:.2f}; seed bounds {N_fewer} / {N_more}")
print({k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in base_out.items()})
print(pd.DataFrame([{"term": t, "pv": round(v, 1)} for t, v, _ in ledger]).to_string(index=False))
print(scen_df[["scenario", "npv_lambda0", "npv_lambda1", "lambda_break_even"]].round(4).to_string(index=False))
print(sens_df[["parameter", "low_label", "high_label", "be_low", "be_high", "npv1_low", "npv1_high"]].round(4).to_string(index=False))
print(pd.DataFrame(resource_rows).round(2).to_string(index=False))
print(pd.DataFrame(finance_rows).round(1).to_string(index=False))
print(pd.DataFrame(gate)[["case", "lifetime_required"]].round(4).to_string(index=False))

# Self-check all 5 outputs in OUT against output/ when writing to a separate folder and reference exists
ref_dir = REPO / "output"
if OUT.resolve() != ref_dir.resolve() and (ref_dir / "tea_ladder_results.json").exists():
    for fn in ("tea_ladder_scenarios.csv", "tea_ladder_sensitivities.csv", "tea_cell_gate.csv", "tea_inputs_table.csv"):
        a = pd.read_csv(OUT / fn, dtype=str, keep_default_na=False)
        b = pd.read_csv(ref_dir / fn, dtype=str, keep_default_na=False)
        assert a.equals(b), f"{fn} differs from output/"
    j_new = json.loads((OUT / "tea_ladder_results.json").read_text(encoding="utf-8"))
    j_ref = json.loads((ref_dir / "tea_ladder_results.json").read_text(encoding="utf-8"))
    assert j_new == j_ref, "tea_ladder_results.json differs from output/"
    print("PASS: all 5 TEA outputs match output/ identically")
else:
    print(f"Wrote all 5 TEA outputs to {OUT}")
