"""Recompute and verify panels a, c, d, e, f from census_755.csv, scored_tools_125.csv and cross_scale_summary.json."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd

SEC = Path(__file__).resolve().parents[1]
DATA = SEC / "data"

CLUSTER_NAMES = [
    ("1", "Atomistic Dynamics & Transport"),
    ("2", "Molecular & Polymer Properties"),
    ("3", "Solid-Material Properties"),
    ("4", "Microstructure & Manufacturing"),
    ("5", "Materials Design & Optimization"),
    ("6", "Battery State & Lifetime"),
    ("7", "Electrode Performance Prediction"),
    ("8", "Structural Stability & Mechanisms"),
]
PANEL_A_METRICS = [
    ("Code", "code_available", "Code availability established by inspected, attributable retrieved content"),
    ("Data", "data_available", "Data availability established by inspected, attributable retrieved content"),
    ("Request", "data_on_request", "Final adjudicated data status is on_request"),
    ("Both", "both_available", "Verified code AND verified data"),
    ("Any", "any_resource", "Verified code OR verified data"),
    (
        "Unknown",
        "unknown",
        "Code or data status is unknown, access_failed or not_reviewed: a statement or link exists "
        "but the retrieved evidence could not establish either availability or unavailability "
        "(narrow definition; papers that make no statement are not counted)",
    ),
]
MODULES_ORDER = [
    "Knowledge Miner",
    "Materials Designer",
    "HT Predictor",
    "Mechanism Simulator",
    "Experiment Planner",
    "Process Coordinator",
]
ITEMS_ORDER = ["R2", "R3", "R4", "R6", "R5", "R7", "D1", "D2", "D3"]
ITEM_DEF = (
    "All primary-paper entries with R1 = True under the program-file policy; "
    "denominator is known values for this item, including entries with uncertain total RRL"
)
PANEL_F_DEF = (
    "Deduplicated primary resources, bounded CSV/TSV scan; header/identifier matches are candidates, "
    "not experimentally validated cross-scale bridges; a nonempty cycling-labelled field (including zero) "
    "does not prove a cycled cell"
)
PANEL_F_ROWS = [
    ("resources", "Unique resources scanned", "unique_resources"),
    ("resources", "With scale-labelled headers", "attributed_resource_count"),
    ("resources", "Headers cover ≥2 scales", "scale_coverage_counts[2]+[3]"),
    ("resources", "Same candidate at ≥2 scales", "resources_candidate_two_scales"),
    ("resources", "Material → cycling-label match", "resources_material_to_cycled_cell"),
    ("pairs", "Possible resource pairs", "possible_pairs"),
    ("pairs", "Share identifier candidates", "pairs_sharing_candidates"),
    ("pairs", "Candidate pairs across scales", "pairs_cross_scale"),
    ("pairs", "Material → cycling-label match", "pairs_material_to_cycled_cell"),
]


def read(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def yes(v: str) -> bool:
    return str(v).strip().lower() == "true"


def cl(v: str) -> int:
    return int(str(v).removeprefix("C"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=SEC / "rerun")
    args = ap.parse_args()
    out_dir = args.outdir if args.outdir.is_absolute() else (SEC / args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    c755 = read("census_755.csv")
    t125 = read("scored_tools_125.csv")
    cs = json.loads((DATA / "cross_scale_summary.json").read_text(encoding="utf-8"))
    assert len(c755) == 755 and len(t125) == 125

    metrics = {}
    for r in c755:
        code, data = yes(r["code_available"]), yes(r["data_available"])
        assert yes(r["both_available"]) == (code and data)
        assert yes(r["any_resource"]) == (code or data)
        metrics[r["paper_id"]] = {
            "Code": code, "Data": data,
            "Request": r["data_status"] == "on_request",
            "Both": code and data, "Any": code or data,
            "Unknown": any(r[x] in ("unknown", "access_failed", "not_reviewed") for x in ("code_status", "data_status")),
        }

    cluster_counts = Counter(cl(r["cluster"]) for r in c755)
    pa_out = []
    for c_str, c_name in CLUSTER_NAMES:
        c = int(c_str)
        den = cluster_counts[c]
        for metric_name, field_name, definition in PANEL_A_METRICS:
            cnt = sum(metrics[x["paper_id"]][metric_name] for x in c755 if cl(x["cluster"]) == c)
            pa_out.append({
                "cluster": c_str, "cluster_name": c_name, "metric": metric_name, "field": field_name,
                "count": cnt, "denominator": den, "percent": 100.0 * cnt / den, "definition": definition,
            })
    pd.DataFrame(pa_out).to_csv(out_dir / "panel_a_clusters.csv", index=False)

    for r in t125:
        rrl = sum(yes(r[k]) for k in ("R2", "R3", "R4", "R6")) + int(yes(r["R5"]) or yes(r["R7"]))
        crl = sum(yes(r[k]) for k in ("D1", "D2", "D3"))
        assert int(float(r["RRL"])) == rrl and int(float(r["CRL"])) == crl

    pc_out = []
    for c_str, _ in CLUSTER_NAMES:
        c = int(c_str)
        for mod in MODULES_ORDER:
            sub = [x for x in t125 if cl(x["cluster"]) == c and x["module"] == mod]
            n3 = sum(int(float(x["CRL"])) == 3 for x in sub)
            pc_out.append({
                "cluster": c_str, "module": mod, "count": len(sub), "crl3": n3,
                "crl3_percent": "" if not sub else (n3 / len(sub)) * 100.0,
            })
    pd.DataFrame(pc_out).to_csv(out_dir / "panel_c_roles.csv", index=False)

    joint = Counter((int(float(r["RRL"])), int(float(r["CRL"]))) for r in t125)
    pd_out = [{"RRL": rrl, "CRL": crl, "count": joint[(rrl, crl)]}
              for rrl in range(6) for crl in range(4)]
    pd.DataFrame(pd_out).to_csv(out_dir / "panel_d_joint.csv", index=False)

    pe_out = []
    for item in ITEMS_ORDER:
        passed = sum(yes(x[item]) for x in t125)
        pe_out.append({
            "item": item, "passed": passed, "denominator": 125, "unknown": 0,
            "scored_entries": 125, "unscored_primary_entries": 48,
            "percent": 100.0 * passed / 125, "definition": ITEM_DEF,
        })
    pd.DataFrame(pe_out).to_csv(out_dir / "panel_e_items.csv", index=False)

    pf_map = {
        ("resources", "Unique resources scanned"): (cs["unique_resources"], cs["unique_resources"]),
        ("resources", "With scale-labelled headers"): (cs["attributed_resource_count"], cs["unique_resources"]),
        ("resources", "Headers cover ≥2 scales"): (cs["scale_coverage_counts"]["2"] + cs["scale_coverage_counts"]["3"], cs["unique_resources"]),
        ("resources", "Same candidate at ≥2 scales"): (cs["resources_candidate_two_scales"], cs["unique_resources"]),
        ("resources", "Material → cycling-label match"): (cs["resources_material_to_cycled_cell"], cs["unique_resources"]),
        ("pairs", "Possible resource pairs"): (cs["possible_pairs"], cs["possible_pairs"]),
        ("pairs", "Share identifier candidates"): (cs["pairs_sharing_candidates"], cs["possible_pairs"]),
        ("pairs", "Candidate pairs across scales"): (cs["pairs_cross_scale"], cs["possible_pairs"]),
        ("pairs", "Material → cycling-label match"): (cs["pairs_material_to_cycled_cell"], cs["possible_pairs"]),
    }
    pf_out = []
    for unit, label, source_field in PANEL_F_ROWS:
        cnt, den = pf_map[(unit, label)]
        pf_out.append({
            "unit": unit, "label": label, "count": cnt, "denominator": den,
            "percent": 100.0 * cnt / den, "source_field": source_field, "definition": PANEL_F_DEF,
        })
    pd.DataFrame(pf_out).to_csv(out_dir / "panel_f_cross_scale.csv", index=False)

    panel_names = ("panel_a_clusters.csv", "panel_c_roles.csv", "panel_d_joint.csv", "panel_e_items.csv", "panel_f_cross_scale.csv")
    if out_dir.resolve() != DATA.resolve() and all((DATA / n).exists() for n in panel_names):
        bad = 0
        for name in panel_names:
            a = pd.read_csv(out_dir / name, dtype=str, keep_default_na=False)
            b = pd.read_csv(DATA / name, dtype=str, keep_default_na=False)
            same = a.equals(b)
            bad += not same
            print(f"{'identical' if same else 'DIFFERS  '}  {name}")
        return 1 if bad else 0
    print(f"Wrote {len(panel_names)} census panel tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
