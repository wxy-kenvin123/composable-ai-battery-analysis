"""Rebuild algorithm, AI-type, task, parallel-count and chord tables from doc_hits.csv.

Reads data/articles755.csv and data/doc_hits.csv (8 columns: id, Cluster, year,
code, AlgorithmName, taxonomy_source, ai_type, task_type), writes recomputed tables
into --outdir (default rerun/) and, when reference tables exist in data/ and outdir
differs from data/, compares every table cell by cell.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "source_original"))
import compute_top3_algorithms as core  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--outdir", type=Path, default=R / "rerun")
args = ap.parse_args()
OUT = args.outdir if args.outdir.is_absolute() else (R / args.outdir)
OUT.mkdir(parents=True, exist_ok=True)

articles = pd.read_csv(R / "data/articles755.csv", dtype={"id": str}).fillna("")
hits = pd.read_csv(R / "data/doc_hits.csv", dtype={"id": str}, keep_default_na=False)
for c in ("Cluster", "year"):
    hits[c] = hits[c].astype(int)

dh = hits[hits["taxonomy_source"] == "taxonomy"].copy()
added = hits[hits["taxonomy_source"] == "ai_type_phrase"].copy()
alg = dh[~dh["code"].str.lower().isin(core._AITYPE_ONLY_CODES)]

groups = ["≤2021", "2022", "2023", "2024", "2025", "2026"]
articles["group"] = articles.year.astype(int).map(lambda x: "≤2021" if x <= 2021 else str(x))
den = articles.groupby("group").size().reindex(groups, fill_value=0)

longs = {}
for field in ("ai_type", "task_type"):
    rows = []
    for d in dh.to_dict("records"):
        for cat in str(d[field]).split(";"):
            cat = cat.strip()
            if cat and cat.lower() != "nan":
                rows.append(dict(id=d["id"], Cluster=d["Cluster"], year=int(d["year"]), category=cat))
    long = pd.DataFrame(rows).drop_duplicates(["id", "category"])
    longs[field] = long

if len(added):
    add_rows = added[["id", "Cluster", "year", "ai_type"]].rename(columns={"ai_type": "category"})
    longs["ai_type"] = pd.concat([longs["ai_type"], add_rows], ignore_index=True).drop_duplicates(["id", "category"])

for field in ("ai_type", "task_type"):
    long = longs[field]
    mat = long.groupby(["category", "Cluster"]).id.nunique().unstack(fill_value=0).reindex(columns=range(1, 9), fill_value=0)
    mat = mat.loc[sorted(mat.index, key=lambda k: (-mat.loc[k].sum(), k))]
    mat.index.name = "category"
    yr = long.assign(group=long.year.map(lambda x: "≤2021" if x <= 2021 else str(x))).groupby(["category", "group"]).id.nunique().unstack(fill_value=0).reindex(columns=groups, fill_value=0)
    yr = yr.reindex(mat.index, fill_value=0)
    mat.to_csv(OUT / f"{field}_matrix.csv")
    yr.to_csv(OUT / f"{field}_year_counts.csv")
    yr.div(den, axis=1).mul(100).to_csv(OUT / f"{field}_year_rates.csv")
    long.to_csv(OUT / f"{field}_doc_presence.csv", index=False)

tasks_by_id: dict[str, set[str]] = {}
pairs: set[tuple[str, str, str]] = set()
for d in dh.to_dict("records"):
    tasks = tuple(t.strip() for t in str(d["task_type"]).split(";") if t.strip())
    tasks_by_id.setdefault(d["id"], set()).update(tasks)
    for a, t in itertools.product(str(d["ai_type"]).split(";"), tasks):
        if a.strip() and t:
            pairs.add((d["id"], a.strip(), t))
own = {(i, a) for i, a, _ in pairs}
for row in longs["ai_type"].to_dict("records"):
    if (row["id"], row["category"]) in own:
        continue
    for t in tasks_by_id.get(row["id"], ()):
        pairs.add((row["id"], row["category"], t))

pairs_df = pd.DataFrame(sorted(pairs), columns=["id", "ai_type", "task_type"])
pairs_df.to_csv(OUT / "chord_pairs_raw.csv", index=False)
topai = pairs_df.groupby("ai_type").id.nunique().sort_values(ascending=False).head(5).index
toptt = pairs_df.groupby("task_type").id.nunique().sort_values(ascending=False).head(5).index
fold = pairs_df.assign(
    ai_grp=pairs_df.ai_type.where(pairs_df.ai_type.isin(topai), "other AI types"),
    task_grp=pairs_df.task_type.where(pairs_df.task_type.isin(toptt), "other tasks"),
).drop_duplicates(["id", "ai_grp", "task_grp"])
fold.to_csv(OUT / "chord_pairs_folded.csv", index=False)
chord_exp = fold.groupby(["ai_grp", "task_grp"]).id.nunique().unstack(fill_value=0)
chord_exp.to_csv(OUT / "chord_matrix_expected.csv")
# Also emit R-ordered chord matrix (used by Fig. 4a / Source Data Fig. 4a)
ai_order = [
    "deep learning", "supervised machine learning", "ensemble learning",
    "post-hoc explanation", "evolutionary algorithms", "other AI types",
]
task_order = [
    "regression", "classification", "feature extraction",
    "optimization", "image classification", "other tasks",
]
chord_r = chord_exp.reindex(index=ai_order, columns=task_order, fill_value=0)
chord_r.index.name = ""
chord_r.to_csv(OUT / "chord_matrix_R.csv")
pd.DataFrame({"year_group": groups, "papers": den.values}).to_csv(OUT / "year_denominators.csv", index=False)

# parallel_counts (>=5 articles)
pc = alg.drop_duplicates(["id", "code"]).groupby(["code", "AlgorithmName", "Cluster"]).id.nunique().unstack(fill_value=0).reindex(columns=range(1, 9), fill_value=0)
pc.columns = [f"C{c}" for c in range(1, 9)]
pc["total"] = pc.sum(axis=1)
pc = pc[pc["total"] >= 5].reset_index().rename(columns={"code": "Algorithm"})
pc = pc[["Algorithm"] + [f"C{c}" for c in range(1, 9)] + ["AlgorithmName", "total"]]
pc = pc.sort_values(["total", "Algorithm"], ascending=[True, True]).reset_index(drop=True)
pc.to_csv(OUT / "parallel_counts.csv", index=False)

assert len(articles) == 755 and len(pc) == 63

names = ["chord_pairs_raw.csv", "chord_pairs_folded.csv", "chord_matrix_expected.csv", "year_denominators.csv"] + \
        [f"{f}_{k}.csv" for f in ("ai_type", "task_type") for k in ("matrix", "year_counts", "year_rates", "doc_presence")]
ref_dir = R / "data"
if OUT.resolve() != ref_dir.resolve() and all((ref_dir / n).exists() for n in names + ["parallel_counts.csv"]):
    bad = 0
    for n in names:
        a = pd.read_csv(OUT / n, dtype=str, keep_default_na=False)
        b = pd.read_csv(ref_dir / n, dtype=str, keep_default_na=False)
        same = a.equals(b)
        bad += not same
        print(f"{'identical' if same else 'DIFFERS  '}  {n}")
    # parallel_counts compared by (Algorithm, row values) so tie ordering doesn't mask numerical identity
    pa = pd.read_csv(OUT / "parallel_counts.csv", dtype=str, keep_default_na=False).set_index("Algorithm").sort_index()
    pb = pd.read_csv(ref_dir / "parallel_counts.csv", dtype=str, keep_default_na=False).set_index("Algorithm").sort_index()
    pc_same = pa.equals(pb)
    bad += not pc_same
    print(f"{'identical' if pc_same else 'DIFFERS  '}  parallel_counts.csv (63 algorithms)")
    print(f"{len(names) + 1 - bad} of {len(names) + 1} tables reproduced")
    sys.exit(1 if bad else 0)
else:
    print(f"Wrote {len(names) + 2} algorithm tables (63 algorithms >= 5 articles) to {OUT}")
    sys.exit(0)
