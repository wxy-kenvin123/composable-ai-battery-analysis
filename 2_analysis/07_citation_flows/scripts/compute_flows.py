"""Compute provider->receiver knowledge-flow tables from citation_edges_used.csv.

Adapted from Hemmelder et al. (2026), Nature Energy 11, 313-323,
https://doi.org/10.1038/s41560-026-01985-z (MIT licence).
The three matrix-aggregation and diagonal-benchmark helper functions below are
reproduced from the original MIT-licensed implementation with full attribution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parents[1]
LABELS = [f"C{i}" for i in range(1, 9)]
GROUPS = ["≤2021", "2022", "2023", "2024", "2025", "2026"]
COLORS = ["#8DB2D7", "#F9DA8E", "#79BED5", "#F8B62D", "#274C9F", "#E9504F", "#B7DFD9", "#FBDED7"]
OUTPUTS = [
    "flow_counts.csv", "flow_relative_to_within.csv", "flow_share_all_references.csv",
    "flow_share_corpus_references.csv", "cluster_denominators.csv", "receiver_group_denominators.csv",
    "flows_by_receiver_year.csv", "fig2_cross_cluster_flows.csv", "fig2_cross_cluster_flows_all.csv",
    "cluster_definitions.csv", "flow_summary.json",
]


# ---------------------------------------------------------------------------
# MIT-licensed helpers adapted from Andre Hemmelder (2026),
# battery_knowledge_flows_repo (DOI: 10.1038/s41560-026-01985-z):
#   - D_visualization_package/visualization/heatmaps/matrix.py
#   - C_network_analysis_package/network_analysis/diagnostics.py
# ---------------------------------------------------------------------------
def build_raw_matrix(
    edges: pd.DataFrame,
    labels: list[str],
    source_col: str = "source_label",
    target_col: str = "target_label",
    weight_col: str = "weight",
) -> np.ndarray:
    """Convert an edge table to a dense provider x receiver matrix aligned to labels."""
    idx = {lab: i for i, lab in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=float)
    for _, r in edges.iterrows():
        s = str(r[source_col])
        t = str(r[target_col])
        if s not in idx or t not in idx:
            continue
        mat[idx[s], idx[t]] += float(r.get(weight_col, 0.0))
    return mat


def diagonal_benchmark(matrix: np.ndarray) -> np.ndarray:
    """Extract the diagonal (within-group self-referencing baseline)."""
    return np.diag(matrix).astype(float)


def benchmark_matrix(matrix: np.ndarray, diag: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Normalize each receiver column relative to its diagonal benchmark (percent)."""
    denom = np.maximum(diag.astype(float), epsilon)
    return (matrix / denom.reshape(1, -1)) * 100.0


def year_group(year: int) -> str:
    return "≤2021" if int(year) <= 2021 else str(int(year))


def divide_percent(numerator, denominator):
    a, b = np.broadcast_arrays(np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float))
    result = np.full(a.shape, np.nan)
    np.divide(100 * a, b, out=result, where=np.isfinite(b) & (b > 0))
    return result


def write_matrix(outdir: Path, values, name: str):
    frame = pd.DataFrame(values, index=LABELS, columns=LABELS)
    frame.index.name = "provider_cluster"
    frame.to_csv(outdir / name, encoding="utf-8-sig")
    return frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=RUN / "data")
    ap.add_argument("--outdir", type=Path, default=RUN / "rerun" / "data")
    ap.add_argument("--k8-names", type=Path, default=RUN.parent / "03_clustering" / "output" / "data" / "K8_names.csv")
    ap.add_argument("--compare-dir", type=Path, default=None, help="Optional reference dir for cell-by-cell comparison")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_csv(a.data_dir / "corpus.csv", dtype={"id": str})
    coverage = pd.read_csv(a.data_dir / "reference_coverage_used.csv", dtype={"id": str})
    used = pd.read_csv(a.data_dir / "citation_edges_used.csv", dtype={"source_id": str, "target_id": str}).fillna("")
    assert len(corpus) == 755 and corpus.id.nunique() == 755
    assert set(coverage.id) == set(corpus.id) and coverage.id.is_unique
    assert (coverage.denominator_is_analysis_choice.astype(int) == 1).all()
    assert set(corpus.Cluster.astype(int)) == set(range(1, 9))
    assert not used.duplicated(["source_id", "target_id"]).any()
    assert not (used.source_id == used.target_id).any()
    assert set(used.source_id) | set(used.target_id) <= set(corpus.id)
    assert (used.weight.astype(float) == 1).all()

    corpus["year_group"] = corpus.year.astype(int).map(year_group)
    meta = corpus.set_index("id")
    for side in ("source", "target"):
        expected = used[f"{side}_id"].map(meta.Cluster).astype(int)
        assert np.array_equal(expected, used[f"{side}_cluster"].astype(int))
        used[f"{side}_label"] = "C" + used[f"{side}_cluster"].astype(int).astype(str)
    used["receiver_year_group"] = used.target_id.map(meta.year_group)

    c = coverage.set_index("id").reindex(corpus.id).copy()
    c["Cluster"] = corpus.Cluster.to_numpy()
    c["year_group"] = corpus.year_group.to_numpy()
    c["denominator_all_references"] = pd.to_numeric(c.denominator_all_references, errors="coerce")
    c["denominator_known"] = c.denominator_all_references.notna() & (c.denominator_all_references > 0)

    group_meta = c.groupby(["Cluster", "year_group"]).agg(
        papers=("Cluster", "size"),
        papers_with_reference_denominator=("denominator_known", "sum"),
        all_reference_entries=("denominator_all_references", lambda x: x.sum(min_count=1)),
    ).reindex(pd.MultiIndex.from_product([range(1, 9), GROUPS], names=["Cluster", "year_group"]))
    for field in ("papers", "papers_with_reference_denominator"):
        group_meta[field] = group_meta[field].fillna(0).astype(int)
    group_meta["denominator_complete"] = group_meta.papers == group_meta.papers_with_reference_denominator
    group_meta.loc[~group_meta.denominator_complete, "all_reference_entries"] = np.nan
    group_meta.loc[group_meta.papers == 0, "all_reference_entries"] = np.nan

    raw = build_raw_matrix(used, LABELS)
    assert raw.sum() == len(used)
    within = np.diag(raw)
    original_ratio = benchmark_matrix(raw, diagonal_benchmark(raw))
    original_ratio[:, within == 0] = np.nan
    independently_calculated = divide_percent(raw, within[None, :])
    assert np.allclose(original_ratio, independently_calculated, equal_nan=True)

    d = []
    denom_rows = []
    for cluster in range(1, 9):
        sub = c[c.Cluster == cluster]
        complete = bool(sub.denominator_known.all())
        total = float(sub.denominator_all_references.sum()) if complete else np.nan
        d.append(total)
        j = cluster - 1
        denom_rows.append(dict(
            Cluster=cluster, papers=len(sub),
            papers_with_reference_denominator=int(sub.denominator_known.sum()),
            denominator_complete=complete, all_reference_entries=total,
            matched_corpus_references=int(raw[:, j].sum()),
            within_cluster_references=int(within[j]),
            within_share_all_references=divide_percent(within[j], total).item(),
            within_share_corpus_references=divide_percent(within[j], raw[:, j].sum()).item(),
        ))
    all_share = divide_percent(raw, np.array(d)[None, :])
    corpus_share = divide_percent(raw, raw.sum(axis=0)[None, :])

    write_matrix(a.outdir, raw, "flow_counts.csv")
    write_matrix(a.outdir, original_ratio, "flow_relative_to_within.csv")
    write_matrix(a.outdir, all_share, "flow_share_all_references.csv")
    write_matrix(a.outdir, corpus_share, "flow_share_corpus_references.csv")

    denominator_table = pd.DataFrame(denom_rows)
    denominator_table.to_csv(a.outdir / "cluster_denominators.csv", index=False, encoding="utf-8-sig")
    group_meta.to_csv(a.outdir / "receiver_group_denominators.csv", encoding="utf-8-sig")

    long_rows = []
    sum_by_year = np.zeros((8, 8))
    for group in GROUPS:
        group_edges = used[used.receiver_year_group == group]
        group_raw = build_raw_matrix(group_edges, LABELS)
        sum_by_year += group_raw
        for j in range(8):
            gm = group_meta.loc[(j + 1, group)]
            all_denom = gm.all_reference_entries
            internal_denom = group_raw[:, j].sum()
            for i in range(8):
                long_rows.append(dict(
                    year_group=group, provider_cluster=i + 1, receiver_cluster=j + 1,
                    citation_count=int(group_raw[i, j]), receiver_papers=int(gm.papers),
                    denominator_all_references=all_denom, denominator_corpus_references=int(internal_denom),
                    percentage_all_references=divide_percent(group_raw[i, j], all_denom).item(),
                    percentage_corpus_references=divide_percent(group_raw[i, j], internal_denom).item(),
                    cross_cluster=i != j,
                ))
    assert np.array_equal(sum_by_year, raw)
    flows = pd.DataFrame(long_rows)
    flows.to_csv(a.outdir / "flows_by_receiver_year.csv", index=False, encoding="utf-8-sig")
    flows[flows.cross_cluster].to_csv(a.outdir / "fig2_cross_cluster_flows.csv", index=False, encoding="utf-8-sig")

    overall_rows = []
    for j in range(8):
        for i in range(8):
            if i == j:
                continue
            overall_rows.append(dict(
                year_group="All years", provider_cluster=i + 1, receiver_cluster=j + 1,
                citation_count=int(raw[i, j]), receiver_papers=int(denominator_table.iloc[j].papers),
                denominator_all_references=d[j], denominator_corpus_references=int(raw[:, j].sum()),
                percentage_all_references=all_share[i, j], percentage_corpus_references=corpus_share[i, j],
                cross_cluster=True,
            ))
    pd.DataFrame(overall_rows).to_csv(a.outdir / "fig2_cross_cluster_flows_all.csv", index=False, encoding="utf-8-sig")

    names = pd.read_csv(a.k8_names)[["Cluster", "n", "name_en", "figure_short_name"]].sort_values("Cluster")
    assert names.Cluster.tolist() == list(range(1, 9))
    assert names.n.tolist() == corpus.groupby("Cluster").size().tolist()
    names["color"] = COLORS
    names.to_csv(a.outdir / "cluster_definitions.csv", index=False, encoding="utf-8-sig")

    summary = dict(
        corpus_papers=len(corpus), citation_edges=len(used),
        cross_cluster_edges=int((used.source_cluster != used.target_cluster).sum()),
        within_cluster_edges=int((used.source_cluster == used.target_cluster).sum()),
        all_reference_denominator_complete=bool(c.denominator_known.all()),
        total_all_reference_entries=float(c.denominator_all_references.sum()),
        maximum_relative_flow=float(np.nanmax(original_ratio)),
        receiver_year_groups=GROUPS, within_cluster_counts=within.astype(int).tolist(),
        year_group_papers=corpus.groupby("year_group").size().to_dict(),
    )
    (a.outdir / "flow_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"citation pairs {summary['citation_edges']}, cross-theme {summary['cross_cluster_edges']}, "
          f"within-theme {summary['within_cluster_edges']}, reference items {summary['total_all_reference_entries']:.0f}")
    if a.compare_dir is not None:
        bad = 0
        for name in OUTPUTS:
            p_new, p_ref = a.outdir / name, a.compare_dir / name
            same = p_new.read_bytes() == p_ref.read_bytes()
            how = "identical"
            if not same and name.endswith(".csv"):
                x = pd.read_csv(p_new, dtype=str, keep_default_na=False)
                y = pd.read_csv(p_ref, dtype=str, keep_default_na=False)
                same, how = x.equals(y), "equal (same cells)"
            elif not same:
                same = json.loads(p_new.read_text(encoding="utf-8")) == json.loads(p_ref.read_text(encoding="utf-8"))
                how = "equal (same JSON)"
            bad += not same
            print(f"{how if same else 'DIFFERS':20s} {name}")
        print(f"{len(OUTPUTS) - bad} of {len(OUTPUTS)} tables reproduced")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
