"""Reproduce all derived tables of 2_analysis/ in dependency order and verify outputs.

Usage
-----
    python reproduce_all.py                       # rerun all 10 analysis sections + verify
    python reproduce_all.py --rscript /path/Rscript
    python reproduce_all.py --check-only          # verify already-generated outputs

Case 1 and Case 2 workflows in ``1_workflow_framework/`` can be run directly with
the commands listed in ``1_workflow_framework/README.md``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_step(label: str, cmd: list[str], cwd: Path) -> None:
    print(f"\n=== [{label}] {' '.join(cmd)} (cwd={cwd.relative_to(ROOT).as_posix()}) ===")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def reproduce_analysis(rscript: str) -> None:
    """Run all 10 executable sections of 2_analysis/ in DAG order into their standard paths."""
    py = sys.executable
    a2 = ROOT / "2_analysis"

    steps: list[tuple[str, list[str], Path]] = [
        ("01_corpus", [rscript, "annual_records/nightingale_rose_2026_split.R", "--svg-only"], a2 / "01_corpus"),
        ("04_algorithms", [py, "-B", "scripts/aggregate_from_hits.py", "--outdir", "data"], a2 / "04_algorithms"),
        ("03_clustering/cluster", [rscript, "code/cluster.R", "output"], a2 / "03_clustering"),
        ("03_clustering/adapt", [rscript, "code/adapt_model.R", "output"], a2 / "03_clustering"),
        ("03_clustering/k_choice", [py, "-I", "k_choice/verify_k8_ruler.py"], a2 / "03_clustering"),
        ("05_data_sources", [py, "-B", "scripts/recount_data_sources.py", "--write-data"], a2 / "05_data_sources"),
        ("06_scale_tests/position", [rscript, "scripts/position.R", "output"], a2 / "06_scale_tests"),
        (
            "06_scale_tests/split_tests",
            [py, "-B", "scripts/split_tests.py", "output/data", "output/data_final_coding"],
            a2 / "06_scale_tests",
        ),
        ("06_scale_tests/dca_axes", [rscript, "scripts/dca_axes.R", "output"], a2 / "06_scale_tests"),
        ("06_scale_tests/control_test", [rscript, "scripts/control_test.R", "output/control_test"], a2 / "06_scale_tests"),
        ("07_citation_flows", [py, "-B", "scripts/compute_flows.py", "--outdir", "data"], a2 / "07_citation_flows"),
        ("08_census", [py, "-B", "scripts/recount_census.py", "--outdir", "data"], a2 / "08_census"),
        ("09_maps", [rscript, "scripts/plot_datahub_map.R", "output"], a2 / "09_maps"),
        ("10_case2_sharing/readouts", [py, "-I", "analysis/ablation_readouts.py"], a2 / "10_case2_sharing"),
        ("11_tea", [py, "-B", "scripts/recompute_tea_ladder.py", "output"], a2 / "11_tea"),
    ]
    for label, cmd, cwd in steps:
        run_step(label, cmd, cwd)


def verify_outputs() -> None:
    """Confirm that key regenerated outputs and Source Data targets exist and pass sanity checks."""
    required = [
        "1_workflow_framework/results/workflow_run_evidence/table_htvs_funnel.csv",
        "1_workflow_framework/results/case_a_full/table_case_a_agent_trace.csv",
        "1_workflow_framework/results/case_a_full/table_case_a_journeys.csv",
        "1_workflow_framework/results/case_a_full/table_case_a_value_added.csv",
        "2_analysis/01_corpus/annual_records/annual_records_by_stream.csv",
        "2_analysis/03_clustering/output/data/cluster_model.rds",
        "2_analysis/03_clustering/output/data/assignments_k8.csv",
        "2_analysis/03_clustering/output/data/dca_sites.csv",
        "2_analysis/03_clustering/output/data/wordcloud_selected_terms.csv",
        "2_analysis/04_algorithms/data/ai_type_matrix.csv",
        "2_analysis/04_algorithms/data/task_type_matrix.csv",
        "2_analysis/04_algorithms/data/chord_matrix_R.csv",
        "2_analysis/04_algorithms/data/parallel_counts.csv",
        "2_analysis/04_algorithms/data/parallel_number_key.csv",
        "2_analysis/05_data_sources/data/final_stats.json",
        "2_analysis/06_scale_tests/output/data/cluster_positions.csv",
        "2_analysis/06_scale_tests/output/data/boundary_gaps.csv",
        "2_analysis/06_scale_tests/output/data/dca_axes.csv",
        "2_analysis/06_scale_tests/output/control_test/data/summary.csv",
        "2_analysis/06_scale_tests/output/data_final_coding/split_tests.csv",
        "2_analysis/07_citation_flows/data/flow_summary.json",
        "2_analysis/07_citation_flows/data/flow_share_all_references.csv",
        "2_analysis/08_census/data/panel_a_clusters.csv",
        "2_analysis/08_census/data/panel_f_cross_scale.csv",
        "2_analysis/09_maps/data/country_counts.csv",
        "2_analysis/09_maps/data/country_cluster_counts.csv",
        "2_analysis/09_maps/output/data/flows.csv",
        "2_analysis/09_maps/output/data/hub_positions.csv",
        "2_analysis/10_case2_sharing/oedb_v2026-05-11/oedb_mcp_exportData_v2026-05-11.csv",
        "2_analysis/10_case2_sharing/oedb_v2026-05-11/oedb-electrolytes-v2026-05-11.csv",
        "2_analysis/10_case2_sharing/results/sharing_conditions/seed_summary.csv",
        "2_analysis/11_tea/output/tea_ladder_results.json",
    ]
    missing = [rel for rel in required if not (ROOT / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing expected output files: {missing}")

    flow_summary = json.loads((ROOT / "2_analysis/07_citation_flows/data/flow_summary.json").read_text(encoding="utf-8"))
    assert flow_summary["corpus_papers"] == 755
    assert flow_summary["all_reference_denominator_complete"] is True

    ds_stats = json.loads((ROOT / "2_analysis/05_data_sources/data/final_stats.json").read_text(encoding="utf-8"))
    assert (ds_stats["narrow_articles"], ds_stats["broad_articles"], ds_stats["evidence_rows"]) == (257, 270, 1676)

    tea = json.loads((ROOT / "2_analysis/11_tea/output/tea_ladder_results.json").read_text(encoding="utf-8"))
    assert "base" in tea and "scenarios" in tea and len(tea["scenarios"]) > 0
    print(f"\nPASS: all {len(required)} key outputs verified.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-only", action="store_true", help="Only verify already-generated outputs")
    ap.add_argument(
        "--rscript",
        default=os.environ.get("RSCRIPT") or shutil.which("Rscript") or "Rscript",
        help="Path to the Rscript executable (default: RSCRIPT env var or Rscript on PATH)",
    )
    args = ap.parse_args()

    if args.check_only:
        verify_outputs()
        return 0
    reproduce_analysis(args.rscript)
    verify_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
