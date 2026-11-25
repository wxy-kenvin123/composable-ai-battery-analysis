"""Rerun Case 1 from this package and compare the outputs with the shipped run files.

Usage
-----
    python reproduce_case1.py screening    [--out DIR]   # screening replay, Note S9 (438 -> 319 -> 306 -> 56)
    python reproduce_case1.py continuation [--out DIR]   # ML-SSE executed on the 56 survivors
    python reproduce_case1.py chain        [--out DIR]   # end-to-end chain (Figure S5); see README for the literature step
    python reproduce_case1.py analysis     [--out DIR]   # value-added and journey tables, Figures S5 and S6, from results/
    python reproduce_case1.py compare      [--out DIR]   # compare DIR with results/
    python reproduce_case1.py all          [--out DIR]   # all of the above

Before the first three steps, fetch the upstream repositories (tools/fetch_external_data.py) and set
BATTAGENT_REPOS_CACHE. Outputs go to DIR (default ./rerun_case1) and never over results/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"

# shipped folder -> rerun subfolder
FOLDERS = {
    "screening": ("workflow_run_evidence", ["console_ml_htvs_screening.txt", "run_record_ml_htvs.json",
                                            "table_htvs_agent_trace.csv", "table_htvs_funnel.csv",
                                            "table_htvs_gate_decisions.csv", "table_htvs_shortlist.csv"]),
    "continuation": ("htvs_mlsse_continuation", ["run_record_htvs_mlsse_continuation.json",
                                                 "table_htvs_mlsse_ranked.csv", "table_htvs_mlsse_agent_trace.csv"]),
    "chain": ("case_a_full", ["run_record_case_a_full.json", "table_case_a_ranked.csv", "table_case_a_virtual_feedback.csv",
                              "table_case_a_agent_trace.csv", "table_case_a_agent_coverage.csv", "table_case_a_cards.csv",
                              "table_case_a_card_papers.csv", "table_case_a_suggestions.csv", "table_case_a_fd_rules.csv",
                              "table_case_a_retrieval_calibration.csv"]),
    "analysis": ("case_a_full", ["table_case_a_value_added.csv", "value_added_summary.json",
                                 "table_case_a_journeys.csv", "journeys_summary.json"]),
}
# Without the authors' literature engine the Knowledge Miner falls back to its deterministic stub, so only
# the stages before and beside the literature step can match the shipped end-to-end run.
CHAIN_WITHOUT_ENGINE = ["table_case_a_ranked.csv", "table_case_a_virtual_feedback.csv", "table_case_a_fd_rules.csv"]
CHAIN_KEYS_WITHOUT_ENGINE = ["n_shortlist", "n_cards", "n_top", "deep_dive", "deep_dive_rule", "aimd_matched",
                             "n_above_convention", "n_ood", "convention_log10_S_cm", "weight_sha256", "ranked"]
VOLATILE_KEYS = {"generated_utc", "environment", "resolved_root", "execution_how", "aimd_analysis_how"}


# ------------------------------------------------------------------------------------------------ runs
def run_screening(out: Path) -> None:
    from battery_workflow.examples.ml_htvs_screening import export_ml_htvs
    dest = out / "workflow_run_evidence"
    dest.mkdir(parents=True, exist_ok=True)
    export_ml_htvs(dest)
    print(f"screening replay written to {dest}")


def run_continuation(out: Path) -> None:
    from battery_workflow.examples.ml_htvs_continuation import export_continuation
    export_continuation(outdir=out / "htvs_mlsse_continuation")


def run_chain(out: Path) -> None:
    from battery_workflow.examples.ml_htvs_case_a_full import export_case_a_full
    from battery_workflow.knowledge_backend import get_knowledge_backend
    real = get_knowledge_backend(top_k=6).name != "stub"
    if not real:
        print("Literature engine not configured: the Knowledge Miner runs on its deterministic stub, so the rerun\n"
              "reproduces the executed ranking, the AIMD feedback and the rules table but not the literature cards.")
    export_case_a_full(outdir=out / "case_a_full", require_real_backend=False)


def run_analysis(out: Path) -> None:
    """Run the value-added, journey and figure scripts in a copy so the shipped files stay untouched."""
    work = out / "analysis_workspace"
    if work.exists():
        shutil.rmtree(work)
    for rel in ("battery_workflow", "scripts", "results/workflow_run_evidence", "results/htvs_mlsse_continuation",
                "results/case_a_full"):
        shutil.copytree(ROOT / rel, work / rel, ignore=shutil.ignore_patterns("__pycache__"))
    (work / "figures").mkdir(parents=True, exist_ok=True)
    for script in ("27_case_a_value_added.py", "29_case_a_journeys.py", "26_plot_case_a_full.py", "28_plot_case_a_value_added.py"):
        print(f"running scripts/{script}")
        subprocess.run([sys.executable, str(work / "scripts" / script)], cwd=work, check=True)
    dest = out / "case_a_full"
    dest.mkdir(parents=True, exist_ok=True)
    for name in FOLDERS["analysis"][1]:
        shutil.copy2(work / "results" / "case_a_full" / name, dest / name)
    for p in (work / "figures").glob("*"):
        shutil.copy2(p, out / p.name)
    print(f"analysis tables copied to {dest}; figures written to {out}")


# ------------------------------------------------------------------------------------------------ compare
_PATH = re.compile(r"(?:[A-Za-z]:[\\/]{1,2}|/)[^\s\"',;]*?(?=[\\/]{1,2}(?:gh_[\w.\-]+|oedb_[\w.\-]+))")


def norm(text: str) -> str:
    text = re.sub(r"<[A-Za-z_\-]+>", "<path>", text)          # placeholders in the shipped files
    text = _PATH.sub("<path>", text)                           # absolute roots of the reader's clones
    text = re.sub(r"(inprocess|subprocess):[^;\"\n]+", r"\1:<python>", text)
    return text.replace("\\\\", "/").replace("\\", "/")


def _num(a: str, b: str) -> bool:
    try:
        x, y = float(a), float(b)
    except ValueError:
        return False
    return abs(x - y) <= 1e-9 * max(1.0, abs(x), abs(y))


def _strip(obj, keys=None):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in VOLATILE_KEYS and (keys is None or k in keys)}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    if isinstance(obj, str):
        return norm(obj)
    return obj


def _flat(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flat(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flat(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def compare_file(shipped: Path, rerun: Path, keys=None) -> tuple[str, str]:
    if not rerun.exists():
        return "missing", "not produced by the rerun"
    a_raw, b_raw = shipped.read_bytes(), rerun.read_bytes()
    if a_raw == b_raw:
        return "identical", ""
    if shipped.suffix == ".json":
        a = _strip(json.loads(a_raw.decode("utf-8")), keys)
        b = _strip(json.loads(b_raw.decode("utf-8")), keys)
        if a == b:
            return "equal", "after removing volatile keys and local paths" + (" (deterministic keys only)" if keys else "")
        fa, fb = dict(_flat(a)), dict(_flat(b))
        diff = [k for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
        if all(k in fa and k in fb and _num(str(fa[k]), str(fb[k])) for k in diff):
            return "numerically equal", f"{len(diff)} values differ by <= 1e-9 relative"
        return "differs", f"{len(diff)} values differ, first: {diff[0]} = {fa.get(diff[0])!r} vs {fb.get(diff[0])!r}"
    a_lines = norm(a_raw.decode("utf-8")).splitlines()
    b_lines = norm(b_raw.decode("utf-8")).splitlines()
    if a_lines == b_lines:
        return "equal", "after local paths are masked"
    if shipped.suffix == ".csv" and len(a_lines) == len(b_lines):
        ra = list(csv.reader(a_lines))
        rb = list(csv.reader(b_lines))
        note = ""
        added = [c for c in rb[0] if c not in ra[0]] if ra and rb else []
        if added and [c for c in rb[0] if c not in added] == ra[0]:
            # a later framework version writes extra trace columns; compare the rest if they are empty
            keep = [j for j, c in enumerate(rb[0]) if c not in added]
            if all(not row[j] for row in rb[1:] for j, c in enumerate(rb[0]) if c in added and j < len(row)):
                rb = [[row[j] for j in keep] for row in rb]
                note = f"; the rerun adds empty column(s): {', '.join(added)}"
                if ra == rb:
                    return "equal", "after local paths are masked" + note
        cells = [(i, j) for i, (x, y) in enumerate(zip(ra, rb)) for j, (u, v) in enumerate(zip(x, y)) if u != v]
        if all(len(x) == len(y) for x, y in zip(ra, rb)) and all(_num(ra[i][j], rb[i][j]) for i, j in cells):
            return "numerically equal", f"{len(cells)} cells differ by <= 1e-9 relative" + note
        i, j = cells[0] if cells else (0, 0)
        return "differs", f"{len(cells)} cells differ, first: row {i}, column {ra[0][j] if ra else j}"
    first = next((i for i, (x, y) in enumerate(zip(a_lines, b_lines)) if x != y), min(len(a_lines), len(b_lines)))
    return "differs", f"{len(a_lines)} vs {len(b_lines)} lines, first difference at line {first + 1}"


def compare(out: Path) -> int:
    rows = []
    for step, (folder, names) in FOLDERS.items():
        rerun_dir = out / (folder if step != "screening" else "workflow_run_evidence")
        if not rerun_dir.exists():
            continue
        engine = True
        if step == "chain":
            rec = rerun_dir / "run_record_case_a_full.json"
            if not rec.exists():
                continue
            engine = json.loads(rec.read_text(encoding="utf-8")).get("knowledge_backend") != "stub"
        for name in names:
            if step == "chain" and not engine and name not in CHAIN_WITHOUT_ENGINE + ["run_record_case_a_full.json"]:
                rows.append((step, name, "not comparable", "needs the authors' literature engine; shipped file is the record"))
                continue
            keys = CHAIN_KEYS_WITHOUT_ENGINE if (step == "chain" and not engine and name.endswith(".json")) else None
            verdict, note = compare_file(RESULTS / folder / name, rerun_dir / name, keys)
            rows.append((step, name, verdict, note))
    if not rows:
        print(f"nothing to compare in {out}; run a step first")
        return 1
    width = max(len(r[1]) for r in rows)
    for step, name, verdict, note in rows:
        print(f"{step:<13} {name:<{width}}  {verdict:<18} {note}")
    report = out / "comparison.csv"
    with open(report, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "file", "verdict", "note"])
        w.writerows(rows)
    bad = [r for r in rows if r[2] in ("differs", "missing")]
    print(f"\n{len(rows) - len(bad)} of {len(rows)} files match or are not comparable; details in {report}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["screening", "continuation", "chain", "analysis", "compare", "all"])
    ap.add_argument("--out", type=Path, default=ROOT / "rerun_case1")
    a = ap.parse_args()
    out = a.out.resolve()
    if out == RESULTS.resolve() or RESULTS.resolve() in out.parents:
        sys.exit("--out must not point into results/")
    out.mkdir(parents=True, exist_ok=True)
    steps = ["screening", "continuation", "chain", "analysis"] if a.step == "all" else [a.step]
    for step in steps:
        if step == "compare":
            continue
        {"screening": run_screening, "continuation": run_continuation, "chain": run_chain, "analysis": run_analysis}[step](out)
    return compare(out) if a.step in ("compare", "all") else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
