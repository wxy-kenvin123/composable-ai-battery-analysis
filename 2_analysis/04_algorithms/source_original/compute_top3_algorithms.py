#!/usr/bin/env python3
"""
Compute per-cluster Top-3 algorithms (k=8) by scanning full-text .txt files.

Counting rule:
- "Mentioned docs" count: a doc contributes at most +1 to an algorithm if it mentions it.

Inputs (defaults inferred from this script location):
- ./txt_algo/*.txt
- ../article_clusters_analysis.xlsx (must contain columns: article, Cluster)

Output:
- ./cluster_top3_algorithms_k8.xlsx
  - sheet: summary
  - sheet: all_counts

Optional (recommended):
- Reuse the AI-SDG review taxonomy to standardize algorithm names and map them to
  `ai_type` and `task_type` via these Excel tables:
  - algorithms_list.xlsx (algorithms, algorithm_name)
  - algorithms_aitype.xlsx (algorithms, ai_type)
  - algorithms_tasktype.xlsx (algorithms, task_type)
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern, Sequence, Tuple


def _die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _soft_imports():
    try:
        import pandas as pd  # type: ignore
    except Exception:
        _die(
            "Missing dependency: pandas\n"
            "Install with: pip install pandas openpyxl"
        )
    try:
        import openpyxl  # noqa: F401  # type: ignore
    except Exception:
        _die(
            "Missing dependency: openpyxl\n"
            "Install with: pip install pandas openpyxl"
        )
    return pd


def _soft_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        return None, None, exc
    return plt, np, None


def _normalize_text(raw: str) -> str:
    # Light cleanup: keep letters/digits/+ to preserve terms like 1DCNN, kmeans++.
    raw = raw.lower()
    # Treat hyphen as a separator to improve matching of phrases like "short-term" vs "short term".
    raw = raw.replace("-", " ")
    raw = re.sub(r"[^a-z0-9\+ ]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def _read_text(path: Path) -> str:
    return _normalize_text(path.read_text(encoding="utf-8", errors="ignore"))


@dataclass(frozen=True)
class AlgoPattern:
    name: str
    patterns: Tuple[Pattern[str], ...]


def _compile_patterns(raw: Dict[str, List[str]]) -> List[AlgoPattern]:
    compiled: List[AlgoPattern] = []
    for name, pats in raw.items():
        compiled.append(
            AlgoPattern(
                name=name,
                patterns=tuple(re.compile(p, flags=re.IGNORECASE) for p in pats),
            )
        )
    return compiled


def _detect_algorithms(text: str, algos: Iterable[AlgoPattern]) -> List[str]:
    hits: List[str] = []
    for algo in algos:
        if any(p.search(text) for p in algo.patterns):
            hits.append(algo.name)
    return hits


@dataclass(frozen=True)
class TaxonomyAlgo:
    code: str
    name: str
    aliases: Tuple[str, ...]
    ai_types: Tuple[str, ...]
    task_types: Tuple[str, ...]


_GENERIC_STOP_ALIASES = frozenset({"overall", "various", "unspecified"})

# Short aliases prone to false-positive collisions
_SHORT_ALIAS_BLOCKLIST = frozenset({"gan"})

# High-ambiguity algorithm codes to exclude entirely (from audit_rules.json)
# nca = neighbourhood components analysis (confused with NCA cathode material)
# rdf = radial distribution function (chemistry term, not AI)
# rf  = random forest abbreviation (too short, matches author initials)
# rmsd = root mean square deviation (chemistry metric, not AI)
_EXCLUDE_ALGO_CODES = frozenset({"nca", "rdf", "rf", "rmsd"})
# Included in ai_type counts but excluded from algorithm rankings
_AITYPE_ONLY_CODES = frozenset({"active_learning_local"})


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def _alias_variants(raw: str) -> List[str]:
    """
    Generate a small set of matching-friendly aliases for a string.

    We normalize both the doc text and aliases with the same rule, then match via
    space-bounded substring search (phrase mode) or token checks (token mode).
    Variants help when normalization removes punctuation/digits and when sources
    differ in spacing (e.g., 'k-means'/'k means'/'kmeans', 'U-Net'/'unet').
    """
    base = _normalize_text(raw)
    if not base:
        return []

    variants = [base]

    # Some sources write multi-token names without spaces (e.g., "kmeans", "unet").
    if " " in base:
        variants.append(base.replace(" ", ""))

    # Optional '+' tolerance (e.g., "kmeans++" vs "kmeans") while keeping "c++" safe
    # due to the minimum-length filter below.
    if "+" in base:
        variants.append(base.replace("+", ""))
        variants.append(base.replace(" ", "").replace("+", ""))

    # Digit tolerance (can increase false positives for some families; see README).
    if any(ch.isdigit() for ch in base):
        no_digits = _normalize_text(re.sub(r"\d+", "", raw))
        if no_digits:
            variants.append(no_digits)
            if " " in no_digits:
                variants.append(no_digits.replace(" ", ""))
            if "+" in no_digits:
                variants.append(no_digits.replace("+", ""))
                variants.append(no_digits.replace(" ", "").replace("+", ""))

    # Plural tolerance: add trailing 's' variant for multi-token aliases
    # (e.g., "generative adversarial network" -> "generative adversarial networks")
    if " " in base and not base.endswith("s"):
        variants.append(base + "s")

    variants = [v for v in _dedupe_preserve_order(variants) if v and v not in _GENERIC_STOP_ALIASES and v not in _SHORT_ALIAS_BLOCKLIST]
    # Avoid ultra-short aliases (too noisy)
    variants = [v for v in variants if len(v) >= 3]
    return variants


def _load_ai_sdg_taxonomy(pd, taxonomy_dir: Path) -> List[TaxonomyAlgo]:
    algorithms_list_path = taxonomy_dir / "algorithms_list.xlsx"
    algorithms_aitype_path = taxonomy_dir / "algorithms_aitype.xlsx"
    algorithms_tasktype_path = taxonomy_dir / "algorithms_tasktype.xlsx"

    missing = [p for p in [algorithms_list_path, algorithms_aitype_path, algorithms_tasktype_path] if not p.exists()]
    if missing:
        _die(
            "AI-SDG taxonomy files not found (required):\n"
            + "\n".join(f"- {p}" for p in missing)
        )

    alg_list = pd.read_excel(algorithms_list_path)
    if not {"algorithms", "algorithm_name"}.issubset(set(alg_list.columns)):
        _die("algorithms_list.xlsx must contain columns: algorithms, algorithm_name")

    alg_list = alg_list[["algorithms", "algorithm_name"]].copy()
    alg_list["algorithms"] = alg_list["algorithms"].astype(str).str.strip()
    alg_list["algorithm_name"] = alg_list["algorithm_name"].astype(str).str.strip()
    alg_list = alg_list[(alg_list["algorithms"] != "") & (alg_list["algorithms"].str.lower() != "nan")]

    ai_map_df = pd.read_excel(algorithms_aitype_path)
    if not {"algorithms", "ai_type"}.issubset(set(ai_map_df.columns)):
        _die("algorithms_aitype.xlsx must contain columns: algorithms, ai_type")
    ai_map_df = ai_map_df[["algorithms", "ai_type"]].copy()
    ai_map_df["algorithms"] = ai_map_df["algorithms"].astype(str).str.strip()
    ai_map_df["ai_type"] = ai_map_df["ai_type"].astype(str).str.strip()
    ai_map_df = ai_map_df[(ai_map_df["algorithms"] != "") & (ai_map_df["ai_type"] != "")]

    task_map_df = pd.read_excel(algorithms_tasktype_path)
    if not {"algorithms", "task_type"}.issubset(set(task_map_df.columns)):
        _die("algorithms_tasktype.xlsx must contain columns: algorithms, task_type")
    task_map_df = task_map_df[["algorithms", "task_type"]].copy()
    task_map_df["algorithms"] = task_map_df["algorithms"].astype(str).str.strip()
    task_map_df["task_type"] = task_map_df["task_type"].astype(str).str.strip()
    task_map_df = task_map_df[(task_map_df["algorithms"] != "") & (task_map_df["task_type"] != "")]

    ai_types_by_algo: Dict[str, Tuple[str, ...]] = (
        ai_map_df.groupby("algorithms")["ai_type"]
        .apply(lambda s: tuple(sorted(set(x for x in s.tolist() if x))))
        .to_dict()
    )
    task_types_by_algo: Dict[str, Tuple[str, ...]] = (
        task_map_df.groupby("algorithms")["task_type"]
        .apply(lambda s: tuple(sorted(set(x for x in s.tolist() if x))))
        .to_dict()
    )

    tax_algos: List[TaxonomyAlgo] = []
    for _, row in alg_list.iterrows():
        code = str(row["algorithms"]).strip()
        name = str(row["algorithm_name"]).strip()
        if not code or code.lower() in _GENERIC_STOP_ALIASES or code.lower() in _EXCLUDE_ALGO_CODES:
            continue

        aliases_raw = []
        aliases_raw.extend(_alias_variants(code))
        if name and name.lower() != "nan" and name.lower() not in _GENERIC_STOP_ALIASES:
            aliases_raw.extend(_alias_variants(name))
        aliases = tuple(_dedupe_preserve_order(aliases_raw))
        if not aliases:
            continue

        tax_algos.append(
            TaxonomyAlgo(
                code=code,
                name=name if name and name.lower() != "nan" else code,
                aliases=aliases,
                ai_types=ai_types_by_algo.get(code, tuple()),
                task_types=task_types_by_algo.get(code, tuple()),
            )
        )

    if not tax_algos:
        _die(f"No usable algorithms loaded from: {algorithms_list_path}")
    return tax_algos


def _load_optional_order_list(pd, xlsx_path: Path, column: str) -> List[str]:
    if not xlsx_path.exists():
        return []
    try:
        df = pd.read_excel(xlsx_path)
    except Exception:
        return []
    if column not in df.columns:
        return []
    vals = []
    for v in df[column].dropna().tolist():
        s = str(v).strip()
        if not s or s.lower() == "nan":
            continue
        vals.append(s)
    return _dedupe_preserve_order(vals)


def _token_present(token: str, tokens: set[str]) -> bool:
    """
    Loose token presence check to improve recall with messy/cleaned OCR text.

    Examples:
    - 'network' should match doc token 'networks'
    - 'networks' should match doc token 'network'
    """
    if token in tokens:
        return True
    if len(token) < 4:
        return False
    if token.endswith("s") and token[:-1] in tokens:
        return True
    if (token + "s") in tokens:
        return True
    return False


def _detect_algorithms_taxonomy(
    text: str,
    tokens: set[str],
    tax_algos: Iterable[TaxonomyAlgo],
    *,
    multi_token_mode: str = "tokens",
) -> List[str]:
    """
    Detect algorithm codes using taxonomy-provided aliases.

    Speed optimization:
    - Single-token alias: token membership check (with loose plural tolerance)
    - Multi-token alias: require all tokens present (with loose plural tolerance),
      then apply multi_token_mode.

    multi_token_mode:
    - 'phrase': multi-word alias must appear as a contiguous phrase (higher precision)
    - 'tokens': all alias tokens appear anywhere in the doc (higher recall; may overcount)
    """
    padded = f" {text} "
    hits: List[str] = []
    for algo in tax_algos:
        for alias in algo.aliases:
            alias_tokens = alias.split()
            if len(alias_tokens) == 1:
                if _token_present(alias_tokens[0], tokens):
                    hits.append(algo.code)
                    break
                continue

            if any(not _token_present(t, tokens) for t in alias_tokens):
                continue
            if multi_token_mode == "phrase":
                if f" {alias} " in padded:
                    hits.append(algo.code)
                    break
            elif multi_token_mode == "tokens":
                # Guardrail: for 2-word aliases, require contiguous phrase to avoid
                # massive over-counting from generic token co-occurrence
                # (e.g., 'regression' + 'algorithms' appearing far apart).
                if len(alias_tokens) == 2:
                    if f" {alias} " in padded:
                        hits.append(algo.code)
                        break
                else:
                    hits.append(algo.code)
                    break
            else:
                raise ValueError(f"Unknown multi_token_mode: {multi_token_mode!r}")
    return hits


def _top_n_by_total(counts_by_cluster: Dict[int, Dict[str, int]], n: int, preferred_order: Optional[List[str]] = None) -> List[str]:
    totals: Dict[str, int] = {}
    for cluster_counts in counts_by_cluster.values():
        for key, val in cluster_counts.items():
            totals[key] = totals.get(key, 0) + int(val)

    order_index: Dict[str, int] = {}
    if preferred_order:
        order_index = {name: i for i, name in enumerate(preferred_order)}

    def sort_key(item):
        key, total = item
        return (-total, order_index.get(key, 10_000), key)

    ranked = sorted(totals.items(), key=sort_key)
    ranked = [k for k, total in ranked if total > 0]
    return ranked[:n]


def _all_nonzero_columns(counts_by_cluster: Dict[int, Dict[str, int]], preferred_order: Optional[List[str]] = None) -> List[str]:
    found = {k for cluster_counts in counts_by_cluster.values() for k, v in cluster_counts.items() if int(v) > 0}
    if not found:
        return []
    if preferred_order:
        ordered = [k for k in preferred_order if k in found]
        ordered += sorted([k for k in found if k not in set(ordered)])
        return ordered
    return sorted(found)


def _build_matrix_df(
    pd,
    cluster_ids: List[int],
    counts_by_cluster: Dict[int, Dict[str, int]],
    columns: List[str],
    column_name: str,
) -> "pd.DataFrame":
    rows = []
    for cid in cluster_ids:
        row = {"Cluster": cid}
        for col in columns:
            row[col] = int(counts_by_cluster.get(cid, {}).get(col, 0))
        rows.append(row)
    df = pd.DataFrame(rows)
    # keep a readable column label in exported tables
    df.attrs["column_name"] = column_name  # type: ignore[attr-defined]
    return df


def _plot_heatmap(
    plt,
    np,
    matrix_df,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str = "Cluster",
    annotate: bool = True,
    cmap: str = "YlOrRd",
) -> None:
    if matrix_df.empty:
        return

    cols = [c for c in matrix_df.columns if c != "Cluster"]
    data = matrix_df[cols].to_numpy(dtype=float)
    y_labels = matrix_df["Cluster"].astype(str).tolist()

    # dynamic sizing for readability
    fig_w = max(8.0, 0.45 * len(cols) + 2.5)
    fig_h = max(3.5, 0.45 * len(y_labels) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of articles")

    if annotate and data.size <= 400:  # avoid over-annotating huge matrices
        vmax = float(np.nanmax(data)) if data.size else 0.0
        threshold = vmax * 0.6 if vmax else 0.0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = int(data[i, j])
                color = "white" if val >= threshold else "black"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _build_top3_summary(
    pd,
    clusters_df,
    cluster_ids: List[int],
    counts: Dict[int, Dict[str, int]],
    processed_sizes: Optional[Dict[int, int]] = None,
    label_map: Optional[Dict[str, str]] = None,
) -> "pd.DataFrame":
    rows = []
    for cid in cluster_ids:
        ordered = sorted(counts[cid].items(), key=lambda kv: kv[1], reverse=True)
        ordered = [(k, v) for k, v in ordered if v > 0]
        top3 = ordered[:3]
        while len(top3) < 3:
            top3.append((None, None))

        def _label(k: Optional[str]) -> Optional[str]:
            if k is None:
                return None
            if label_map and k in label_map:
                return label_map[k]
            return k

        row = {
            "Cluster": cid,
            "Size": int((clusters_df["Cluster"] == cid).sum()),
        }
        if processed_sizes is not None:
            row["Docs_Processed"] = int(processed_sizes.get(cid, 0))

        row.update(
            {
                "Algo1": _label(top3[0][0]),
                "Count1": top3[0][1],
                "Algo2": _label(top3[1][0]),
                "Count2": top3[1][1],
                "Algo3": _label(top3[2][0]),
                "Count3": top3[2][1],
            }
        )
        rows.append(
            row
        )
    return pd.DataFrame(rows)


def main() -> None:
    pd = _soft_imports()

    # CLI + infer paths from repo layout
    this_dir = Path(__file__).resolve().parent
    project_dir = this_dir.parent

    default_taxonomy_dir = project_dir / "data" / "taxonomy"

    parser = argparse.ArgumentParser(
        description="Compute per-cluster Top-3 algorithms by scanning full-text txt files."
    )
    parser.add_argument(
        "--detector",
        choices=["auto", "taxonomy", "regex"],
        default="auto",
        help="auto: use AI-SDG taxonomy if available else regex; taxonomy: force taxonomy; regex: force regex.",
    )
    parser.add_argument("--taxonomy-dir", type=str, default=str(default_taxonomy_dir))
    parser.add_argument("--clusters", type=str, default=str(project_dir.parent / "03_clustering" / "output" / "data" / "clustering" / "article_clusters_analysis.csv"))
    parser.add_argument("--text-dir", type=str, default=str(this_dir / "txt_algo"))
    parser.add_argument("--out", type=str, default=str(this_dir / "cluster_top3_algorithms_k8.xlsx"))
    parser.add_argument(
        "--taxonomy-multi-token-mode",
        choices=["tokens", "phrase"],
        default="phrase",
        help=(
            "When using taxonomy detector, how to match multi-word algorithm aliases. "
            "tokens: all words appear anywhere (higher recall; may overcount). "
            "phrase: require contiguous phrase (higher precision)."
        ),
    )
    parser.add_argument(
        "--plot-heatmaps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate heatmap PNGs (requires matplotlib).",
    )
    parser.add_argument("--fig-dir", type=str, default=str(this_dir / "figures"))
    parser.add_argument("--ai-type-top-n", type=int, default=20)
    parser.add_argument(
        "--role-top-n",
        type=int,
        default=0,
        help="Top-N 'AI roles' (uses task_type as proxy). 0 means use all found roles.",
    )
    args = parser.parse_args()

    taxonomy_dir = Path(args.taxonomy_dir).expanduser()
    clusters_path = Path(args.clusters).expanduser()
    text_dir = Path(args.text_dir).expanduser()
    out_path = Path(args.out).expanduser()
    fig_dir = Path(args.fig_dir).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not clusters_path.exists():
        _die(f"Missing clusters file: {clusters_path}")
    if not text_dir.exists():
        _die(
            f"Missing txt dir: {text_dir}\n"
            "If you haven't generated algorithm-friendly txt files yet, run:\n"
            "  your full-text extraction script first\n"
            "Or pass an existing directory via --text-dir."
        )

    detector_used = "regex"

    # Read cluster mapping
    clusters_df = pd.read_excel(clusters_path)
    if "article" not in clusters_df.columns or "Cluster" not in clusters_df.columns:
        _die("article_clusters_analysis.xlsx must contain columns: article, Cluster")

    clusters_df = clusters_df[["article", "Cluster"]].copy()
    clusters_df["article"] = clusters_df["article"].astype(str).str.strip()
    clusters_df = clusters_df[(clusters_df["article"] != "") & (clusters_df["article"].str.lower() != "nan")]
    clusters_df["article"] = clusters_df["article"].map(lambda s: Path(s).name)

    clusters_df = clusters_df[clusters_df["Cluster"].notna()]
    try:
        clusters_df["Cluster"] = clusters_df["Cluster"].astype(int)
    except Exception:
        _die(f"Column 'Cluster' must be numeric/integer in: {clusters_path}")

    if clusters_df.empty:
        _die(f"No usable rows found in cluster mapping: {clusters_path}")

    # Guardrail: the same article should not map to multiple clusters.
    conflict = clusters_df.groupby("article")["Cluster"].nunique()
    conflict = conflict[conflict > 1]
    if not conflict.empty:
        examples = conflict.index.tolist()[:10]
        _die(
            "Cluster mapping contains duplicate articles assigned to multiple clusters (examples):\n"
            + "\n".join(f"- {a}" for a in examples)
        )

    clusters_df = clusters_df.drop_duplicates(subset=["article"], keep="first").reset_index(drop=True)
    cluster_by_article: Dict[str, int] = dict(zip(clusters_df["article"].tolist(), clusters_df["Cluster"].tolist()))

    cluster_ids = sorted(clusters_df["Cluster"].unique().tolist())

    # Decide detector + load resources
    use_taxonomy = False
    tax_algos: List[TaxonomyAlgo] = []
    taxonomy_by_code: Dict[str, TaxonomyAlgo] = {}
    ai_type_order: List[str] = []
    task_type_order: List[str] = []

    if args.detector in ("taxonomy", "auto"):
        required = [
            taxonomy_dir / "algorithms_list.xlsx",
            taxonomy_dir / "algorithms_aitype.xlsx",
            taxonomy_dir / "algorithms_tasktype.xlsx",
        ]
        if args.detector == "taxonomy" or all(p.exists() for p in required):
            tax_algos = _load_ai_sdg_taxonomy(pd, taxonomy_dir)
            taxonomy_by_code = {a.code: a for a in tax_algos}
            use_taxonomy = True
            detector_used = "taxonomy"
            ai_type_order = _load_optional_order_list(pd, taxonomy_dir / "ai_type_list.xlsx", "ai_type")
            task_type_order = _load_optional_order_list(pd, taxonomy_dir / "task_type_list.xlsx", "task_type")
        elif args.detector == "taxonomy":
            _die(
                "Requested --detector taxonomy, but required taxonomy files were not found under:\n"
                f"- {taxonomy_dir}\n"
                "Make sure this directory contains algorithms_list.xlsx / algorithms_aitype.xlsx / algorithms_tasktype.xlsx."
            )

    # Fallback regex dictionary: name -> regex patterns
    # (kept small on purpose; taxonomy mode is recommended for wider coverage)
    algo_patterns_raw: Dict[str, List[str]] = {
        "LinearRegression": [r"\blinear\s+regression\b"],
        "LogisticRegression": [r"\blogistic\s+regression\b"],
        "DecisionTree": [r"\bdecision\s+tree\b"],
        "RandomForest": [r"\brandom\s+forest\b", r"\brf\b"],
        "XGBoost": [r"\bxgboost\b", r"\bextreme\s+gradient\s+boost(ing)?\b"],
        "LightGBM": [r"\blightgbm\b", r"\blight\s+gbm\b"],
        "CatBoost": [r"\bcatboost\b"],
        "GradientBoosting": [r"\bgradient\s+boost(ing)?\b", r"\bgbm\b", r"\bgbdt\b"],
        "SVM": [r"\bsupport\s+vector(\s+(machine|regression|classifier))?\b", r"\bsvm\b", r"\bsvr\b"],
        "KNN": [r"\bk\s*-?nearest\s+neighbou?rs?\b", r"\bknn\b"],
        "KMeans": [r"\bk\s*-?means\b", r"\bkmeans\b", r"\bkmeans\+\+\b"],
        "PCA": [r"\bprincipal\s+component\b", r"\bpca\b"],
        "NeuralNetwork": [r"\bartificial\s+neural\s+network\b", r"\bneural\s+network\b", r"\bann\b", r"\bmlp\b"],
        "CNN": [r"\bconvolutional\s+neural\s+network\b", r"\bcnn\b", r"\bu\s*net\b", r"\bunet\b"],
        "RNN_LSTM_GRU": [r"\brecurrent\s+neural\s+network\b", r"\brnn\b", r"\blstm\b", r"\bgru\b"],
        "Transformer_LLM": [r"\btransformer\b", r"\bllm\b", r"\blarge\s+language\s+model\b", r"\bgpt\b", r"\bbert\b"],
        "GraphNeuralNetwork": [r"\bgraph\s+neural\s+network\b", r"\bgnn\b", r"\bgcn\b", r"\bgat\b", r"\bmpnn\b"],
        "ActiveLearning": [r"\bactive\s+learning\b"],
        "ReinforcementLearning": [r"\breinforcement\s+learning\b", r"\brl\b", r"\bq\s*learning\b"],
        "BayesianOptimization": [r"\bbayesian\s+optimization\b"],
        "GeneticAlgorithm": [r"\bgenetic\s+algorithm\b", r"\bga\b", r"\bnsga\b"],
        "ParticleSwarmOptimization": [r"\bparticle\s+swarm\s+optimization\b", r"\bpso\b"],
        "SHAP": [r"\bshap\b", r"\bshapley\b"],
        "mRMR": [r"\bmrmr\b"],
        "AutoEncoder_VAE_GAN": [r"\bauto\s*encoder\b", r"\bvae\b", r"\bvariational\s+autoencoder\b", r"\bgan\b", r"\bcgan\b"],
        "GaussianProcess": [r"\bgaussian\s+process\b", r"\bgpr\b"],
        "ExtremeLearningMachine": [r"\bextreme\s+learning\s+machine(s)?\b", r"\belm\b"],
    }
    regex_algos = _compile_patterns(algo_patterns_raw)
    regex_algo_names = [a.name for a in regex_algos]

    # Prepare counts: cluster -> algo -> mentioned_docs
    algo_keys: List[str]
    if use_taxonomy:
        algo_keys = [a.code for a in tax_algos]
    else:
        algo_keys = regex_algo_names

    counts: Dict[int, Dict[str, int]] = {cid: {a: 0 for a in algo_keys} for cid in cluster_ids}
    ai_type_counts: Dict[int, Dict[str, int]] = {cid: {} for cid in cluster_ids}
    task_type_counts: Dict[int, Dict[str, int]] = {cid: {} for cid in cluster_ids}

    txt_files = sorted(text_dir.glob("*.txt"))
    if not txt_files:
        _die(f"No txt files found in: {text_dir}")

    mapped_articles = set(cluster_by_article.keys())
    txt_names = {p.name for p in txt_files}

    missing_txt = sorted(mapped_articles - txt_names)
    skipped_txt = sorted(txt_names - mapped_articles)
    if missing_txt:
        print(
            f"Warning: {len(missing_txt)} mapped article(s) have no txt file under: {text_dir}\n"
            "Counts will be computed from the available txt files only.",
            file=sys.stderr,
        )

    processed_sizes: Dict[int, int] = {cid: 0 for cid in cluster_ids}
    doc_hits_rows: list = []  # collect per-doc per-algo hits for doc_hits sheet
    for path in txt_files:
        article = path.name
        cluster_id = cluster_by_article.get(article)
        if cluster_id is None:
            continue
        processed_sizes[cluster_id] += 1
        text = _read_text(path)
        tokens = set(text.split())
        # Make '+' effectively optional for token checks (e.g., "kmeans" vs "kmeans++")
        if any("+" in t for t in tokens):
            tokens |= {t.replace("+", "") for t in tokens if "+" in t and len(t.replace("+", "")) >= 3}

        if use_taxonomy:
            hit_algos = _detect_algorithms_taxonomy(
                text,
                tokens,
                tax_algos,
                multi_token_mode=str(args.taxonomy_multi_token_mode),
            )
            for code in hit_algos:
                counts[cluster_id][code] += 1
                meta = taxonomy_by_code.get(code)
                doc_hits_rows.append({
                    "article": article,
                    "Cluster": cluster_id,
                    "Algorithm": code,
                    "AlgorithmName": meta.name if meta else code,
                    "source": "taxonomy",
                    "ai_type": "; ".join(meta.ai_types) if meta else "",
                    "task_type": "; ".join(meta.task_types) if meta else "",
                })

            # Also count AI types / task types (doc-level presence)
            doc_ai_types = set()
            doc_task_types = set()
            for code in hit_algos:
                meta = taxonomy_by_code.get(code)
                if not meta:
                    continue
                doc_ai_types.update(meta.ai_types)
                doc_task_types.update(meta.task_types)
            for t in doc_ai_types:
                ai_type_counts[cluster_id][t] = ai_type_counts[cluster_id].get(t, 0) + 1
            for t in doc_task_types:
                task_type_counts[cluster_id][t] = task_type_counts[cluster_id].get(t, 0) + 1
        else:
            hit_algos = _detect_algorithms(text, regex_algos)
            for algo in hit_algos:
                counts[cluster_id][algo] += 1

    if skipped_txt:
        print(f"Warning: {len(skipped_txt)} txt file(s) not found in cluster mapping (skipped).", file=sys.stderr)

    # Build algorithm summary (Top3 per cluster)
    label_map: Optional[Dict[str, str]] = None
    if use_taxonomy:
        label_map = {a.code: a.name for a in tax_algos}
    summary_df = _build_top3_summary(pd, clusters_df, cluster_ids, counts, processed_sizes=processed_sizes, label_map=label_map)

    # Build all_counts table
    all_rows = []
    for cid in cluster_ids:
        for algo in algo_keys:
            if algo.lower() in _AITYPE_ONLY_CODES:
                continue  # Included in ai_type counts only
            if use_taxonomy:
                meta = taxonomy_by_code.get(algo)
                all_rows.append(
                    {
                        "Cluster": cid,
                        "Algorithm": algo,
                        "AlgorithmName": meta.name if meta else algo,
                        "Mentioned_Docs": counts[cid][algo],
                        "ai_type": "; ".join(meta.ai_types) if meta and meta.ai_types else None,
                        "task_type": "; ".join(meta.task_types) if meta and meta.task_types else None,
                    }
                )
            else:
                all_rows.append({"Cluster": cid, "Algorithm": algo, "Mentioned_Docs": counts[cid][algo]})
    all_counts_df = pd.DataFrame(all_rows)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        all_counts_df.to_excel(writer, sheet_name="all_counts", index=False)

        # doc_hits: per-document per-algorithm hit records
        if doc_hits_rows:
            doc_hits_df = pd.DataFrame(doc_hits_rows)
            doc_hits_df.to_excel(writer, sheet_name="doc_hits", index=False)

        if skipped_txt:
            pd.DataFrame({"skipped_txt_files": skipped_txt}).to_excel(writer, sheet_name="skipped_txt", index=False)
        if missing_txt:
            pd.DataFrame({"missing_txt_files": missing_txt}).to_excel(writer, sheet_name="missing_txt", index=False)

        # Optional taxonomy-derived rollups
        if use_taxonomy:
            # AI type summary + counts
            ai_types_found = sorted({t for cid in cluster_ids for t in ai_type_counts[cid].keys()})
            if ai_type_order:
                ai_types = ai_type_order + [t for t in ai_types_found if t not in set(ai_type_order)]
            else:
                ai_types = ai_types_found
            ai_counts_rows = []
            for cid in cluster_ids:
                for t in ai_types:
                    ai_counts_rows.append({"Cluster": cid, "ai_type": t, "Mentioned_Docs": ai_type_counts[cid].get(t, 0)})
            ai_counts_df = pd.DataFrame(ai_counts_rows)

            ai_summary_counts = {cid: dict(ai_type_counts[cid]) for cid in cluster_ids}
            ai_summary_df = _build_top3_summary(pd, clusters_df, cluster_ids, ai_summary_counts, processed_sizes=processed_sizes, label_map=None)
            ai_summary_df = ai_summary_df.rename(columns={"Algo1": "Type1", "Algo2": "Type2", "Algo3": "Type3"})
            ai_summary_df.to_excel(writer, sheet_name="ai_type_summary", index=False)
            ai_counts_df.to_excel(writer, sheet_name="ai_type_counts", index=False)

            # Task type summary + counts
            task_types_found = sorted({t for cid in cluster_ids for t in task_type_counts[cid].keys()})
            if task_type_order:
                task_types = task_type_order + [t for t in task_types_found if t not in set(task_type_order)]
            else:
                task_types = task_types_found
            task_counts_rows = []
            for cid in cluster_ids:
                for t in task_types:
                    task_counts_rows.append({"Cluster": cid, "task_type": t, "Mentioned_Docs": task_type_counts[cid].get(t, 0)})
            task_counts_df = pd.DataFrame(task_counts_rows)

            task_summary_counts = {cid: dict(task_type_counts[cid]) for cid in cluster_ids}
            task_summary_df = _build_top3_summary(pd, clusters_df, cluster_ids, task_summary_counts, processed_sizes=processed_sizes, label_map=None)
            task_summary_df = task_summary_df.rename(columns={"Algo1": "Type1", "Algo2": "Type2", "Algo3": "Type3"})
            task_summary_df.to_excel(writer, sheet_name="task_type_summary", index=False)
            task_counts_df.to_excel(writer, sheet_name="task_type_counts", index=False)

            # Heatmap matrices (cluster x top-N)
            ai_type_top = _top_n_by_total(ai_type_counts, n=int(args.ai_type_top_n), preferred_order=ai_type_order)
            if int(args.role_top_n) > 0:
                task_type_top = _top_n_by_total(task_type_counts, n=int(args.role_top_n), preferred_order=task_type_order)
            else:
                task_type_top = _all_nonzero_columns(task_type_counts, preferred_order=task_type_order)

            ai_type_matrix = _build_matrix_df(pd, cluster_ids, ai_type_counts, ai_type_top, column_name="ai_type")
            task_type_matrix = _build_matrix_df(pd, cluster_ids, task_type_counts, task_type_top, column_name="task_type")

            ai_type_matrix.to_excel(writer, sheet_name="ai_type_top_matrix", index=False)
            task_type_matrix.to_excel(writer, sheet_name="ai_role_matrix", index=False)

            settings = pd.DataFrame(
                [
                    {"key": "detector", "value": "taxonomy"},
                    {"key": "taxonomy_dir", "value": str(taxonomy_dir)},
                    {"key": "clusters_path", "value": str(clusters_path)},
                    {"key": "text_dir", "value": str(text_dir)},
                    {"key": "taxonomy_multi_token_mode", "value": str(args.taxonomy_multi_token_mode)},
                    {"key": "ai_type_top_n", "value": str(args.ai_type_top_n)},
                    {"key": "ai_role_top_n", "value": str(args.role_top_n)},
                    {"key": "plot_heatmaps", "value": str(bool(args.plot_heatmaps))},
                    {"key": "fig_dir", "value": str(fig_dir)},
                ]
            )
            settings.to_excel(writer, sheet_name="settings", index=False)
        else:
            settings = pd.DataFrame(
                [
                    {"key": "detector", "value": "regex"},
                    {"key": "clusters_path", "value": str(clusters_path)},
                    {"key": "text_dir", "value": str(text_dir)},
                ]
            )
            settings.to_excel(writer, sheet_name="settings", index=False)

    # Heatmap plots (PNG)
    if args.plot_heatmaps and not use_taxonomy:
        print(
            "Info: heatmaps are skipped because taxonomy mode is not enabled.\n"
            "Run with --detector taxonomy (and optionally --taxonomy-dir ...) to generate ai_type/task_type heatmaps.",
            file=sys.stderr,
        )

    if use_taxonomy and args.plot_heatmaps:
        plt, np, err = _soft_import_matplotlib()
        if err or plt is None or np is None:
            print(
                "Warning: matplotlib/numpy not available; skip heatmap plotting.\n"
                "Install with: pip install matplotlib numpy",
                file=sys.stderr,
            )
        else:
            fig_dir.mkdir(parents=True, exist_ok=True)
            # Re-build top matrices for plotting (keep consistent with Excel)
            ai_type_top = _top_n_by_total(ai_type_counts, n=int(args.ai_type_top_n), preferred_order=ai_type_order)
            if int(args.role_top_n) > 0:
                task_type_top = _top_n_by_total(task_type_counts, n=int(args.role_top_n), preferred_order=task_type_order)
            else:
                task_type_top = _all_nonzero_columns(task_type_counts, preferred_order=task_type_order)
            ai_type_matrix = _build_matrix_df(pd, cluster_ids, ai_type_counts, ai_type_top, column_name="ai_type")
            task_type_matrix = _build_matrix_df(pd, cluster_ids, task_type_counts, task_type_top, column_name="task_type")

            _plot_heatmap(
                plt,
                np,
                task_type_matrix,
                fig_dir / "heatmap_ai_roles_task_type.png",
                title=(
                    "AI roles (task_type) by cluster "
                    f"({'all' if int(args.role_top_n) == 0 else f'top {int(args.role_top_n)}; found {len(task_type_top)}'})"
                ),
                xlabel="AI role (task_type)",
            )
            _plot_heatmap(
                plt,
                np,
                ai_type_matrix,
                fig_dir / f"heatmap_ai_types_top{int(args.ai_type_top_n)}.png",
                title=f"AI types by cluster (requested top {int(args.ai_type_top_n)}; found {len(ai_type_top)})",
                xlabel="AI type",
            )
            print(
                f"Saved heatmaps to: {fig_dir}\n"
                "- heatmap_ai_roles_task_type.png\n"
                f"- heatmap_ai_types_top{int(args.ai_type_top_n)}.png",
                file=sys.stderr,
            )

    print(f"Detector used: {detector_used}", file=sys.stderr)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
