"""Break-point tests on theme profiles (Supplementary Note 1, "Continuity from atoms to cells").

Package version of parts 3 and 4 of scripts/original/links.py, with the same functions and test definitions. It
reads only files of this package:
  - word-space neighbour links between themes (pair_mixing.csv, written by scripts/position.R);
  - core algorithms shared between themes (04_algorithms/fig2h/data/gC_atoms_to_cells_arcs.csv);
  - detected algorithms (04_algorithms/data/doc_hits.csv, checked against parallel_counts.csv);
  - the final data-source coding (05_data_sources/data/paper_summary.csv and dataset_by_cluster.csv).
Part 2 of links.py (the vocabulary counts of bridge articles) needs the
full texts and is not repeated; its output is output/data/bridge_counts.csv.

Usage: python scripts/split_tests.py [MIXING_DIR]     (default output/data; rerun/data after position.R)
Writes rerun/data_final_coding/*.csv (compare with output/data_final_coding) and prints the numbers of the
manuscript; exits 1 if one differs.
"""
from __future__ import annotations

import csv
import itertools
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

SEC = Path(__file__).resolve().parents[1]
PKG = SEC.parent
MIX_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else SEC / "output" / "data"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else SEC / "rerun" / "data_final_coding"
ARCS = PKG / "04_algorithms" / "fig2h" / "data" / "gC_atoms_to_cells_arcs.csv"
PARALLEL = PKG / "04_algorithms" / "data" / "parallel_counts.csv"
DOC_HITS = PKG / "04_algorithms" / "data" / "doc_hits.csv"
ASSIGN = PKG / "03_clustering" / "output" / "data" / "assignments_k8.csv"
SOURCES = PKG / "05_data_sources" / "data"

CLUSTERS = list(range(1, 9))
LEFT, RIGHT = {1, 8, 2, 3}, {4, 5, 6, 7}
PAIRS = list(itertools.combinations(CLUSTERS, 2))
K_NN = 10


def rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(name: str, data: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)


def cosine(u: list[float], v: list[float]) -> float:
    nu, nv = math.sqrt(sum(a * a for a in u)), math.sqrt(sum(b * b for b in v))
    return sum(a * b for a, b in zip(u, v)) / (nu * nv) if nu and nv else 0.0


def split_tests(sim: dict[tuple[int, int], float], label: str) -> dict:
    """C1-C6 rank, break point A (rank of C6) and B (exact test over the 35 four-four splits)."""
    ordered = sorted(PAIRS, key=lambda p: sim[p])
    c16_rank = ordered.index((1, 6)) + 1
    mean_to_others = {c: sum(sim[tuple(sorted((c, o)))] for o in CLUSTERS if o != c) / 7 for c in CLUSTERS}
    c6_rank = sorted(CLUSTERS, key=lambda c: mean_to_others[c]).index(6) + 1

    def gap(group: set[int]) -> float:
        same = [sim[p] for p in PAIRS if (p[0] in group) == (p[1] in group)]
        cross = [sim[p] for p in PAIRS if (p[0] in group) != (p[1] in group)]
        return sum(same) / len(same) - sum(cross) / len(cross)

    splits = {frozenset(g) for g in itertools.combinations(CLUSTERS, 4) if 1 in g}     # 35 distinct splits
    observed = gap(LEFT)
    p_b = sum(gap(set(g)) >= observed - 1e-12 for g in splits) / len(splits)
    same = [sim[p] for p in PAIRS if (p[0] in LEFT) == (p[1] in LEFT)]
    cross = [sim[p] for p in PAIRS if (p[0] in LEFT) != (p[1] in LEFT)]
    return {"measure": label, "C1_C6_value": round(sim[(1, 6)], 4), "C1_C6_rank_of_28_lowest_first": c16_rank,
            "C1_C6_in_lowest_quarter": c16_rank <= 7,
            "A_C6_mean_to_others": round(mean_to_others[6], 4), "A_C6_rank_of_8_lowest_first": c6_rank,
            "A_exact_p": round(c6_rank / 8, 3),
            "B_mean_same_side": round(sum(same) / len(same), 4), "B_mean_across": round(sum(cross) / len(cross), 4),
            "B_exact_p_of_35_splits": round(p_b, 3)}


def main() -> int:
    cluster_of = {r["id"]: int(r["Cluster"]) for r in rows(ASSIGN)}
    assert len(cluster_of) == 755
    size = Counter(cluster_of.values())
    tests = []

    # word-space neighbours (position.R)
    mix_rows = {(int(r["a"]), int(r["b"])): r for r in rows(MIX_DIR / "pair_mixing.csv")}
    mix = {p: float(r["knn_share"]) for p, r in mix_rows.items()}
    tests.append(split_tests(mix, "word-space neighbours shared (position.R)"))
    links = {p: int(r["knn_links"]) for p, r in mix_rows.items()}
    # links expected if each article's ten neighbours were drawn at random from the other 754 articles
    expected = {p: K_NN * 2 * size[p[0]] * size[p[1]] / (len(cluster_of) - 1) for p in PAIRS}
    c6 = {o: (links[tuple(sorted((6, o)))], links[tuple(sorted((6, o)))] / expected[tuple(sorted((6, o)))])
          for o in CLUSTERS if o != 6}

    # method sharing
    core_any, core_job, arc_style = Counter(), Counter(), defaultdict(list)
    for r in rows(ARCS):
        pr = tuple(sorted(int(c[1:]) for c in r["pair"].split("-")))
        core_any[pr] += int(r["shared_core_algorithms"])
        core_job[pr] += 0 if r["style"] == "name" else int(r["shared_core_algorithms"])
        arc_style[pr].append(f'{r["style"]} {r["shared_core_algorithms"]}')
    assert sum(core_any.values()) == 189
    assert core_any[(1, 6)] == 2 and core_job[(1, 6)] == 0
    core_any = {p: core_any[p] for p in PAIRS}
    core_job = {p: core_job[p] for p in PAIRS}
    hit_rows = [r for r in rows(DOC_HITS) if r.get("taxonomy_source", "taxonomy") == "taxonomy" and r["AlgorithmName"] not in ("", "active learning")]
    use = {(r["id"], r["AlgorithmName"]) for r in hit_rows}
    papers_with_hits = defaultdict(set)
    for i, _ in use:
        papers_with_hits[cluster_of[i]].add(i)
    algos = sorted({a for _, a in use})
    counts = {c: Counter(a for i, a in use if cluster_of[i] == c) for c in CLUSTERS}
    for r in rows(PARALLEL):
        for c in CLUSTERS:
            assert counts[c][r["AlgorithmName"]] == int(r[f"C{c}"]), (r["AlgorithmName"], c)
    profile = {c: [counts[c][a] / len(papers_with_hits[c]) for a in algos] for c in CLUSTERS}
    method_cos = {p: cosine(profile[p[0]], profile[p[1]]) for p in PAIRS}
    tests.append(split_tests(core_any, "core algorithms shared, any use (panel g)"))
    tests.append(split_tests(core_job, "core algorithms shared for the same use (panel g)"))
    tests.append(split_tests(method_cos, "algorithm-use profile, cosine"))
    write("method_similarity.csv", [
        {"pair": f"C{a}-C{b}", "core_shared_any_use": core_any[(a, b)], "core_shared_same_use": core_job[(a, b)],
         "arc_parts": "; ".join(arc_style[(a, b)]), "algorithm_profile_cosine": round(method_cos[(a, b)], 4),
         "crosses_A": (a == 6) != (b == 6), "crosses_B": (a in LEFT) != (b in LEFT)} for a, b in PAIRS])

    # data sources (final coding)
    kinds = ["public_db", "public_model", "own_computation", "own_experiment", "literature", "industry_field"]
    summ = rows(SOURCES / "paper_summary.csv")
    assert len(summ) == 755 and all(cluster_of[r["paper_id"]] == int(r["cluster"]) for r in summ)
    src = {c: [sum(r[k] == "True" for r in summ if int(r["cluster"]) == c) / sum(int(r["cluster"]) == c for r in summ)
               for k in kinds] for c in CLUSTERS}
    source_cos = {p: cosine(src[p[0]], src[p[1]]) for p in PAIRS}
    tests.append(split_tests(source_cos, "data-source profile, cosine"))
    write("source_similarity.csv", [
        {"pair": f"C{a}-C{b}", "source_profile_cosine": round(source_cos[(a, b)], 4),
         "crosses_A": (a == 6) != (b == 6), "crosses_B": (a in LEFT) != (b in LEFT)} for a, b in PAIRS])
    write("source_profiles.csv", [{"cluster": f"C{c}", **{k: round(v, 3) for k, v in zip(kinds, src[c])}} for c in CLUSTERS])
    named = defaultdict(set)
    for r in rows(SOURCES / "dataset_by_cluster.csv"):
        named[int(r["cluster"])].add(r["dataset"])
    shared = {p: named[p[0]] & named[p[1]] for p in PAIRS}
    jac = {p: len(shared[p]) / len(named[p[0]] | named[p[1]]) for p in PAIRS}
    tests.append(split_tests(jac, "named datasets shared, Jaccard"))
    write("dataset_overlap.csv", [
        {"pair": f"C{a}-C{b}", "named_a": len(named[a]), "named_b": len(named[b]), "shared": len(shared[(a, b)]),
         "jaccard": round(jac[(a, b)], 4), "shared_datasets": "; ".join(sorted(shared[(a, b)])),
         "crosses_A": (a == 6) != (b == 6), "crosses_B": (a in LEFT) != (b in LEFT)} for a, b in PAIRS])
    write("split_tests.csv", tests)

    def mean_to_others(sim):
        return {c: sum(sim[tuple(sorted((c, o)))] for o in CLUSTERS if o != c) / 7 for c in CLUSTERS}

    measures = {"word_space_neighbours": mix, "core_shared_same_use": core_job, "algorithm_profile_cosine": method_cos,
                "source_profile_cosine": source_cos, "named_dataset_jaccard": jac}
    conn = {k: mean_to_others(v) for k, v in measures.items()}
    write("cluster_connectedness.csv", [{"cluster": f"C{c}", **{k: round(conn[k][c], 4) for k in measures}} for c in CLUSTERS])

    # ------------------------------------------------------------------ numbers of the manuscript
    t = {r["measure"]: r for r in tests}
    src_t, alg_t = t["data-source profile, cosine"], t["algorithm-use profile, cosine"]
    others = [conn["core_shared_same_use"][c] for c in CLUSTERS if c != 1]
    other5 = [c6[o] for o in (2, 3, 4, 5, 7)]
    got = {
        "source profile, same side / across / P": (round(src_t["B_mean_same_side"], 2), round(src_t["B_mean_across"], 2),
                                                     round(src_t["B_exact_p_of_35_splits"] * 35)),
        "algorithm profile, same side / across / P": (round(alg_t["B_mean_same_side"], 2), round(alg_t["B_mean_across"], 2),
                                                        round(alg_t["B_exact_p_of_35_splits"], 2)),
        "same-use core algorithms, P": round(t["core algorithms shared for the same use (panel g)"]["B_exact_p_of_35_splits"], 2),
        "named datasets, P": round(t["named datasets shared, Jaccard"]["B_exact_p_of_35_splits"], 2),
        "C6 named datasets / shared with other themes": (len(named[6]), len(named[6] & set().union(*(named[c] for c in CLUSTERS if c != 6)))),
        "C6 same-use core algorithms per theme pair": round(conn["core_shared_same_use"][6], 1),
        "C1 same-use core algorithms per theme pair; others": (round(conn["core_shared_same_use"][1], 2),
                                                               round(min(others), 1), round(max(others), 1)),
        "neighbour links C6-C1, C6-C8": (c6[1][0], c6[8][0]),
        "neighbour links C6 and the other five themes": sorted(n for n, _ in other5),
        "links / expected, C1 and C8": (round(c6[1][1], 3), round(c6[8][1], 2)),
        "links / expected, other five themes": (round(min(r for _, r in other5), 2), round(max(r for _, r in other5), 2)),
        "data sources of C1, C8 own computation; C3 external databases (%)": (round(100 * src[1][2]), round(100 * src[8][2]),
                                                                            round(100 * src[3][0])),
        "C4-C7 own experiments (%)": (round(100 * min(src[c][3] for c in (4, 5, 6, 7))), round(100 * max(src[c][3] for c in (4, 5, 6, 7)))),
    }
    want = {
        "source profile, same side / across / P": (0.88, 0.62, 1),
        "algorithm profile, same side / across / P": (0.72, 0.70, 0.26),
        "same-use core algorithms, P": 0.26,
        "named datasets, P": 0.66,
        "C6 named datasets / shared with other themes": (10, 0),
        "C6 same-use core algorithms per theme pair": 9.1,
        "C1 same-use core algorithms per theme pair; others": (0.29, 5.0, 9.1),
        "neighbour links C6-C1, C6-C8": (1, 6),
        "neighbour links C6 and the other five themes": [42, 44, 70, 100, 136],
        "links / expected, C1 and C8": (0.004, 0.09),
        "links / expected, other five themes": (0.26, 1.36),
        "data sources of C1, C8 own computation; C3 external databases (%)": (94, 91, 72),
        "C4-C7 own experiments (%)": (56, 83),
    }
    bad = 0
    for k, v in want.items():
        ok = got[k] == v
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'}  {k}: {got[k]}" + ("" if ok else f" (manuscript {v})"))
    print(f"{len(want) - bad} of {len(want)} numbers as in the manuscript; tables written to {OUT}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
