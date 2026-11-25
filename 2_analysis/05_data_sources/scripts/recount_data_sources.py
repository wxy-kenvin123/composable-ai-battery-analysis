"""Recount the data-source tables and the numbers of Supplementary Note 1 and Supplementary Table 3.

Reads data/evidence.csv (4 columns: paper_id, category, dataset, role), data/paper_summary.csv
(theme, year and flags of the 755 articles), data/database_names.csv (canonical names and the narrow exclusions), the
article-country records of 09_maps and the database hosts of 09_maps. Recomputes every summary table of data/ (writing
them to data/ when --write-data is passed or when they are not yet present, and comparing cell by cell when already
present) and checks every number printed in the manuscript. Writes rerun/recount.json and exits with status 1 if
anything differs.

Usage: python scripts/recount_data_sources.py [--write-data]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SEC = Path(__file__).resolve().parents[1]
DATA = SEC / "data"
MAPS = SEC.parent / "09_maps" / "data"
OUT = SEC / "rerun"
CATS = ["public_db", "public_model", "own_computation", "own_experiment", "literature", "industry_field", "unclear"]
N = 755


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-data", action="store_true", help="Write recomputed summary tables into data/")
    args = ap.parse_args()
    bad: list[str] = []

    def check(ok: bool, what: str) -> None:
        print(("ok    " if ok else "FAIL  ") + what)
        if not ok:
            bad.append(what)

    ev = read(DATA / "evidence.csv")
    summ = {r["paper_id"]: r for r in read(DATA / "paper_summary.csv")}
    papers = {pid: {"paper_id": pid, "cluster": r["cluster"], "year": r["year"]} for pid, r in summ.items()}
    names = read(DATA / "database_names.csv")
    excluded = {r["dataset"] for r in names if r["narrow_excluded"] == "True"}
    countries = defaultdict(set)
    for r in read(MAPS / "country_article_records.csv"):
        countries[r["article_id"]].add(r["country"].strip())
    check(len(papers) == N, f"{len(papers)} articles")

    per = defaultdict(list)
    for r in ev:
        per[r["paper_id"]].append(r)
    check(set(per) == set(papers), "every article has at least one evidence row")
    check(all(r["category"] in CATS for r in ev), "categories are the seven of the rules")
    check({r["dataset"] for r in ev if r["category"] == "public_db"} <= {r["dataset"] for r in names},
          "every public_db name is a canonical name of database_names.csv")
    flags = {p: {r["category"] for r in rows} for p, rows in per.items()}
    check(all(len(f) == 1 for p, f in flags.items() if "unclear" in f), "unclear only where an article has no other category")
    public = {p: {r["dataset"] for r in rows if r["category"] == "public_db"} for p, rows in per.items()}

    # ------------------------------------------------------------------ the distributed tables
    off = [p for p in papers if p not in summ
           or any(summ[p][c] != str(c in flags[p]) for c in CATS)
           or summ[p]["public_datasets"] != ";".join(sorted(public[p]))
           or (summ[p]["cluster"], summ[p]["year"]) != (papers[p]["cluster"], papers[p]["year"])]
    check(not off, f"paper_summary.csv flags and dataset lists ({len(off)} articles differ)")
    keys = {"cluster": lambda p: [papers[p]["cluster"]], "year": lambda p: [papers[p]["year"]],
            "country": lambda p: countries[p]}
    for column, key_of in keys.items():
        want = defaultdict(Counter)
        for p in papers:
            for k in key_of(p):
                want[k]["n_papers"] += 1
                for c in flags[p]:
                    want[k][c] += 1
        target = DATA / f"{column}_summary.csv"
        if args.write_data or not target.exists():
            sort_key = (lambda k: int(k)) if column in ("cluster", "year") else (lambda k: k)
            rows_out = [
                {column: k, "n_papers": want[k]["n_papers"], **{c: want[k][c] for c in CATS}}
                for k in sorted(want, key=sort_key)
            ]
            write_csv(target, [column, "n_papers", *CATS], rows_out)
        got = {r[column]: Counter({c: int(v) for c, v in r.items() if c != column and int(v)})
               for r in read(target)}
        check(got == dict(want), f"{column}_summary.csv ({len(got)} rows)")
    for column in ("cluster", "country"):
        want = Counter((d, k) for p in papers for d in public[p] for k in keys[column](p))
        target = DATA / f"dataset_by_{column}.csv"
        if args.write_data or not target.exists():
            sort_key = (lambda item: (item[0][0], int(item[0][1]))) if column == "cluster" else (lambda item: (item[0][0], item[0][1]))
            rows_out = [{"dataset": d, column: k, "n_papers": n} for (d, k), n in sorted(want.items(), key=sort_key)]
            write_csv(target, ["dataset", column, "n_papers"], rows_out)
        got = Counter({(r["dataset"], r[column]): int(r["n_papers"]) for r in read(target)})
        check(got == want, f"dataset_by_{column}.csv ({len(got)} rows)")
    catalog = defaultdict(set)
    for p in papers:
        for d in public[p]:
            catalog[d].add(p)
    cat_target = DATA / "dataset_catalog.csv"
    if args.write_data or not cat_target.exists():
        rows_out = [{"dataset": d, "n_papers": len(p), "paper_ids": ";".join(sorted(p))} for d, p in sorted(catalog.items())]
        write_csv(cat_target, ["dataset", "n_papers", "paper_ids"], rows_out)
    got = {r["dataset"]: (int(r["n_papers"]), r["paper_ids"]) for r in read(cat_target)}
    check(got == {d: (len(p), ";".join(sorted(p))) for d, p in catalog.items()}, f"dataset_catalog.csv ({len(got)} databases)")

    # ------------------------------------------------------------------ numbers of the manuscript
    narrow = {d: p for d, p in catalog.items() if d not in excluded}
    narrow_papers = set().union(*narrow.values())
    broad_papers = set().union(*catalog.values())
    roles = defaultdict(set)
    for r in ev:
        if r["category"] == "public_db" and r["dataset"] not in excluded:
            roles[r["paper_id"]].add(r["role"])
    only_other = sum(1 for s in roles.values() if s == {"other"})
    by_cat = {c: sum(c in f for f in flags.values()) for c in CATS}
    theme_n = Counter(papers[p]["cluster"] for p in papers)
    theme_pct = {c: {k: round(100 * sum(c in flags[p] for p in papers if papers[p]["cluster"] == k) / theme_n[k])
                     for k in sorted(theme_n)} for c in CATS[:-1]}
    top = sorted(((len(p), d) for d, p in narrow.items()), key=lambda t: (-t[0], t[1]))
    top5 = [(d, n) for n, d in top if n >= 5]
    hubs = {r["dataset"]: r for r in read(MAPS / "hubs.csv")}
    stats_target = DATA / "final_stats.json"
    if args.write_data or not stats_target.exists():
        top_desc = sorted(((len(p), d) for d, p in narrow.items()), reverse=True)
        top10 = top_desc[:10]
        tenth_n = top10[-1][0]
        tie_at_10th = sorted(d for n, d in top_desc if n == tenth_n)
        top10_union = len(set().union(*(narrow[d] for _, d in top10)))
        stats_obj = {
            "evidence_rows": len(ev),
            "broad_articles": len(broad_papers),
            "narrow_articles": len(narrow_papers),
            "narrow_top10": [[n, d] for n, d in top10],
            "top10_union": top10_union,
            "tie_at_10th": tie_at_10th,
        }
        stats_target.write_text(json.dumps(stats_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    stats = json.loads(stats_target.read_text(encoding="utf-8"))
    f = {
        "narrow_articles": len(narrow_papers), "narrow_pct": round(100 * len(narrow_papers) / N, 1),
        "narrow_for_training_screening_validation": len(narrow_papers) - only_other, "narrow_other_use_only": only_other,
        "distinct_narrow": len(narrow), "used_by_2plus": sum(len(p) >= 2 for p in narrow.values()),
        "used_by_5plus": len(top5), "broad_articles": len(broad_papers), "broad_pct": round(100 * len(broad_papers) / N, 1),
        "excluded_names": len(excluded), "articles_by_category": by_cat, "theme_share_pct": theme_pct,
        "theme_articles": dict(sorted(theme_n.items())), "top_databases": top5,
        "top_hosted_in_united_states": sum(hubs[d]["host_country"] == "United States" for d, _ in top5),
    }
    check(set(hubs) == {d for d, _ in top5}, "09_maps/data/hubs.csv lists exactly the databases used by at least five articles")
    check((stats["narrow_articles"], stats["broad_articles"], stats["evidence_rows"]) == (len(narrow_papers), len(broad_papers), len(ev)),
          "final_stats.json")
    expected = {
        "narrow_articles": 257, "narrow_pct": 34.0, "narrow_for_training_screening_validation": 247, "narrow_other_use_only": 10,
        "distinct_narrow": 107, "used_by_2plus": 38, "used_by_5plus": 11, "broad_articles": 270, "broad_pct": 35.8,
        "excluded_names": 13,
        "articles_by_category": {"public_db": 270, "public_model": 124, "own_computation": 446, "own_experiment": 299,
                                 "literature": 257, "industry_field": 7, "unclear": 29},
        "theme_share_pct": {
            "public_db": {"1": 28, "2": 46, "3": 72, "4": 5, "5": 23, "6": 24, "7": 15, "8": 28},
            "public_model": {"1": 27, "2": 15, "3": 22, "4": 9, "5": 8, "6": 0, "7": 4, "8": 33},
            "own_computation": {"1": 94, "2": 56, "3": 59, "4": 46, "5": 49, "6": 27, "7": 9, "8": 91},
            "own_experiment": {"1": 14, "2": 45, "3": 16, "4": 56, "5": 83, "6": 66, "7": 68, "8": 0},
            "literature": {"1": 41, "2": 42, "3": 49, "4": 9, "5": 24, "6": 12, "7": 51, "8": 16},
            "industry_field": {"1": 0, "2": 1, "3": 1, "4": 3, "5": 0, "6": 3, "7": 0, "8": 0}},
        "theme_articles": {"1": 153, "2": 110, "3": 160, "4": 79, "5": 104, "6": 59, "7": 47, "8": 43},
        "top_databases": [("Materials Project database", 142), ("ICSD", 25), ("PubChem", 24), ("OQMD", 11), ("QM9", 9),
                          ("AFLOW", 8), ("Liverpool Ionics Dataset (LiIon)", 8), ("ILThermo", 6), ("ZINC", 6),
                          ("COD", 5), ("NASA", 5)],
        "top_hosted_in_united_states": 7,
    }
    for k, v in expected.items():
        got_v = f[k]
        if isinstance(v, list):
            got_v = [tuple(x) for x in got_v]
        check(got_v == v, f"{k} = {v if not isinstance(v, dict) else '(table)'} as in Supplementary Note 1 and Table 3"
              + ("" if got_v == v else f" (recounted {got_v})"))
    OUT.mkdir(exist_ok=True)
    (OUT / "recount.json").write_text(json.dumps(f, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{'PASS' if not bad else 'FAIL'}: {len(bad)} difference(s); written {OUT / 'recount.json'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
