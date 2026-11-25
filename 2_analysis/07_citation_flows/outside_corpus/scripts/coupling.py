"""Share of each theme's references outside the corpus that cite works also cited by another theme.

Package version of the coupling analysis of 1 November 2026: the computation is unchanged; it writes the last row of
Supplementary Table 4b (output/coupling_share_by_theme.csv) and leaves out the exploratory printouts. Read-only.

    python scripts/coupling.py [OUTDIR]          (default rerun/)
"""
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
D = HERE.parent / "data"                          # 07_citation_flows/data
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "rerun"
OUT.mkdir(parents=True, exist_ok=True)
it = pd.read_csv(D / "reference_items.csv", dtype=str)
it = it[it.included_in_D == "1"].copy()
corp = pd.read_csv(D / "corpus.csv", dtype=str)


def ndoi(s):
    s = str(s or "").strip().lower()
    s = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", s)
    return s if s.startswith("10.") else ""


it["doi"] = it.normalized_doi.map(ndoi)
cdoi = {ndoi(d) for d in corp.doi}
it["theme"] = it.id.map(dict(zip(corp.id, corp.Cluster.astype(int))))
ext = it[(it.doi != "") & ~it.doi.isin(cdoi)]                     # reference items with a DOI outside the corpus
cnt = ext.drop_duplicates(["id", "doi"]).groupby(["theme", "doi"]).size()
sets = {c: set(cnt.loc[c].index) for c in range(1, 9)}             # distinct external works cited by each theme
themes_per_work = Counter(d for c in range(1, 9) for d in sets[c])
multi = {d for d, k in themes_per_work.items() if k >= 2}
per = ext.drop_duplicates(["id", "doi"])
rows = []
for a in range(1, 9):
    pa = per[per.theme == a]
    others = set().union(*[sets[b] for b in range(1, 9) if b != a])
    rows.append({"theme": f"C{a}", "items": len(pa), "share_pct": round(100 * pa.doi.isin(others).mean(), 1)})
rows.append({"theme": "All", "items": len(ext), "share_pct": round(100 * ext.doi.isin(multi).mean(), 1)})
pd.DataFrame(rows).to_csv(OUT / "coupling_share_by_theme.csv", index=False, encoding="utf-8")
print(f"distinct external works {len(themes_per_work)}, cited by two or more themes {len(multi)}")
print(pd.DataFrame(rows).to_string(index=False))
