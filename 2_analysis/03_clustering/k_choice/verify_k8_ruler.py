"""Recompute signature-gain, sibling-overlap and AI/task MI checks over K = 1..16.

Run:  python -I verify_k8_ruler.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import contingency_matrix, expected_mutual_information

RUN = Path(__file__).resolve().parent
DATA = RUN / "data"
LAB = RUN.parents[1] / "04_algorithms" / "data"

N_SET = [5, 6, 7, 8, 10, 12, 15, 20, 25, 30]
N_SIB = [10, 15, 20]
results = []


def record(name, n, mismatches, max_diff=np.nan):
    ok = mismatches == 0
    results.append(dict(check=name, compared=n, mismatches=mismatches, max_abs_diff=max_diff, ok=ok))
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {n} compared, {mismatches} mismatches"
          + ("" if np.isnan(max_diff) else f", max |diff| {max_diff:.3g}"))


x_df = pd.read_csv(DATA / "x_log_permille.csv.gz", index_col=0, dtype={"id": str})
x_df.index = x_df.index.astype(str)
X = x_df.to_numpy(dtype=float)
P = X > 0
words = np.array(x_df.columns)
ids = x_df.index.to_numpy()
n = X.shape[0]
assert X.shape == (755, 1309)

asg = pd.read_csv(DATA / "assignments_k1_k16.csv", dtype={"id": str}).set_index("id").loc[ids]
G = {k: asg[f"k{k}"].to_numpy() for k in range(1, 17)}


def indval(g, k):
    avg = np.column_stack([X[g == j].mean(axis=0) for j in range(1, k + 1)])
    fr = np.column_stack([P[g == j].mean(axis=0) for j in range(1, k + 1)])
    tot = avg.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        iv = avg / tot * fr
    return np.nan_to_num(iv, nan=0.0, posinf=0.0, neginf=0.0)


def top_terms(iv, j, N):
    order = sorted(range(len(words)), key=lambda i: (-iv[i, j], words[i]))
    return [words[i] for i in order[:N]]


IV = {k: indval(G[k], k) for k in range(1, 17)}

U = {N: np.array([len(set().union(*[top_terms(IV[k], j, N) for j in range(k)])) for k in range(1, 17)])
     for N in N_SET}
sg = pd.read_csv(DATA / "signature_gain.csv")
r_all = sg[sg.term_set.str.startswith("all terms")]
mism = 0
for N in N_SET:
    ru = r_all[r_all.N == N].sort_values("K").U.to_numpy()
    mism += int((ru != U[N]).sum())
record("U_N(K), all terms, K = 1..16, 10 list lengths", 16 * len(N_SET), mism)

isa = pd.read_csv(DATA / "isa_by_k.csv.gz")


def isa_mats(k):
    d = isa[isa.K == k]
    iv = d.pivot(index="word", columns="cluster", values="indval").loc[words].to_numpy()
    pv = d.drop_duplicates("word").set_index("word").loc[words, "pval"].to_numpy()
    return iv, pv


def sig_lists(k, N):
    iv, pv = isa_mats(k)
    h = np.rint(np.round(iv, 2) * 100).astype(int)
    rows = np.flatnonzero((pv < 0.05) & (h.sum(axis=1) > k))
    return [[words[i] for i in sorted(rows, key=lambda i: -h[i, c])[:N]] for c in range(k)]


def sig_lists_raw(k, N):
    iv, pv = isa_mats(k)
    rows = np.flatnonzero((pv < 0.05) & (iv.mean(axis=1) > 0.01))
    return [[words[i] for i in sorted(rows, key=lambda i: (-iv[i, c], words[i]))[:N]] for c in range(k)]


wc = pd.read_csv(RUN.parent / "output" / "data" / "wordcloud_selected_terms.csv")
sl8 = sig_lists(8, 5)
record("significant-term lists K = 8 top 5 = word cloud", 40,
       int(sum(a != b for a, b in zip([w for l in sl8 for w in l], wc.word))))
for label, fn, prefix in [("significant terms, word-cloud rule", sig_lists, "significant indicator terms (word cloud rule)"),
                          ("significant terms, unrounded", sig_lists_raw, "significant indicator terms, unrounded")]:
    Us = {N: np.array([len(set().union(*fn(k, N))) for k in range(2, 17)]) for N in N_SET}
    r_set = sg[sg.term_set.str.startswith(prefix)]
    mism = sum(int((r_set[r_set.N == N].sort_values("K").U.to_numpy() != Us[N]).sum()) for N in N_SET)
    record(f"U_N(K), {label}, K = 2..16", 15 * len(N_SET), mism)

sib = pd.read_csv(DATA / "sibling_overlap.csv")
mism = 0
for k in range(1, 16):
    a, b = G[k], G[k + 1]
    tab = pd.crosstab(a, b)
    par = tab.index[(tab > 0).sum(axis=1) == 2]
    assert len(par) == 1
    ch = list(tab.columns[(tab.loc[par[0]] > 0).to_numpy()])
    row = sib.iloc[k - 1]
    ov = [len(set(top_terms(IV[k + 1], ch[0] - 1, N)) & set(top_terms(IV[k + 1], ch[1] - 1, N))) for N in N_SIB]
    mism += int(ov[0] != row.shared_top10) + int(ov[1] != row.shared_top15) + int(ov[2] != row.shared_top20)
    mism += int(int((a == par[0]).sum()) != row.parent_n)
record("sibling overlap, 15 splits x (3 depths + parent size)", 60, mism)


def labels(fname):
    d = pd.read_csv(LAB / fname, dtype={"id": str})
    return {c: np.isin(ids, d.id[d.category == c].to_numpy()).astype(int) for c in sorted(d.category.unique())}


AI, TK = labels("ai_type_doc_presence.csv"), labels("task_type_doc_presence.csv")
info = pd.read_csv(DATA / "ai_task_information.csv").set_index("K")
diffs = []
for k in range(1, 17):
    for lab, pre in [(AI, "ai"), (TK, "task")]:
        mi = sum(mutual_info_score(G[k], y) for y in lab.values())
        emi = sum(expected_mutual_information(contingency_matrix(G[k], y), n) for y in lab.values())
        diffs += [abs(mi - info.loc[k, f"{pre}_MI"]), abs(emi - info.loc[k, f"{pre}_EMI"])]
diffs = np.array(diffs)
record("MI and exact EMI, K = 1..16, AI and task labels", len(diffs), int((diffs > 1e-9).sum()), diffs.max())

(RUN / "rerun").mkdir(exist_ok=True)
pd.DataFrame(results).to_csv(RUN / "rerun" / "verification_summary.csv", index=False)
ok_n = sum(r["ok"] for r in results)
print(f"\n{ok_n} of {len(results)} checks passed")
sys.exit(0 if ok_n == len(results) else 1)
