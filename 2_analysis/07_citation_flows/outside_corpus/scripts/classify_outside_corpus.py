# What the corpus cites outside itself: classify reference items by OpenAlex topic and by AI terms in the title.
# Run with python -I; argv[1] = folder of OpenAlex title responses (untrusted), argv[2] = the folder
# with reference_items_openalex.csv written by classify_fields.py (outputs go there too).
#     python -I scripts/classify_outside_corpus.py raw/titles rerun
import json
import re
import sys
from pathlib import Path

import pandas as pd

RAW, OUTDIR = Path(sys.argv[1]), Path(sys.argv[2])
D = Path(__file__).resolve().parents[2] / "data"     # 07_citation_flows/data


def ndoi(s):
    s = str(s or "").strip().lower()
    s = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", s)
    return s if s.startswith("10.") else ""


title = {}
for f in sorted(RAW.glob("batch_*.json")):
    for w in json.loads(f.read_text(encoding="utf-8")).get("results", []):
        d = ndoi(w.get("doi"))
        if d:
            title[d] = str(w.get("title") or "")
o = pd.read_csv(OUTDIR / "reference_items_openalex.csv", dtype=str)
o["title"] = o.doi.map(title).fillna("")
ext = o[o.kind == "classified"].copy()
print("external classified items", len(ext), "| with title", (ext.title != "").mean().round(4))

AI_RX = re.compile(
    r"machine[- ]?learn|deep[- ]?learn|neural|artificial intelligen|\bai\b|data[- ]driven|random forest|"
    r"gaussian process|bayesian optimi|active learning|transfer learning|support vector|gradient boost|xgboost|"
    r"decision tree|graph network|convolutional|recurrent|lstm|transformer|language model|\bllms?\b|generative|"
    r"autoencoder|reinforcement learning|symbolic regression|explainable|interpretable machine|deep potential|"
    r"machine[- ]learned|genetic algorithm|evolutionary|particle swarm|intelligent|learning", re.I)
BATT = re.compile(r"batter", re.I)
METHOD_FIELDS = {"Computer Science", "Mathematics", "Decision Sciences"}
CHEMPHYS_SUB = {"Atomic and Molecular Physics, and Optics", "Physical and Theoretical Chemistry",
                "Statistical and Nonlinear Physics", "Condensed Matter Physics", "Nuclear and High Energy Physics"}
DOMAIN_FIELDS = {"Materials Science", "Chemistry", "Chemical Engineering", "Physics and Astronomy", "Energy",
                 "Engineering"}

ext["ai"] = ext.title.str.contains(AI_RX)
ext["batt_topic"] = ext.topic.fillna("").str.contains(BATT) | ext.title.str.contains(BATT)


def cat(r):
    if r.field in METHOD_FIELDS:
        return "1 AI, statistics and computing methods"
    if r.ai and r.batt_topic:
        return "2 AI for batteries"
    if r.ai:
        return "3 AI for other materials and chemistry"
    if r.batt_topic:
        return "4 Battery research without AI"
    if r.subfield in CHEMPHYS_SUB:
        return "5 Computational and physical chemistry"
    if r.field in DOMAIN_FIELDS:
        return "6 Other materials, chemistry and engineering"
    return "7 Other fields"


ext["cat"] = ext.apply(cat, axis=1)
n = len(ext)
print("\ncategory share of classified external items (%):")
print((ext.cat.value_counts().sort_index() / n * 100).round(1).to_string())
print("\ncategory by theme (%):")
tab = (pd.crosstab(ext.theme, ext.cat, normalize="index") * 100).round(1)
print(tab.to_string())
tab.to_csv(OUTDIR / "category_share_by_theme.csv", encoding="utf-8")

# (exploratory printouts that the paper does not report are left out)
ext.to_csv(OUTDIR / "external_items_categorized.csv", index=False, encoding="utf-8")
