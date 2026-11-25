"""Count the battery-side terms of the 755 analysed articles in their full texts (Supplementary Fig. 4).

Reference term-counting script (requires non-redistributed full texts and abstracts).

Usage:
  python scripts/count_mentions.py --texts <DIR> --records <CSV> --themes <CSV>
    --texts    folder of <id>.txt: full texts without the reference sections, the texts of the algorithm detection
    --records  CSV with the columns id, year, title and abstract of the articles
    --themes   03_clustering/output/data/clustering/article_clusters_analysis.csv (article, Cluster)

Counts (one row per article)
  words           words of the full text
  c_<component>   mentions of anode, cathode, electrolyte, interphase, separator, binder and current collector in the
                  full text; "solid electrolyte interphase" and "cathode electrolyte interphase" count as interphase
                  only, and the abbreviations of solid, polymer and composite electrolytes count as electrolyte
  k_<kind>        mentions of the three kinds of electrolyte in the full text (liquid electrolytes are recognized
                  by specific phrases only, because words such as solvent, carbonate and additive occur in articles
                  on every kind); a_<kind> the same in title and abstract
  n_<ion>         mentions of each working ion in the full text; t_<ion> the same in title and abstract
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]


def clean(s: str) -> str:
    s = "" if not isinstance(s, str) else s
    s = re.sub(r"</?(inf|sup|sub)>", "", s)
    s = re.sub(r"[‐‑‒–—−]", "-", s)
    return re.sub(r"\s+", " ", s).strip()


MASKS = [
    (r"(?i)solid[- ]electrolyte[- ]inter(phase|face)s?", " SEItoken "),
    (r"(?i)cathode[- ]electrolyte[- ]inter(phase|face)s?", " CEItoken "),
]

# ---- components (case-insensitive, on the masked text)
COMPONENT = {
    "Anode": r"\banodes?\b|negative electrodes?",
    "Cathode": r"\bcathodes?\b|positive electrodes?",
    "Electrolyte": r"\belectrolytes?\b",
    "Interphase": r"seitoken|ceitoken|\bsei\b|\bcei\b|interphases?\b",
    "Separator": r"\bseparators?\b",
    "Binder": r"\bbinders?\b",
    "Current collector": r"current collectors?",
}
# abbreviations of solid, solid polymer, gel polymer and composite polymer electrolytes; SE(3) is a symmetry group
ELECTROLYTE_ABBR = r"\bSSEs?\b|\bSEs\b|\bSE\b(?!\()|\bSPEs?\b|\bGPEs?\b|\bCPEs?\b"
# ---- kind of electrolyte
ELECTROLYTE_KIND = {
    "Inorganic solid": (r"solid[- ]state electrolyte|solid electrolyte|inorganic electrolyte|superionic|ionic conductor|"
                        r"ion conductors?\b|ion[- ]conducting (?:crystal|glass|ceramic|oxide|sulfide|halide)|"
                        r"argyrodite|garnet|thiophosphate|\bllzo\b|\blgps\b|\blatp\b|\bllto\b|nasicon|lisicon|ceramic|"
                        r"glass(?:y)? electrolyte|(?:sulfide|oxide|halide|chloride)[- ](?:based )?(?:solid[- ]state |solid )?electrolyte|"
                        r"anti-?perovskite"),
    "Polymer": (r"polymer electrolyte|polymer-based electrolyte|polymeric electrolyte|solid polymer|gel polymer|\bpeo\b|"
                r"poly\(ethylene oxide\)|polymer-in-salt|composite polymer|polymer solid[- ]state|polymer[- ]based solid|"
                r"polymer solid electrolyte"),
    "Liquid": (r"liquid electrolyte|electrolyte solution|aqueous electrolyte|non-?aqueous electrolyte|"
               r"organic electrolyte|ionic liquids?|water-in-salt|co-?solvent|diluent|eutectic|"
               r"solvation (?:structure|shell|sheath)|concentrated electrolyte|electrolyte additives?|"
               r"carbonate[- ]based electrolyte|ether[- ]based electrolyte"),
}
# ---- working ion (case-sensitive element symbols on the unmasked text)
# "Li" as a surname is excluded when followed by "et al", by initials or by "and"/"&" with another name.
ION = {
    "Li": (r"[Ll]ithium|\bLi(?![a-z])(?! et al)(?!,? [A-Z]\.)(?! (?:and|&) [A-Z][a-z])|\bLIBs?\b|\bLMBs?\b|\bLLZO\b|"
           r"\bLGPS\b|\bLATP\b|\bLLTO\b|\bLFP\b|\bLCO\b|\bNMC\d*\b|\bNCM\d*\b|\bNCA\b"),
    "Na": r"\b[Ss]odium|\bNa(?![a-z])|\bSIBs?\b|\bNIBs?\b",
    "K": r"\b[Pp]otassium|\bK\+|\bK[- ]ion|\bPIBs?\b|\bKIBs?\b",
    "Zn": r"\b[Zz]inc|\bZn(?![a-z])|\bA?ZIBs?\b",
    "Mg": r"\b[Mm]agnesium|\bMg(?![a-z])",
    "Ca": r"\b[Cc]alcium|\bCa(?![a-z])(?!NN)",                   # CaNN is a cascaded neural network
    "Al": r"[Aa]lumin(?:i)?um[- ]ion|\bAl[- ]ion|\bAl3\+|[Aa]lumin(?:i)?um batter|[Aa]lumin(?:i)?um[- ]air|\bAIBs?\b",
    "Flow and other": (r"[Rr]edox[- ]flow|\bRFBs?\b|\bVRFBs?\b|\bVFBs?\b|[Ll]ead[- ]acid|[Nn]ickel[- ]metal hydride|"
                       r"[Ff]luoride[- ]ion|[Pp]roton batter|[Vv]anadium (?:redox )?flow|[Aa]ll-vanadium"),
}


def masked_lc(s: str) -> str:
    for pat, tok in MASKS:
        s = re.sub(pat, tok, s)
    return s.lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True, type=Path)
    ap.add_argument("--records", required=True, type=Path)
    ap.add_argument("--themes", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=HERE / "data" / "mention_counts.csv")
    a = ap.parse_args()
    d = pd.read_csv(a.records, dtype=str)
    c = pd.read_csv(a.themes, dtype=str)
    m = c.merge(d[["id", "year", "title", "abstract"]], left_on="article", right_on="id", how="left")
    assert len(m) == 755 and m["abstract"].notna().all() and m["id"].is_unique
    m["theme"] = "C" + m["Cluster"]
    ta = m["title"].map(clean) + ". " + m["abstract"].map(clean)
    text = pd.Series([clean((a.texts / f"{i}.txt").read_text(encoding="utf-8", errors="ignore")) for i in m["id"]],
                     index=m.index)
    assert (text.str.len() > 3000).all()
    m["words"] = text.str.split().str.len()
    lc, ta_lc = text.map(masked_lc), ta.map(masked_lc)
    for name, pat in COMPONENT.items():
        m["c_" + name] = lc.str.count(pat)
    m["c_Electrolyte"] += text.str.count(ELECTROLYTE_ABBR)               # abbreviations, on the cased text
    for name, pat in ELECTROLYTE_KIND.items():
        m["k_" + name] = lc.str.count(pat)
        m["a_" + name] = ta_lc.str.count(pat)
    for name, pat in ION.items():
        m["n_" + name] = text.str.count(pat)
        m["t_" + name] = ta.str.count(pat)
    keep = (["id", "theme", "year", "title", "words"] + ["c_" + k for k in COMPONENT]
            + ["k_" + k for k in ELECTROLYTE_KIND] + ["a_" + k for k in ELECTROLYTE_KIND]
            + ["n_" + k for k in ION] + ["t_" + k for k in ION])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    m[keep].to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"{len(m)} articles -> {a.out}")


if __name__ == "__main__":
    main()
