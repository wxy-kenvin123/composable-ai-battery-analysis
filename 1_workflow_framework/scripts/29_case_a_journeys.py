"""Derive the candidate journeys and the formula co-occurrence from committed tables.

Two journeys illustrate what the chain records. The first is the post-hoc
recheck: LiAlS2 sits at rank 40 of 56 under the executed model, below where a
rank-only reading stops, and re-enters through a deep-dive clause that was
declared after its published AIMD match had been seen -- an illustrative case,
not a policy effect. The second is a formula co-occurrence: for the top-ranked
Li-B-S survivors, the committed literature card's verdict and the AIMD
trajectory index both name the same composition, Li5B7S13.

The co-occurrence flag is a string-level check that the committed card
verdict names the formula AND the pinned family match contains the same
reduced formula. It does not establish author independence, affiliation
separation, or citation-graph separation; none of those were assessed. The
card verdict is model-written prose; the flag is locked to the committed
snapshot, so a re-export that changes the verdict text fails the regression
test.

This script reads only committed ``results/case_a_full`` tables, calls no
model, and writes one row per (candidate, station) plus a summary JSON that
the journey figure must read instead of hardcoding text.

Usage
-----
    python scripts/29_case_a_journeys.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "case_a_full"

#: The co-occurring formula, in the two spellings the two artifacts use: the
#: loader reduces mp-532413 to alphabetical "B7Li5S13"; card prose writes
#: Li5B7S13 (often with Unicode subscripts, normalised below).
ANCHOR_REDUCED = "B7Li5S13"
ANCHOR_PROSE = "LI5B7S13"
ANCHOR_DISPLAY = "Li5B7S13"

_SUBSCRIPTS = str.maketrans("\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089",
                            "0123456789")


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalise(text: str) -> str:
    return (text or "").translate(_SUBSCRIPTS).upper()


def verdict_names_anchor(card: dict) -> bool:
    return ANCHOR_PROSE in normalise(card.get("corpus_verdict", ""))


def family_match_contains_anchor(virtual: dict) -> bool:
    return ANCHOR_REDUCED in (virtual.get("aimd_family_formulas") or "").split(",")


def derive(evidence: Path = EVIDENCE) -> tuple[list[dict], dict]:
    record = json.loads((evidence / "run_record_case_a_full.json").read_text(encoding="utf-8"))
    cards = {r["composition"]: r for r in read_csv(evidence / "table_case_a_cards.csv")}
    virtual = {r["composition"]: r
               for r in read_csv(evidence / "table_case_a_virtual_feedback.csv")}
    value = {r["composition"]: r
             for r in read_csv(evidence / "table_case_a_value_added.csv")}
    suggestions = read_csv(evidence / "table_case_a_suggestions.csv")
    n_top = int(record["n_top"])

    rule_citations: dict[str, set] = {}
    for row in suggestions:
        keys = [c for c in row["citations"].split(";") if c]
        rule_citations.setdefault(row["composition"], set()).update(keys)

    rows: list[dict] = []
    co_occurring: list[str] = []
    for composition in record["deep_dive"]:
        card = cards[composition]
        fb = virtual[composition]
        va = value[composition]
        rank = int(card["rank"])
        co_occurs = verdict_names_anchor(card) and family_match_contains_anchor(fb)
        if co_occurs:
            co_occurring.append(composition)

        executed_fact = (
            f"log10 sigma = {float(fb['executed_log10_sigma']):.2f} (rank {rank}/56)"
            + ("; the one prediction above the 1e-4 S/cm convention"
               if fb["executed_above_convention"] == "True" else "")
        )
        if fb["aimd_exact_match"] == "True":
            sigma_800 = fb.get("aimd_sigma_mS_cm_800K")
            aimd_fact = (
                f"exact AIMD match ({fb['aimd_exact_mp_id']}); the release labels it "
                "among nine superionic conductors (800-1700 K, structure-specific)"
                + (f"; sigma(800 K) = {float(sigma_800):.2f} mS/cm from the authors' "
                   "own analysis objects" if sigma_800 else "")
            )
        elif fb["aimd_family_mp_ids"]:
            aimd_fact = (
                f"family AIMD only: {fb['aimd_family_formulas']} "
                f"({fb['aimd_family_mp_ids']}); not this stoichiometry"
            )
        else:
            aimd_fact = "no published simulation for this composition or its system"
        retrieval_fact = (
            f"{card['n_papers']} papers above the relevance floor; "
            f"{card['n_failure_modes']} cited failure modes; "
            + ("family precedent cited" if int(card["n_family_precedent"]) > 0
               else "no family precedent")
        )
        structures = int(card["structures_published"])
        stations = [
            ("served", "PropertyScreener", "lookup",
             "published electrochemical survivor, carried unchanged",
             "ML-HTVS release", ""),
            ("executed", "ExecutedToolScreener", "executed",
             executed_fact, "ML-SSE model.joblib", ""),
            ("virtual experiment", "SimulationFeedback", "simulation",
             aimd_fact, "LiSSE-MP AIMD trajectories",
             "[@kang2023discovery]" if (fb["aimd_exact_match"] == "True"
                                        or fb["aimd_family_mp_ids"]) else ""),
            ("retrieval", "KnowledgeMiner", "retrieval",
             retrieval_fact, "authors' 723-paper corpus", card["citations"]),
            ("structures", "MechanismSimulator", "lookup",
             (f"{structures} released structures" if structures
              else "no structure in the release; its record stops here"),
             "ML-HTVS release", ""),
            ("action", "FormulationDesigner", "rule",
             f"{va['disposition']} ({va['rules_fired']})",
             "fixed rule set (exported)",
             ";".join(sorted(rule_citations.get(composition, set())))),
        ]
        for index, (station, role, grade, fact, source, citation) in enumerate(stations, 1):
            rows.append({
                "composition": composition,
                "rank": rank,
                "station_index": index,
                "station": station,
                "role": role,
                "grade": grade,
                "fact": fact,
                "source": source,
                "citation": citation,
                "baseline_stops_after_station": 2 if rank > n_top else "",
                "posthoc_deep_dive_admission": rank > n_top,
                "same_formula_in_both_artifacts": co_occurs,
            })

    summary = {
        "n_journeys": len(record["deep_dive"]),
        "anchor_formula": ANCHOR_DISPLAY,
        "co_occurring_compositions": co_occurring,
        "co_occurrence_scope": (
            "string-level: the committed card verdict names the formula and the "
            "pinned family match contains the same reduced formula; author "
            "independence and citation-graph separation were not assessed"
        ),
        "posthoc_added": [c for c in record["deep_dive"]
                          if int(cards[c]["rank"]) > n_top],
        "posthoc_note": (
            "the AIMD-match deep-dive clause was declared post hoc, after this "
            "candidate's match had been seen; illustrative, not a policy effect"
        ),
        "measurement_note": (
            "this run used no measurement; the three linked releases contain no "
            "measured-cell result used here"
        ),
    }
    return rows, summary


def main() -> int:
    rows, summary = derive()
    out = EVIDENCE / "table_case_a_journeys.csv"
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    summary_path = EVIDENCE / "journeys_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"wrote {summary_path}")

    print(f"\n  journeys                    : {summary['n_journeys']}")
    print(f"  post-hoc deep-dive admission: {', '.join(summary['posthoc_added'])}")
    print(f"  formula co-occurrence on {summary['anchor_formula']}:")
    for composition in summary["co_occurring_compositions"]:
        print(f"    {composition}: committed card verdict names it AND the pinned "
              "family match contains it")
    print(f"  scope: {summary['co_occurrence_scope']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
