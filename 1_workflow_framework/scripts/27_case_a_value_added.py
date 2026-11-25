"""What the chain added, against a rank-only reading of the executed model.

The executed ML-SSE ranking already exists without any framework: it is exactly
what ``ml_htvs_continuation.py`` exports. The baseline for "what did the chain
add" is a rank-only reading of that output -- examine the top few, set the rest
aside -- with no gate invented on top of it. That baseline has two limits:

- The "top 5" budget is this run's retrieval budget, not a published decision
  rule, so the five constrained advances are partly a consequence of the
  analysis design.
- The deep-dive clause that admits AIMD-matched survivors was declared post
  hoc, after the LiAlS2 match had been seen. Its one admission is therefore an
  illustrative case, not a measured selection-policy effect, and it is counted
  separately.

Accounting, in three disjoint buckets:

- ``downstream disposition added``: a top-N candidate whose record gained fired
  fixed rules from the deep-dive stages (5 in the committed run).
- ``post-hoc illustrative addition``: a deep-dive candidate admitted by the
  post-hoc AIMD clause (1). Not evidence of policy value.
- ``carried baseline warning``: an out-of-training flag the executed stage had
  already recorded in the continuation export. The chain carries it; it does
  not create it (4).

This script reads committed CSVs, calls no model, retrieves nothing, and is
deterministic. It writes the per-candidate table plus a summary JSON that the
figure script must read -- figures are not allowed to hardcode these numbers.

Usage
-----
    python scripts/27_case_a_value_added.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "case_a_full"

sys.path.insert(0, str(ROOT))

from battery_workflow.knowledge_backend import composition_context  # noqa: E402

#: Highest-priority fired rule wins the *displayed* disposition. This order is
#: an analysis-time declaration (consequence first), not a property of the run;
#: the full fired-rule list and every involved role are exported beside it.
DISPOSITION_OF_RULE = [
    ("condition-mismatch-recheck",
     "re-assess under aligned conditions (condition-mismatched AIMD)",
     "SimulationFeedback"),
    ("screen-failure-modes",
     "screen cited failure modes first", "KnowledgeMiner"),
    ("structure-check",
     "enumerate structures first", "MechanismSimulator"),
    ("structure-reuse",
     "check against released structures", "MechanismSimulator"),
    ("family-simulated",
     "reuse the published AIMD protocol for the family", "SimulationFeedback"),
    ("family-route",
     "reuse the cited family route", "KnowledgeMiner"),
    ("no-precedent",
     "no corpus precedent; simulate before synthesis", "KnowledgeMiner"),
    ("convention-note",
     "field convention annotated, never enforced", "ExecutedToolScreener"),
    ("provisional-rank",
     "rank provisional (out-of-training element)", "ExecutedToolScreener"),
]

RULE_ROLE = {rule_id: role for rule_id, _, role in DISPOSITION_OF_RULE}
RANK_ONLY = "executed rank only"
CARRIED = "carried baseline warning (out-of-training element)"


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def derive(evidence: Path = EVIDENCE) -> tuple[list[dict], dict]:
    """Return (per-candidate rows, summary). Everything comes from committed tables."""
    record = json.loads((evidence / "run_record_case_a_full.json").read_text(encoding="utf-8"))
    ranked = sorted(read_csv(evidence / "table_case_a_ranked.csv"),
                    key=lambda row: int(row["rank"]))
    suggestions = read_csv(evidence / "table_case_a_suggestions.csv")
    feedback = {row["composition"]: row
                for row in read_csv(evidence / "table_case_a_virtual_feedback.csv")}
    cards = {row["composition"]: row
             for row in read_csv(evidence / "table_case_a_cards.csv")}
    trace = read_csv(evidence / "table_case_a_agent_trace.csv")

    n_top = int(record["n_top"])
    deep_dive = set(record["deep_dive"])
    rules_of: dict[str, list[str]] = {}
    for row in suggestions:
        rules_of.setdefault(row["composition"], []).append(row["rule_id"])
    citations_of: dict[str, set] = {}
    for row in suggestions:
        keys = [c for c in row["citations"].split(";") if c]
        citations_of.setdefault(row["composition"], set()).update(keys)

    known_rules = set(RULE_ROLE)
    rows: list[dict] = []
    for entry in ranked:
        composition = entry["composition"]
        rank = int(entry["rank"])
        ood = entry["ood_elements"]
        fired = rules_of.get(composition, [])
        unknown = [rule for rule in fired if rule not in known_rules]
        if unknown:
            raise RuntimeError(
                f"{composition}: fired rules {unknown} missing from DISPOSITION_OF_RULE; "
                "refusing to attribute silently"
            )
        card = cards.get(composition, {})
        virtual = feedback.get(composition, {})

        baseline = (
            f"top {n_top}: examine further" if rank <= n_top
            else "below the fold: not examined"
        )

        disposition, primary_role = RANK_ONLY, ""
        for rule_id, label, contributor in DISPOSITION_OF_RULE:
            if rule_id in fired:
                disposition, primary_role = label, contributor
                break
        roles_involved = sorted({RULE_ROLE[rule] for rule in fired})
        if disposition == RANK_ONLY and ood:
            disposition, primary_role = CARRIED, "ExecutedToolScreener"
            roles_involved = ["ExecutedToolScreener"]

        if composition in deep_dive and rank > n_top:
            change = "post-hoc illustrative addition"
        elif composition in deep_dive and fired:
            change = "downstream disposition added"
        elif ood:
            change = "carried baseline warning"
        else:
            change = "unchanged"

        rows.append({
            "rank": rank,
            "composition": composition,
            "baseline_reading": baseline,
            "deep_dive": composition in deep_dive,
            "ood_elements": ood,
            "ood_flag_origin": "continuation export (pre-existing)" if ood else "",
            "aimd_evidence": (
                "exact: " + virtual["aimd_exact_mp_id"] if virtual.get("aimd_exact_match") == "True"
                else "family: " + virtual["aimd_family_mp_ids"] if virtual.get("aimd_family_mp_ids")
                else "none" if virtual else ""
            ),
            "condition_mismatched_evidence": virtual.get("condition_mismatched_evidence", ""),
            "n_family_precedent": card.get("n_family_precedent", ""),
            "n_failure_modes": card.get("n_failure_modes", ""),
            "structures_published": card.get("structures_published", ""),
            "rules_fired": ";".join(fired),
            "n_rule_citations": len(citations_of.get(composition, set())),
            "disposition": disposition,
            "primary_role_displayed": primary_role,
            "roles_involved": ";".join(roles_involved),
            "change_class": change,
        })

    stage_touch: dict[str, int] = {}
    for step in trace:
        if step.get("kind") == "agent":
            stage_touch[step["stage"]] = stage_touch.get(step["stage"], 0) + 1

    # The query construction is deterministic and public (knowledge_backend), so
    # the per-composition question count is derivable without touching the corpus.
    questions_per_card = {
        composition: len(composition_context(composition)["queries"])
        for composition in sorted(deep_dive)
    }
    per_card_counts = set(questions_per_card.values())
    if len(per_card_counts) != 1:
        raise RuntimeError(f"non-uniform question counts: {questions_per_card}")
    questions_each = per_card_counts.pop()

    summary = {
        "n_candidates": len(rows),
        "n_top": n_top,
        "n_deep_dive": len(deep_dive),
        "n_downstream_added": sum(
            1 for r in rows if r["change_class"] == "downstream disposition added"),
        "n_posthoc_illustrative": sum(
            1 for r in rows if r["change_class"] == "post-hoc illustrative addition"),
        "n_carried_warnings": sum(
            1 for r in rows if r["change_class"] == "carried baseline warning"),
        "n_unchanged": sum(1 for r in rows if r["change_class"] == "unchanged"),
        "stage_touch": stage_touch,
        "retrieval_questions_asked": questions_each * len(deep_dive),
        "retrieval_questions_uniform": questions_each * len(rows),
        "retrieval_question_share": (questions_each * len(deep_dive))
                                    / (questions_each * len(rows)),
        "deep_dive_share": len(deep_dive) / len(rows),
        "deep_dive_rule": record["deep_dive_rule"],
        "baseline_note": (
            "rank-only reading of the executed shortlist; the top-N budget is this "
            "run's retrieval budget, not a published decision rule"
        ),
        "carried_note": (
            "out-of-training flags were already present in the continuation export; "
            "the chain carries them and is not credited for them"
        ),
        "posthoc_note": (
            "the AIMD-match clause was declared post hoc; its one admission is an "
            "illustrative case, not a policy-effect measurement"
        ),
    }
    return rows, summary


def main() -> int:
    rows, summary = derive()
    out = EVIDENCE / "table_case_a_value_added.csv"
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    summary_path = EVIDENCE / "value_added_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"wrote {summary_path}")

    print("\nAgainst a rank-only reading of the executed model")
    print("-" * 74)
    print(f"  downstream dispositions added      : {summary['n_downstream_added']} "
          f"(top-{summary['n_top']} candidates; partly a consequence of the budget)")
    print(f"  post-hoc illustrative addition     : {summary['n_posthoc_illustrative']} "
          "(AIMD clause declared after the match was seen)")
    print(f"  carried baseline warnings          : {summary['n_carried_warnings']} "
          "(pre-existing OOD flags from the continuation export)")
    print(f"  unchanged                          : {summary['n_unchanged']} of "
          f"{summary['n_candidates']}")
    print(f"  retrieval questions                : "
          f"{summary['retrieval_questions_asked']} asked vs "
          f"{summary['retrieval_questions_uniform']} uniform "
          f"({summary['retrieval_question_share']:.1%} of question count; "
          "no wall-time or token cost was measured)")
    print(f"  deep-dive rule                     : {summary['deep_dive_rule']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
