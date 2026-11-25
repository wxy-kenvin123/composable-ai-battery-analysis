"""
Agents backed by a published tool's released data.

The generic agents in this package read their physical values from
``synthetic_controlflow_fixture.py``, which is fine for a control-flow smoke
test and useless as evidence. The subclasses here keep the agents' behaviour --
tool selection, handoff consumption, structured results, the actions the
Coordinator dispatches -- and replace only the source of the numbers: every
value an agent returns is looked up in a CSV released by the tool being
replayed, addressed by the row's own identifier.

That boundary is the point, and it is also the limit. These adapters do not run
ROOST, pymatgen, either Gaussian-process optimizer, a robot, or a cycler; they
serve what those programs already wrote down. The framework is what carries a candidate
between the stages, applies the published acceptance criteria, and records why
the ones that stop, stopped -- which is the part the 68-tool audit found missing
and the part these cases are meant to test.

    Case A, ML-HTVS-Li-SE      HtvsPropertyScreener -> HtvsStructureBuilder
    Case A, executed chain     HtvsSurvivorCarrier -> ExecutedToolScreener
                               -> PublishedAimdFeedback -> (KnowledgeMiner)
                               -> HtvsStructureBuilder
                               -> ShortlistFormulationDesigner

Two agents in that list are not lookups and say so in their own docstrings:
``ExecutedToolScreener`` runs a released model, and
``ShortlistFormulationDesigner`` fires fixed rules over what the other stages
produced. ``PublishedAimdFeedback`` is a lookup with a proviso: the trajectory
files it serves are the authors' own serialized analysis objects, and reporting
a conductivity from one means deserializing it, which recomputes their MSD
statistics from their stored displacements. ``KnowledgeMiner`` has no
published-data counterpart at all -- it needs the external literature index, and
a stub would inject text that no cited release contains -- so it is registered by
the executed Case A chain and by the smoke test, never by a replay.
"""

from __future__ import annotations

import json
from typing import Any

from ..candidate_record import CandidateRecord, ComponentSpec, Phase, StageResult
from ..datasets import lisse_mp, ml_htvs, ml_sse
from ..executors.ml_sse import CONVENTION_LOG10_S_CM, predict_formulas
from .experiment_feedback import ExperimentFeedback
from .formulation_designer import FormulationDesigner
from .mechanism_simulator import MechanismSimulator
from .property_screener import PropertyScreener


def _received(candidate: CandidateRecord, module_name: str) -> list[str]:
    """Field names the ProcessCoordinator wrote for this module upstream."""
    handoff = candidate.test_conditions.get("_handoff", {}).get(module_name, {})
    return list((handoff.get("fields") or {}).keys())


def _pinned(source: Any) -> str:
    """'repo @ commit' -- what every provenance string in this module starts with."""
    return f"{source.repository.rsplit('/', 1)[-1]} @ {source.commit[:8]}"


# --------------------------------------------------------------------------- #
# Case A -- ML-HTVS-Li-SE
# --------------------------------------------------------------------------- #

class HtvsPropertyScreener(PropertyScreener):
    """Serves the ROOST predictions and pymatgen stability windows as published.

    One row of the released screening tables per candidate, addressed by
    composition. Whether a property is present is itself published information:
    the upstream pipeline only computed band gaps for compositions that had
    already cleared the hull filter, so ``None`` here means "the authors never
    paid for this", and the gate that needs it will say so.
    """

    def __init__(self, dataset: Any):
        super().__init__()
        self.dataset = dataset
        self._by_id = {c.candidate_id: c for c in dataset.candidates}
        self._source = (
            f"published: {_pinned(ml_htvs.SOURCE)} "
            "data/02_6_e_hull_roost_results.csv, 03_2_band_gap_roost.csv, "
            "04_1_stability_window.csv"
        )

    def predict_material_properties(
        self, parameters: dict[str, Any], candidate: CandidateRecord
    ) -> StageResult:
        entry = self._by_id.get(candidate.candidate_id)
        if entry is None:
            raise KeyError(f"no published row for '{candidate.candidate_id}'")

        reasoning = self.select_tool_and_reason(
            task_desc=(
                "material-level screening of a substitution-generated lithium "
                f"solid electrolyte, {candidate.system_name}"
            ),
            candidate=candidate,
        )
        metrics: dict[str, Any] = dict(entry.metrics())
        metrics["composition"] = entry.composition
        metrics["properties_published"] = sum(1 for v in entry.metrics().values() if v is not None)
        metrics["_metrics_source"] = self._source
        metrics.update(reasoning)

        return self._make_result(
            phase=Phase.SCREENING,
            passed=True,          # the gates decide; the agent only serves the row
            metrics=metrics,
            artifacts={"rationale": reasoning["rationale"], "llm_call": self._last_llm_call_json},
        )


class HtvsStructureBuilder(MechanismSimulator):
    """Reports the structures the authors enumerated for a shortlist member.

    This is the stage the published pipeline ran next, and the one where its
    record thins out: the release contains enumerated structures for seven of
    the fifty-six survivors. The agent reports the absence for the other
    forty-nine instead of supplying a value, so a shortlist entry that has no
    published successor stage is visible as such in the exported evidence.
    """

    def __init__(self, dataset: Any):
        super().__init__()
        self.dataset = dataset
        self._source = f"published: {_pinned(ml_htvs.SOURCE)} data/06_2_drop_duplicate.csv"

    def get_supported_actions(self) -> list[str]:
        return ["enumerate_published_structures", "run_interface_simulation", "analyze_transport"]

    def enumerate_published_structures(
        self, parameters: dict[str, Any], candidate: CandidateRecord
    ) -> StageResult:
        composition = self.electrolyte_composition(candidate)
        structures = self.dataset.structures_for(composition)

        reasoning = self.select_tool_and_reason(
            task_desc=(
                "enumerate symmetry-distinct crystal structures for DFT validation of "
                f"{candidate.system_name}"
            ),
            candidate=candidate,
        )
        metrics: dict[str, Any] = {
            "published_structures": len(structures),
            "structure_data_published": bool(structures),
            "spacegroups": ",".join(sorted({s.spacegroup for s in structures})),
            "received_upstream_fields": _received(candidate, self.module_name),
            "_metrics_source": self._source if structures else (
                f"{self._source}: no row for this composition; the release stops here"
            ),
        }
        metrics.update(reasoning)

        return self._make_result(
            phase=Phase.SIMULATION,
            passed=bool(structures),
            metrics=metrics,
            artifacts={"rationale": reasoning["rationale"], "llm_call": self._last_llm_call_json},
        )


class HtvsSurvivorCarrier(PropertyScreener):
    """Publishes a Case A shortlist composition so the next agent can consume it.

    This is not a second screening. The composition has already survived the
    published ML-HTVS cascade; the agent exists so ProcessCoordinator has a
    real upstream field to translate into the second tool's input schema.
    """

    def __init__(self):
        super().__init__()
        self._source = (
            f"published: {_pinned(ml_htvs.SOURCE)} "
            "data/04_2_after_stability_window_screening.csv"
        )

    def predict_material_properties(
        self, parameters: dict[str, Any], candidate: CandidateRecord
    ) -> StageResult:
        composition = self.electrolyte_composition(candidate)
        # No registry selection here: this stage serves a published row and
        # selects nothing. Letting the role-level selector run would stamp the
        # downstream model's name onto a lookup row, which is exactly the trace
        # attribution the audit flagged: the tool a step consumed must be the
        # tool its row names.
        rationale = (
            "serves a published ML-HTVS survivor composition unchanged; no tool "
            "is selected or executed at this stage"
        )
        metrics: dict[str, Any] = {
            "composition": composition,
            "shortlist_member": True,
            "execution": False,
            "agent": self.agent,
            "selected_tool": None,
            "selected_repo": None,
            "selected_crl": None,
            "decision_source": "published-row lookup (ML-HTVS-Li-SE)",
            "rationale": rationale,
            "_metrics_source": self._source,
        }
        return self._make_result(
            phase=Phase.SCREENING,
            passed=True,
            metrics=metrics,
            artifacts={"rationale": rationale, "llm_call": ""},
        )


class ExecutedToolScreener(PropertyScreener):
    """Runs a released model. Distinct from the lookup adapters above.

    Predictions are batched once at construction so the 21 MB forest is loaded
    a single time. Each ``predict_ionic_conductivity`` call then serves the
    executed value for the formula ProcessCoordinator wrote onto the candidate.
    Missing the handoff is an error: looking the composition up locally would
    hide whether the coordinator actually translated the field.
    """

    def __init__(self, predictions: dict[str, Any], how: str, weight_sha256: str):
        super().__init__()
        self.module_name = "ExecutedToolScreener"
        # The functional architecture role is HT Predictor, but the trace must
        # distinguish released-model execution from the preceding table lookup.
        self.agent = "ExecutedToolScreener"
        self.predictions = predictions
        self.how = how
        self._source = f"executed: model.joblib @ {weight_sha256}"

    def get_supported_actions(self) -> list[str]:
        return ["predict_ionic_conductivity"]

    def predict_ionic_conductivity(
        self, parameters: dict[str, Any], candidate: CandidateRecord
    ) -> StageResult:
        received = _received(candidate, self.module_name)
        handoff = candidate.test_conditions.get("_handoff", {}).get(self.module_name, {})
        formula = (handoff.get("fields") or {}).get("formula")
        if not formula:
            raise RuntimeError(
                "ExecutedToolScreener requires ProcessCoordinator to hand off "
                "'formula'; refusing to read the composition locally"
            )
        prediction = self.predictions.get(formula)
        if prediction is None:
            raise KeyError(f"no executed prediction for handed-off formula {formula!r}")

        # The executed tool is not chosen by a selector at this step -- it was
        # fixed when the batch was predicted -- so the trace names it directly.
        # Before this override, the executor rows carried an empty selected_tool
        # while the upstream lookup rows carried "ML-SSE".
        reasoning = {
            "agent": self.agent,
            "selected_tool": "ML-SSE",
            "selected_repo": ml_sse.SOURCE.repository,
            "selected_crl": None,
            "decision_source": "executed released model (fixed at batch prediction)",
            "rationale": (
                "runs the released ML-SSE model.joblib on the handed-off formula; "
                "the executed value is this row's log10_sigma"
            ),
        }
        self._last_llm_call_json = ""
        metrics: dict[str, Any] = {
            "formula": formula,
            "log10_sigma": prediction.log10_sigma,
            "sigma_S_cm": prediction.sigma_S_cm,
            "above_convention_1e4": prediction.above_convention_1e4,
            "convention_log10_S_cm": CONVENTION_LOG10_S_CM,
            "convention_note": (
                "1e-4 S/cm is a field convention for superionic conductors, "
                "not a published ML-SSE or ML-HTVS criterion; not used as a gate"
            ),
            "ood_elements": list(prediction.ood_elements),
            "composition_level_only": True,
            "structure_invisible": True,
            "received_upstream_fields": received,
            "execution": True,
            "execution_how": self.how,
            "_metrics_source": self._source,
        }
        metrics.update(reasoning)
        return self._make_result(
            phase=Phase.SCREENING,
            passed=True,
            metrics=metrics,
            artifacts={"rationale": reasoning["rationale"], "llm_call": self._last_llm_call_json},
        )


class PublishedAimdFeedback(ExperimentFeedback):
    """Serves published AIMD trajectories as simulation-grade virtual experiments.

    The sixth architecture role, run without inventing a cell. LiSSE-MP
    (paper_id 10664) released DiffusionAnalyzer trajectories for nine
    AIMD-confirmed superionic conductors; this agent looks the handed-off
    formula up in that release. An exact stoichiometry match is a published
    virtual experiment for the survivor itself, a same-element-system match is
    family-level context, and no match is recorded as an explicit absence.
    Every row it returns carries ``feedback_grade`` naming what this is and is
    not: a simulation another group published, never a measurement.

    ``SimulationFeedback`` is a distinct trace label inside the Experiment
    Feedback functional role, the same convention ``ExecutedToolScreener``
    uses inside HT Predictor.
    """

    def __init__(self, index: Any, sigma_by_mpid: dict[str, list], sigma_how: str):
        super().__init__()
        self.module_name = "SimulationFeedback"
        self.agent = "SimulationFeedback"
        self.index = index
        self.sigma_by_mpid = sigma_by_mpid
        self.sigma_how = sigma_how
        self._source = f"published: {_pinned(lisse_mp.SOURCE)} AIMD trajectories"

    def get_supported_actions(self) -> list[str]:
        return ["simulation_feedback"]

    def simulation_feedback(
        self, parameters: dict[str, Any], candidate: CandidateRecord
    ) -> StageResult:
        handoff = candidate.test_conditions.get("_handoff", {}).get(self.module_name, {})
        fields = handoff.get("fields") or {}
        formula = fields.get("formula")
        if not formula:
            raise RuntimeError(
                "SimulationFeedback requires ProcessCoordinator to hand off "
                "'formula'; refusing to read the composition locally"
            )
        above = fields.get("executed_above_convention")

        exact = self.index.exact_match(formula)
        family = self.index.same_system(formula)
        sigma_points = self.sigma_by_mpid.get(exact.mp_id, []) if exact else []

        reasoning = self.select_tool_and_reason(
            task_desc=(
                "check a screened composition against published virtual experiments, "
                f"{formula}"
            ),
            candidate=candidate,
        )
        metrics: dict[str, Any] = {
            # Echoed so the next handoff can carry them without a local read.
            "formula": formula,
            "executed_log10_sigma": fields.get("executed_log10_sigma"),
            "executed_above_convention": above,
            # What this stage is and is not. The negative fact is scoped to what
            # the code can actually know: this run used no measurement. Whether a
            # cell was ever measured anywhere is not checkable here and is not
            # claimed.
            "feedback_grade": "simulation (published AIMD), not a measurement",
            "measurement_used_in_this_run": False,
            "virtual_experiment_source": lisse_mp.SOURCE.citation,
            "release_claim": self.index.release_claim,
            "n_aimd_structures_scanned": len(self.index.structures),
            # Lookup outcome.
            "aimd_exact_match": bool(exact),
            "aimd_exact_mp_id": exact.mp_id if exact else "",
            "aimd_family_mp_ids": ",".join(s.mp_id for s in family),
            "aimd_family_formulas": ",".join(s.formula for s in family),
            "no_published_simulation": not exact and not family,
            # The release labels its nine structures superionic from AIMD at
            # 800-1700 K; the executed composition-level estimate sits below the
            # 1e-4 S/cm convention. The two are not condition-aligned (different
            # temperatures, structure-specific vs composition-level), so this
            # flag marks evidence that requires an aligned reassessment. It is
            # not a verdict that either tool is wrong, and it only fires when
            # the release's numbers were actually parsed.
            "condition_mismatched_evidence": (
                bool(exact) and above is False and bool(sigma_points)
            ),
            "aimd_numbers_available": bool(sigma_points),
            "aimd_analysis_how": self.sigma_how if exact else "",
            "execution": False,
            "received_upstream_fields": list(fields.keys()),
            "_metrics_source": self._source if (exact or family) else (
                f"{self._source}: no structure shares this element system; "
                "the release stops here"
            ),
        }
        for point in sigma_points:
            metrics[f"aimd_sigma_mS_cm_{int(point.temperature_K)}K"] = point.conductivity_mS_cm
            metrics[f"aimd_diffusivity_cm2_s_{int(point.temperature_K)}K"] = point.diffusivity_cm2_s
        # The numbers depend on the analyzer implementation, so the environment
        # that produced them is part of the record.
        metrics["aimd_pymatgen_version"] = (
            f"pymatgen {sigma_points[0].pymatgen}; "
            f"pymatgen-analysis-diffusion {sigma_points[0].analysis_diffusion}"
            if sigma_points else ""
        )
        metrics["aimd_analyzer_module"] = sigma_points[0].analyzer if sigma_points else ""
        metrics.update(reasoning)

        candidate.test_conditions["_virtual_feedback"] = {
            key: value for key, value in metrics.items()
            if key.startswith(("aimd_", "condition_mismatched", "feedback_grade",
                               "release_claim"))
        }

        return self._make_result(
            phase=Phase.SIMULATION,
            passed=True,          # no gate: absence and disagreement are answers
            metrics=metrics,
            artifacts={
                "aimd_points_json": json.dumps(
                    [point.__dict__ for point in sigma_points], ensure_ascii=False
                ),
                "rationale": reasoning["rationale"],
                "llm_call": self._last_llm_call_json,
            },
        )


#: Every suggestion ``ShortlistFormulationDesigner`` can make, in firing order.
#: Exported next to the suggestions themselves: a reader who can see the whole
#: rule set can check that the agent chose among fixed options rather than
#: composing advice per candidate, which is the difference between a rule engine
#: and a language model with a formulation-shaped prompt.
SUGGESTION_RULES: list[dict[str, str]] = [
    {
        "rule_id": "provisional-rank",
        "condition": "the executed prediction used an element absent from the model's training formulae",
        "action": "treat this rank as provisional; do not compare it against in-domain "
                  "candidates without retraining on the missing element",
    },
    {
        "rule_id": "no-precedent",
        "condition": "the card cites no same-family study",
        "action": "no corpus precedent to lean on; enumerate structures and simulate "
                  "transport before spending any synthesis effort",
    },
    {
        "rule_id": "family-route",
        "condition": "the card cites at least one same-family study",
        "action": "reuse the cited family's characterisation route instead of designing one",
    },
    {
        "rule_id": "screen-failure-modes",
        "condition": "the card reports failure modes for this family",
        "action": "screen for the cited family failure modes before cell assembly",
    },
    {
        "rule_id": "structure-check",
        "condition": "the release enumerated no structure for this composition",
        "action": "run the upstream pipeline's own next stage, structure enumeration: "
                  "the executed model is composition-level and cannot see structure",
    },
    {
        "rule_id": "structure-reuse",
        "condition": "the release already enumerated structures for this composition",
        "action": "check the composition-level prediction against the released structures "
                  "with a structure-aware method",
    },
    {
        "rule_id": "convention-note",
        "condition": "the executed prediction sits above the 1e-4 S/cm field convention",
        "action": "report the convention as an annotation, never as a passed criterion: "
                  "neither upstream tool published it as a threshold",
    },
    {
        "rule_id": "condition-mismatch-recheck",
        "condition": "a published AIMD study covers this exact composition under "
                     "conditions the executed estimate does not share (high-temperature, "
                     "structure-specific vs composition-level)",
        "action": "re-assess under aligned conditions with a structure-aware method "
                  "before acting on either number",
    },
    {
        "rule_id": "family-simulated",
        "condition": "the release ran AIMD on same-system compositions but not on this one",
        "action": "reuse the published AIMD protocol for this stoichiometry before "
                  "considering synthesis",
    },
]


def _rule_fires(rule_id: str, card: dict, executed: dict, handoff: dict, feedback: dict) -> bool:
    return {
        "provisional-rank": lambda: bool(executed.get("ood_elements")),
        # Fires on the absence of family-precedent evidence, matching the rule's
        # condition text; the previous corpus_miss check could stay silent when
        # papers cleared the floor but none was a family study.
        "no-precedent": lambda: not card.get("family_precedent"),
        "family-route": lambda: bool(card.get("family_precedent")),
        "screen-failure-modes": lambda: bool(card.get("failure_modes")),
        "structure-check": lambda: not handoff.get("structure_data_published"),
        "structure-reuse": lambda: bool(handoff.get("structure_data_published")),
        "convention-note": lambda: bool(executed.get("above_convention_1e4")),
        "condition-mismatch-recheck": lambda: bool(
            feedback.get("condition_mismatched_evidence")),
        "family-simulated": lambda: (
            not feedback.get("aimd_exact_match")
            and bool(feedback.get("aimd_family_mp_ids"))
        ),
    }.get(rule_id, lambda: False)()


def _rule_citations(rule_id: str, card: dict) -> list[str]:
    # The two AIMD-driven rules cite the release that published the trajectories;
    # the key is in the project bibliography and passes the same whitelist as the
    # card citations.
    if rule_id in ("condition-mismatch-recheck", "family-simulated"):
        return [lisse_mp.SOURCE.citation]
    section = {
        "family-route": "family_precedent",
        "screen-failure-modes": "failure_modes",
    }.get(rule_id)
    if not section:
        return []
    cited = [row.get("citation", "") for row in card.get(section, []) if row.get("citation")]
    return list(dict.fromkeys(cited))


class ShortlistFormulationDesigner(FormulationDesigner):
    """Turns an executed prediction plus a literature card into next steps.

    A rule engine, and named as one. Every suggestion is an entry in
    ``SUGGESTION_RULES`` fired by a condition that is either a flag from the
    executed model or a count from the retrieved card; the agent proposes no
    composition, invents no threshold, and copies no number the upstream stages
    did not produce. What it contributes is the join: the literature the
    KnowledgeMiner retrieved is attached to the specific candidate the screening
    and execution stages ranked, with the citations kept.

    The generic ``recommend_formulations`` inherited from ``FormulationDesigner``
    reads the synthetic fixture and is not available here.
    """

    def __init__(self, source_note: str = ""):
        super().__init__()
        self._source = source_note or "rule-based over executed metrics and retrieved card"

    def get_supported_actions(self) -> list[str]:
        return ["suggest_next_steps"]

    def suggest_next_steps(
        self, parameters: dict[str, Any], candidate: CandidateRecord
    ) -> StageResult:
        handoff = candidate.test_conditions.get("_handoff", {}).get(self.module_name, {})
        fields = handoff.get("fields") or {}
        if "structures_published" not in fields:
            raise RuntimeError(
                "ShortlistFormulationDesigner requires ProcessCoordinator to hand off "
                "'structures_published'; refusing to summarise a candidate whose upstream "
                "stages did not reach it"
            )
        card = candidate.test_conditions.get("_knowledge_card") or {}
        executed = candidate.get_latest_metrics("ExecutedToolScreener")
        feedback = candidate.get_latest_metrics("SimulationFeedback")

        suggestions = [
            {
                "rule_id": rule["rule_id"],
                "condition": rule["condition"],
                "action": rule["action"],
                "citations": _rule_citations(rule["rule_id"], card),
            }
            for rule in SUGGESTION_RULES
            if _rule_fires(rule["rule_id"], card, executed, fields, feedback)
        ]
        citations = list(dict.fromkeys(c for s in suggestions for c in s["citations"]))

        reasoning = self.select_tool_and_reason(
            task_desc=(
                "recommend the next step for a screened lithium solid electrolyte, "
                f"{candidate.system_name}, given an executed conductivity prediction "
                "and the retrieved literature for its family"
            ),
            candidate=candidate,
        )
        metrics: dict[str, Any] = {
            "composition": candidate.system_name,
            "n_rules_available": len(SUGGESTION_RULES),
            "n_rules_fired": len(suggestions),
            "rules_fired": [s["rule_id"] for s in suggestions],
            "n_citations": len(citations),
            "citations": citations,
            "literature_family": fields.get("literature_family", ""),
            "structures_published": fields.get("structures_published"),
            "received_upstream_fields": list(fields.keys()),
            "proposes_new_composition": False,
            "invents_threshold": False,
            "execution": False,
            "_metrics_source": self._source,
        }
        metrics.update(reasoning)
        return self._make_result(
            phase=Phase.SCREENING,
            passed=bool(suggestions),
            metrics=metrics,
            artifacts={
                "suggestions_json": json.dumps(suggestions, ensure_ascii=False),
                "rationale": reasoning["rationale"],
                "llm_call": self._last_llm_call_json,
            },
        )
