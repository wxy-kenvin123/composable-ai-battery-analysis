"""
Knowledge Miner agent (Fig. 12: "Knowledge Miner").

The ONLY end-to-end operational agent: it does not fabricate clues but calls the
authors' real battery PaperQA2 system through ``knowledge_backend`` (641 full-text
papers + 723 structured notes + an OpenAI-compatible LLM), returning structured, citation-backed
design clues (formulations, property targets, failure modes). When the index/key
is unavailable it transparently falls back to a deterministic stub so the demo
still runs.

Mapped audited tools (by main input -> main output): literature/document mining
(see tool_registry: KnowledgeMiner).
"""

from __future__ import annotations

import json
from typing import Any

from .base_module import BaseModule

from ..candidate_record import CandidateRecord, StageResult, Phase
from ..knowledge_backend import get_knowledge_backend


class KnowledgeMiner(BaseModule):
    """Retrieves cited design clues from literature/databases via BatteryQA."""

    def __init__(self, backend: Any = None, top_k: int = 6):
        super().__init__(
            module_name="KnowledgeMiner",
            system_prompt=(
                "You are a battery materials literature mining agent. Given a "
                "research objective, retrieve relevant formulations, property "
                "targets, and known failure modes from the literature, each with "
                "a citation."
            ),
        )
        self.backend = backend or get_knowledge_backend(top_k=top_k)

    def get_supported_actions(self) -> list[str]:
        return ["retrieve_design_clues", "search_failure_modes", "retrieve_literature_card"]

    def retrieve_literature_card(
        self,
        parameters: dict[str, Any],
        candidate: CandidateRecord,
    ) -> StageResult:
        """Retrieve one composition's literature card from the corpus.

        ``retrieve_design_clues`` asks what to formulate, which is the wrong
        question for a screening shortlist: an inorganic sulfide put through the
        design-clue templates comes back with polymer-electrolyte advice. This
        action asks the three questions a shortlist actually raises -- who has
        studied this family and how, what conductivity has been reported for it,
        how does it fail -- and returns them as a card.

        The composition arrives by ProcessCoordinator handoff rather than off the
        record. An upstream stage is what decides a composition is worth the
        retrieval cost, and reading the record locally would make the trace unable
        to show that the decision reached this agent.
        """
        handoff = candidate.test_conditions.get("_handoff", {}).get(self.module_name, {})
        fields = handoff.get("fields") or {}
        target = fields.get("retrieval_target")
        if not target:
            raise RuntimeError(
                "KnowledgeMiner.retrieve_literature_card requires ProcessCoordinator to "
                "hand off 'retrieval_target'; refusing to read the composition locally"
            )

        card = self.backend.literature_card(target)
        candidate.test_conditions["_knowledge_card"] = card

        return self._make_result(
            phase=Phase.SCREENING,
            passed=True,          # a corpus miss is an answer, not an agent failure
            metrics={
                "composition": target,
                "backend": card.get("backend", self.backend.name),
                "family": card.get("family", ""),
                "n_papers": card.get("n_papers", 0),
                "n_family_term_papers": card.get("n_family_term_papers", 0),
                "n_cited": card.get("n_cited", 0),
                "n_failure_modes": len(card.get("failure_modes", [])),
                "n_conductivity_notes": len(card.get("conductivity_notes", [])),
                "corpus_miss": card.get("corpus_miss", True),
                "relevance_floor": card.get("relevance_floor"),
                "hits_below_floor": card.get("hits_below_floor", 0),
                "citations": card.get("citations", []),
                "citation_audit": card.get("citation_audit", {}),
                "received_upstream_fields": list(fields.keys()),
                "execution": False,
                "retrieval": True,
                "text_is_model_generated": True,
                "_metrics_source": (
                    f"retrieved: BatteryQA backend '{card.get('backend', '')}' over the "
                    "authors' corpus; counts are retrieval facts, the card's prose is a "
                    "model summary of retrieved passages"
                ),
            },
            artifacts={"literature_card_json": json.dumps(card, ensure_ascii=False)},
        )

    def retrieve_design_clues(
        self,
        parameters: dict[str, Any],
        candidate: CandidateRecord,
    ) -> StageResult:
        query = parameters.get("query") or candidate.system_name
        clues = self.backend.retrieve_design_clues(query)

        # Make the structured clues available to downstream agents (FormulationDesigner).
        candidate.test_conditions.setdefault("_knowledge", {})
        candidate.test_conditions["_knowledge"] = {
            "formulations": clues.get("formulations", []),
            "property_targets": clues.get("property_targets", {}),
            "failure_modes": clues.get("failure_modes", []),
            "citations": clues.get("citations", []),
        }

        return self._make_result(
            phase=Phase.SCREENING,
            passed=True,
            metrics={
                "backend": clues.get("backend", self.backend.name),
                "n_formulations": len(clues.get("formulations", [])),
                "n_failure_modes": len(clues.get("failure_modes", [])),
                "n_citations": len(clues.get("citations", [])),
                "citations": clues.get("citations", []),
            },
            artifacts={"design_clues_json": json.dumps(clues, ensure_ascii=False)},
        )

    def search_failure_modes(
        self,
        parameters: dict[str, Any],
        candidate: CandidateRecord,
    ) -> StageResult:
        component = parameters.get("component") or candidate.system_name
        clues = self.backend.search_failure_modes(component)
        failure_modes = clues.get("failure_modes", [])
        return self._make_result(
            phase=Phase.FEEDBACK,
            passed=True,
            metrics={
                "backend": clues.get("backend", self.backend.name),
                "failure_modes_found": len(failure_modes),
                "citations": clues.get("citations", []),
            },
            artifacts={"failure_modes_json": json.dumps(failure_modes, ensure_ascii=False)},
        )
