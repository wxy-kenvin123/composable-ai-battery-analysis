"""
Knowledge backend for the KnowledgeMiner agent.

Bridges the (public) workflow framework to the authors' battery PaperQA2 system
(``battery_qa.py`` in the authors' local workspace, not released), which is the only
*operational* knowledge engine among the six agents. Two backends:

  - ``BatteryQABackend``  : calls the real BatteryQA engine out-of-process via
    ``BATTAGENT_PAPERQA_DIR`` (the folder that holds ``battery_qa.py`` and its index) and
    ``BATTAGENT_PAPERQA_PYTHON`` (an interpreter with paperqa/openai installed).
    Returns structured, citation-backed clues ({formulations, property_targets,
    failure_modes, evidence, citations}).
  - ``InProcessBatteryQABackend`` : the same engine imported into the current
    interpreter, so the index is loaded once instead of once per query. For
    benchmarks; requires running inside the environment that has the index.
  - ``StubBackend``       : deterministic, dependency-free fallback used when the
    index / key / engine is unavailable (e.g. a user without the 5.5 GB
    index). Returns the same structured schema so the demo still runs and the
    validation-gate decisions are identical.

The public repository ships only this bridge + the stub; it never redistributes
the 5.5 GB index, the 723 structured notes, or any API key.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Composition-aware query construction
#
# Kept on this side of the bridge on purpose. The engine holds the corpus and
# the API key and is not distributed; if it also built the questions, a reader
# of this repository could see the retrieved answers but not what was asked.
# Everything below is a deterministic string transform over a chemical formula,
# so the queries in an exported card can be re-derived from the shortlist alone.
# --------------------------------------------------------------------------- #

#: Element symbol -> the word a paper is likely to use for it.
ELEMENT_NAMES = {
    "Li": "lithium", "Na": "sodium", "K": "potassium", "Mg": "magnesium",
    "Ca": "calcium", "Sr": "strontium", "Ba": "barium", "B": "boron",
    "Al": "aluminium", "Ga": "gallium", "In": "indium", "Tl": "thallium",
    "C": "carbon", "Si": "silicon", "Ge": "germanium", "Sn": "tin", "Pb": "lead",
    "N": "nitrogen", "P": "phosphorus", "As": "arsenic", "Sb": "antimony",
    "Bi": "bismuth", "O": "oxygen", "S": "sulfur", "Se": "selenium",
    "Te": "tellurium", "F": "fluorine", "Cl": "chlorine", "Br": "bromine",
    "I": "iodine", "H": "hydrogen", "Ti": "titanium", "V": "vanadium",
    "Cr": "chromium", "Mn": "manganese", "Fe": "iron", "Co": "cobalt",
    "Ni": "nickel", "Cu": "copper", "Zn": "zinc", "Cd": "cadmium",
    "Hg": "mercury", "Ag": "silver", "Y": "yttrium", "Zr": "zirconium",
    "Nb": "niobium", "Mo": "molybdenum", "La": "lanthanum", "Hf": "hafnium",
    "Ta": "tantalum", "W": "tungsten", "Sc": "scandium",
}

#: Elements a paper names in anion form when they sit on the anion site.
ANION_FORMS = {
    "S": "sulfide", "O": "oxide", "N": "nitride", "Se": "selenide",
    "Te": "telluride", "F": "fluoride", "Cl": "chloride", "Br": "bromide",
    "I": "iodide", "H": "hydride",
}

_HALOGENS = frozenset({"F", "Cl", "Br", "I"})

#: Ordered anion-set rules -> (family label, search terms). First match wins.
#: A label is a naming convention for the element set, not a phase assignment:
#: nothing here has seen a crystal structure.
_FAMILY_RULES = [
    (lambda e: "S" in e and "P" in e and e & _HALOGENS,
     "argyrodite-type lithium thiophosphate halide",
     ["argyrodite", "thiophosphate", "sulfide", "Li6PS5"]),
    (lambda e: "S" in e and "P" in e,
     "lithium thiophosphate (thio-LISICON, Li-P-S)",
     ["thiophosphate", "thio-LISICON", "sulfide", "LGPS", "Li2S-P2S5"]),
    (lambda e: "S" in e and e & _HALOGENS,
     "lithium sulfide-halide",
     ["sulfide", "halide", "argyrodite", "chloride"]),
    (lambda e: "S" in e and "O" in e,
     "lithium oxysulfide",
     ["oxysulfide", "sulfide", "oxide"]),
    (lambda e: "S" in e,
     "lithium metal sulfide",
     ["sulfide", "sulphide", "chalcogenide"]),
    (lambda e: "O" in e and "P" in e,
     "lithium phosphate (NASICON-type oxide)",
     ["NASICON", "phosphate", "oxide", "LATP"]),
    (lambda e: "O" in e,
     "lithium oxide (garnet or perovskite-type)",
     ["garnet", "LLZO", "oxide", "perovskite"]),
    (lambda e: bool(e & _HALOGENS),
     "lithium halide",
     ["halide", "chloride", "bromide", "iodide"]),
    (lambda e: "N" in e,
     "lithium nitride",
     ["nitride", "LiPON"]),
    (lambda e: "H" in e,
     "lithium complex hydride",
     ["hydride", "borohydride"]),
]


def parse_elements(formula: str) -> list[str]:
    """Element symbols in the order the formula writes them, deduplicated."""
    found = re.findall(r"([A-Z][a-z]?)\d*\.?\d*", formula or "")
    return list(dict.fromkeys(found))


def composition_context(formula: str) -> dict:
    """Turn one formula into the family label and the three questions to ask.

    The questions are the three a screening shortlist actually raises about a
    composition nobody has measured: has this family been studied and how, what
    conductivity has been reported for it, and how does it fail. The generic
    design-clue template asks what to formulate instead, which for an inorganic
    sulfide returns polymer-electrolyte advice.

    The questions are family-level, not stoichiometry-level, because the corpus
    is indexed by paper rather than by composition: two compositions with the
    same element set retrieve the same papers, and the difference between them
    has to come from the reader, not from the retrieval.
    """
    elements = parse_elements(formula)
    element_set = set(elements)
    words = [
        ANION_FORMS[e] if e in ANION_FORMS else ELEMENT_NAMES.get(e, e.lower())
        for e in elements
    ]
    phrase = " ".join(words)

    family, terms = "lithium solid electrolyte (unclassified anion set)", ["solid electrolyte"]
    for matches, label, search_terms in _FAMILY_RULES:
        if matches(element_set):
            family, terms = label, search_terms
            break

    return {
        "formula": formula,
        "elements": elements,
        "phrase": phrase,
        "family": family,
        "family_terms": terms,
        "queries": [
            f"{phrase} {family} solid electrolyte: which studies report this "
            f"material family and what computational or experimental method did they use",
            f"{phrase} {family} ionic conductivity values reported and the "
            f"structural or compositional factors that control them",
            f"{phrase} {family} degradation, moisture sensitivity, interfacial "
            f"reaction and phase stability failure modes",
        ],
    }


# Canonical empty/structured envelope shared by both backends.
def _envelope(query: str, backend: str, **kw: Any) -> dict:
    out = {
        "query": query,
        "backend": backend,
        "formulations": kw.get("formulations", []),
        "property_targets": kw.get("property_targets", {}),
        "failure_modes": kw.get("failure_modes", []),
        "evidence": kw.get("evidence", []),
        "citations": kw.get("citations", []),
    }
    if "available" in kw:
        out["available"] = kw["available"]
    if "error" in kw and kw["error"]:
        out["error"] = kw["error"]
    return out


def _card_envelope(composition: str, backend: str, context: dict, **kw: Any) -> dict:
    """Empty literature card, in the shape ``electrolyte_context`` returns."""
    out = {
        "composition": composition,
        "family": context.get("family", ""),
        "queries": [{"query": q, "n_hits": 0, "n_above_floor": 0, "best_score": 0.0}
                    for q in context.get("queries", [])],
        "backend": backend,
        "papers": [],
        "n_papers": 0,
        "family_precedent": [],
        "conductivity_notes": [],
        "failure_modes": [],
        "corpus_verdict": "",
        "citations": [],
        "n_cited": 0,
        "corpus_miss": True,
        "miss_note": "",
        "citation_audit": {"proposed": 0, "kept": 0, "dropped": 0, "dropped_keys": []},
    }
    out.update(kw)
    return out


class KnowledgeBackend(ABC):
    name: str = "base"

    @abstractmethod
    def retrieve_design_clues(self, query: str) -> dict: ...

    @abstractmethod
    def search_failure_modes(self, component: str) -> dict: ...

    def literature_card(self, composition: str) -> dict:
        """One composition's literature card, or an error envelope.

        Not abstract: a backend that cannot answer composition-level questions
        should say so in the returned record rather than fail to exist.
        """
        context = composition_context(composition)
        return _card_envelope(
            composition, "unsupported", context,
            error=f"{self.name} does not implement literature_card",
        )


class StubBackend(KnowledgeBackend):
    """
    Deterministic fallback. Returns curated, structured PEO/NMC811-relevant clues
    so the workflow demo runs identically with or without the real engine.
    Citations use the ``[@stub]`` marker to make the fallback unmistakable.
    """

    name = "stub"

    def retrieve_design_clues(self, query: str) -> dict:
        return _envelope(
            query,
            "stub",
            formulations=[
                {"composition": "PEO/LiTFSI (EO:Li = 16:1)", "citation": "[@stub]"},
                {"composition": "PEO/LiTFSI (EO:Li = 20:1) + 5 wt% LLZO filler", "citation": "[@stub]"},
                {"composition": "Cross-linked PEO/LiBOB (EO:Li = 12:1)", "citation": "[@stub]"},
            ],
            property_targets={
                "ionic_conductivity": {"target": ">= 1e-4 S/cm at 60 C", "citation": "[@stub]"},
                "electrochemical_window": {"target": ">= 4.5 V vs Li/Li+", "citation": "[@stub]"},
            },
            failure_modes=[
                {
                    "mode": "PEO oxidative decomposition at the high-voltage cathode",
                    "interface_side": "cathode",
                    "evidence": "CEI growth on NMC811 surface in PEO systems above ~4.0 V",
                    "citation": "[@stub]",
                },
                {
                    "mode": "Lithium dendrite penetration at high current density",
                    "interface_side": "anode",
                    "evidence": "PEO insufficient to block dendrites at elevated rates",
                    "citation": "[@stub]",
                },
            ],
            citations=["[@stub]"],
            available={"notes": False, "paperqa": False, "llm": False},
        )

    def search_failure_modes(self, component: str) -> dict:
        clues = self.retrieve_design_clues(component)
        return _envelope(component, "stub", failure_modes=clues["failure_modes"],
                         citations=clues["citations"], available={"notes": False, "paperqa": False, "llm": False})

    def literature_card(self, composition: str) -> dict:
        """A card marked as a stub; there is no corpus behind it.

        The composition-derived family and queries are still real -- they are
        computed in this file -- so a user without the index can check the
        question construction even though the answers are absent.
        """
        context = composition_context(composition)
        return _card_envelope(
            composition, "stub", context,
            miss_note="stub backend: no corpus is bundled with this repository",
        )


class BatteryQABackend(KnowledgeBackend):
    """Out-of-process bridge to the authors' BatteryQA engine (not released)."""

    name = "batteryqa"

    def __init__(self, paperqa_dir: str, python_exe: str, top_k: int = 6, timeout: int = 240):
        self.paperqa_dir = Path(paperqa_dir)
        self.python_exe = python_exe
        self.top_k = top_k
        self.timeout = timeout
        self.script = self.paperqa_dir / "battery_qa.py"
        self._stub = StubBackend()  # automatic fallback on error

    def _run(self, command: str, text: str, extra: list[str] | None = None) -> dict:
        cmd = [self.python_exe, str(self.script), command, text, "--top_k", str(self.top_k)]
        cmd.extend(extra or [])
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.paperqa_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout,
                env=env,
            )
        except Exception as exc:  # noqa: BLE001
            return _envelope(text, "error", error=f"BatteryQA subprocess failed: {exc}")
        if proc.returncode != 0:
            return _envelope(text, "error", error=(proc.stderr or "nonzero exit").strip()[:500])
        try:
            return json.loads(proc.stdout)
        except Exception as exc:  # noqa: BLE001
            return _envelope(text, "error", error=f"BatteryQA JSON parse failed: {exc}")

    def _with_fallback(self, result: dict, stub_fn, arg: str) -> dict:
        """If the real engine errored, transparently fall back to the deterministic stub."""
        if result.get("backend") == "error":
            fb = stub_fn(arg)
            fb["backend"] = "stub-fallback"
            fb["error"] = result.get("error")
            return fb
        return result

    def retrieve_design_clues(self, query: str) -> dict:
        return self._with_fallback(self._run("design_clues", query), self._stub.retrieve_design_clues, query)

    def search_failure_modes(self, component: str) -> dict:
        return self._with_fallback(self._run("failure_modes", component), self._stub.search_failure_modes, component)

    def literature_card(self, composition: str) -> dict:
        """Ask the engine the three composition-aware questions built here.

        Deliberately without the stub fallback the other two methods use: a card
        is evidence about a specific composition, and stub text carrying a real
        formula's name is exactly the artifact that must never reach an export.
        """
        context = composition_context(composition)
        extra = ["--family", context["family"]]
        for query in context["queries"]:
            extra += ["--query", query]
        for term in context["family_terms"]:
            extra += ["--family-term", term]
        result = self._run("electrolyte_context", composition, extra)
        if result.get("backend") == "error":
            return _card_envelope(composition, "error", context, error=result.get("error"))
        result["family_terms"] = context["family_terms"]
        return result

    def calibrate_relevance(self) -> dict:
        """The engine's floor calibration, or an error record."""
        return self._run("calibrate", "")


class InProcessBatteryQABackend(KnowledgeBackend):
    """The same engine, imported rather than shelled out to.

    ``BatteryQABackend`` starts a fresh interpreter per query, which reloads the
    723-note index and reconnects the clients every time -- about sixteen seconds
    of the thirty-five a query takes. That is irrelevant for a workflow run, which
    asks one or two questions, and prohibitive for a benchmark that asks a
    hundred, so this variant holds one engine open.

    It requires ``battery_qa`` to be importable in the *current* interpreter,
    which means running inside the environment that has paperqa and the index.
    ``get_knowledge_backend`` therefore does not return it by default; ask for it
    explicitly.
    """

    name = "batteryqa-inprocess"

    def __init__(self, paperqa_dir: str, top_k: int = 6):
        import sys

        directory = str(Path(paperqa_dir))
        if directory not in sys.path:
            sys.path.insert(0, directory)
        # The engine resolves its index and .bib relative to its own file, but
        # paperqa_config and docs_index.pkl are found relative to the cwd.
        self._cwd = directory
        from battery_qa import BatteryQA  # noqa: PLC0415  (optional dependency)

        self._engine = BatteryQA(top_k=top_k)
        self._stub = StubBackend()

    @property
    def available(self) -> dict:
        return self._engine.available

    def _guard(self, fn, arg: str, stub_fn) -> dict:
        cwd = os.getcwd()
        try:
            os.chdir(self._cwd)
            return fn(arg)
        except Exception as exc:  # noqa: BLE001
            fb = stub_fn(arg)
            fb["backend"] = "stub-fallback"
            fb["error"] = f"{type(exc).__name__}: {exc}"
            return fb
        finally:
            os.chdir(cwd)

    def retrieve_design_clues(self, query: str) -> dict:
        return self._guard(self._engine.retrieve_design_clues, query,
                           self._stub.retrieve_design_clues)

    def search_failure_modes(self, component: str) -> dict:
        return self._guard(self._engine.search_failure_modes, component,
                           self._stub.search_failure_modes)

    def literature_card(self, composition: str) -> dict:
        """Same three questions as the subprocess bridge, one loaded index."""
        context = composition_context(composition)
        cwd = os.getcwd()
        try:
            os.chdir(self._cwd)
            card = self._engine.electrolyte_context(
                composition,
                family=context["family"],
                queries=context["queries"],
                family_terms=context["family_terms"],
            )
        except Exception as exc:  # noqa: BLE001
            return _card_envelope(composition, "error", context,
                                  error=f"{type(exc).__name__}: {exc}")
        finally:
            os.chdir(cwd)
        card["family_terms"] = context["family_terms"]
        return card

    def calibrate_relevance(self) -> dict:
        cwd = os.getcwd()
        try:
            os.chdir(self._cwd)
            return self._engine.calibrate_relevance()
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            os.chdir(cwd)


def _default_python() -> str:
    explicit = os.environ.get("BATTAGENT_PAPERQA_PYTHON")
    if explicit:
        return explicit
    # Falls back to whatever is on PATH.
    return "python"


def get_knowledge_backend(top_k: int = 6, in_process: bool = False) -> KnowledgeBackend:
    """
    Returns the real BatteryQA backend when ``BATTAGENT_PAPERQA_DIR`` points to a
    directory containing ``battery_qa.py``; otherwise the deterministic stub.

    ``in_process=True`` imports the engine instead of shelling out, which is worth
    it only when many queries are asked in one session; it falls back to the
    subprocess bridge if the import fails.
    """
    paperqa_dir = os.environ.get("BATTAGENT_PAPERQA_DIR")
    if not (paperqa_dir and (Path(paperqa_dir) / "battery_qa.py").exists()):
        return StubBackend()
    if in_process:
        try:
            return InProcessBatteryQABackend(paperqa_dir, top_k=top_k)
        except Exception:  # noqa: BLE001  (wrong interpreter, missing paperqa, ...)
            pass
    return BatteryQABackend(paperqa_dir, _default_python(), top_k=top_k)


if __name__ == "__main__":
    import sys
    backend = get_knowledge_backend()
    q = " ".join(sys.argv[1:]) or "PEO/LiTFSI solid polymer electrolyte for 4.5V NMC811"
    print(f"backend = {backend.name}")
    result = backend.retrieve_design_clues(q)
    print(json.dumps(result, ensure_ascii=False, indent=2)[:1500])
