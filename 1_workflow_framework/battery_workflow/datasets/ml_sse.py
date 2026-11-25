"""
ML-SSE: the released composition -> log10 ionic-conductivity model (audit paper_id 10102).

Used only as the *second* tool on the Case A shortlist. The model is executed;
its predictions are not published values for those 56 compositions (the authors
never scored this particular list). Training-set coverage of halogenated
sulfides and integer-ratio parsing were confirmed in Task 0.
"""

from __future__ import annotations

import re
from pathlib import Path

from ._source import PublishedSource

NOT_REDISTRIBUTABLE = True

SOURCE = PublishedSource(
    key="ml_sse",
    tool="ML-SSE",
    paper_id="10102",
    citation="[@kurniawan2025unveiling]",
    doi="10.1016/j.jpcs.2025.112752",
    repository="https://github.com/muhamad-kurniawan/ML-SSE",
    commit="ba397ca4d154e66fc306ecc0e394cf343337fc3f",
    licence="none declared (do not redistribute)",
    redistributable=False,
    env_var="BATTAGENT_ML_SSE_DIR",
    cache_dirname="gh_muhamad-kurniawan__ML-SSE",
    files={
        "model.joblib":
            "e1b7665ebc578e41f3974d276c11887ead6221cb362209e9eb3676e85235de0f",
        "data/element_properties/features.csv":
            "e2744db9e2c451eace8756d57fcda5640673bd71272d670c68c837baee7ebda3",
        "new_formula_prediction.csv":
            "6e64131e21668a9f588b5429b1e5913b4ff462c348e37f8c4f836506dc0cd0da",
        "data/preprocessed_dataset/5fold_cv/1/train.csv":
            "d4a0ade0e00b12030351b0a967327e49788948dea6a71a7ac1c6d27c40a6d483",
    },
)

_ELEMENT = re.compile(r"[A-Z][a-z]?")


def training_elements(root: str | Path | None = None) -> set[str]:
    """Element symbols that appear in fold-1 training formulae."""
    SOURCE.verify(root)
    seen: set[str] = set()
    for row in SOURCE.read_csv("data/preprocessed_dataset/5fold_cv/1/train.csv", root):
        seen.update(_ELEMENT.findall(row.get("formula") or ""))
    return seen
