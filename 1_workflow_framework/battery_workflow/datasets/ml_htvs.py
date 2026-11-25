"""
ML-HTVS-Li-SE: the published solid-electrolyte screening funnel (audit paper_id 10771).

Kim, Mok, Kim & Back screened a substitution-generated chemical space down to a
DFT-validation shortlist with three successive filters, each implemented in its
own notebook and each writing its survivors to a CSV. This module reads those
CSVs and reassembles them into one candidate table so the workflow framework can
re-derive the funnel from the *inputs* rather than restate the published counts.

The upstream funnel, and what it means:

    02_6_e_hull_roost_results.csv    438  every candidate that reached screening
      -> 02_1 cell[30]: roost_ensemble_mean < 0.1 and e_hull_from_roost < 0.2
    02_7_after_e_hull_screening.csv  319  thermodynamically accessible
      -> 02_2 cell[8]:  pred_0 > 0.5            (renamed roost_band_gap)
    03_3_after_band_gap_screening.csv 306  electronically insulating
      -> 02_3 cell[7]:  upper_bound > 1 and total_range > 0.001
    04_2_after_stability_window_screening.csv 56  electrochemically usable

Two properties of this dataset make it worth replaying. First, every threshold is
a literal in the published notebooks, so the acceptance criteria are the authors'
and not ours. Second, the expensive properties were only ever computed for the
survivors of the previous filter -- there are exactly 319 band gaps and exactly
306 stability windows, not 438 of each. The published pipeline is therefore an
executed instance of the fail-fast, fail-cheap ordering, and the missing cells are evidence of the saving rather than missing data.

Nothing here recomputes ROOST predictions or convex hulls; the framework consumes
the published outputs. See ``NOT_REDISTRIBUTABLE`` before packaging any of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._source import PublishedSource, to_float

#: The upstream repository carries no LICENSE file (verified at the pinned
#: commit), so its data may be *read* by a reader who fetches it but may not be
#: redistributed with this repository or with any supplementary
#: material. The checksums below are what make the case auditable instead.
NOT_REDISTRIBUTABLE = True

SOURCE = PublishedSource(
    key="ml_htvs_li_se",
    tool="ML-HTVS-Li-SE",
    paper_id="10771",
    citation="[@kim2023accelerating]",
    doi="10.1021/acsami.3c10798",
    repository="https://github.com/SeoinBack/ML-HTVS-Li-SE",
    commit="c13da8e9c89ef81ee37eebe4ea3f0f7e778aedce",
    licence="none declared (do not redistribute)",
    redistributable=False,
    env_var="BATTAGENT_ML_HTVS_DIR",
    cache_dirname="gh_SeoinBack__ML-HTVS-Li-SE",
    files={
        "data/02_6_e_hull_roost_results.csv":
            "c19e5af9389a023b51379c5a8b7b64989daad6ef4cfdcea16b81a3f2bfb35010",
        "data/02_7_after_e_hull_screening.csv":
            "a5a8ed6d9f08bf4170f939124b9b6f602767725db81abf485594a8af1f960d7b",
        "data/03_2_band_gap_roost.csv":
            "5c881ff187d6ec36772d0c2d7624313019a5571540bd498ae236d5a28c699234",
        "data/03_3_after_band_gap_screening.csv":
            "e9c7f1c206ee06ca6a812cb9e375e6f504ca3c55a3a4f0330fd280e1e905dede",
        "data/04_1_stability_window.csv":
            "70ac28bdb31f598081cba2e88d9896cbeccb5e586cbae2e244de69c731a37ebd",
        "data/04_2_after_stability_window_screening.csv":
            "909aad5200a50ba5c1c2508f4c3ac156b221178cbcd0bf62c7dc4cae684c6f0c",
        "data/06_2_drop_duplicate.csv":
            "76a54c91bc426b6292303faed99605027b2a7f654c69f49a755032c1e1598a8e",
    },
)

#: Structures the authors enumerated for shortlist members before DFT. The
#: released file covers 7 of the 56 survivors, so most of the shortlist has no
#: published next-stage record at all -- which the framework reports rather than
#: fills in.
STRUCTURE_FILE = "data/06_2_drop_duplicate.csv"

#: Stage label -> (input file, survivor file). Used both to load the candidate
#: table and to check the re-derived funnel against the published survivor lists.
STAGE_FILES: dict[str, tuple[str, str]] = {
    "thermodynamic": ("data/02_6_e_hull_roost_results.csv",
                      "data/02_7_after_e_hull_screening.csv"),
    "electronic": ("data/03_2_band_gap_roost.csv",
                   "data/03_3_after_band_gap_screening.csv"),
    "electrochemical": ("data/04_1_stability_window.csv",
                        "data/04_2_after_stability_window_screening.csv"),
}


@dataclass
class HtvsCandidate:
    """One screened composition with whatever the published pipeline computed for it.

    Field names deliberately mirror the upstream CSV column names so that any
    number appearing in a gate decision can be grepped straight out of the
    published file. ``None`` means the property was never computed upstream,
    which only happens for candidates an earlier filter had already rejected.
    """

    composition: str
    roost_ensemble_mean: float | None = None   # ROOST formation energy, eV/atom
    e_hull_from_roost: float | None = None     # energy above hull, eV/atom
    roost_band_gap: float | None = None        # ROOST band gap, eV
    upper_bound: float | None = None           # anodic limit of the window, V
    total_range: float | None = None           # width of the window, V
    pretty_formula: str = ""

    @property
    def candidate_id(self) -> str:
        return f"htvs:{self.composition}"

    def metrics(self) -> dict[str, Any]:
        """The metric dict handed to a validation gate."""
        return {
            "roost_ensemble_mean": self.roost_ensemble_mean,
            "e_hull_from_roost": self.e_hull_from_roost,
            "roost_band_gap": self.roost_band_gap,
            "upper_bound": self.upper_bound,
            "total_range": self.total_range,
        }


@dataclass(frozen=True)
class EnumeratedStructure:
    """One symmetry-distinct structure the authors enumerated for a shortlist member."""

    composition: str
    spacegroup: str
    wyckoffs: str
    species: str


@dataclass
class HtvsDataset:
    """The 438 screened candidates plus the published survivor lists."""

    candidates: list[HtvsCandidate]
    published_survivors: dict[str, list[str]] = field(default_factory=dict)
    structures: dict[str, list[EnumeratedStructure]] = field(default_factory=dict)
    root: Path | None = None

    def __len__(self) -> int:
        return len(self.candidates)

    def published_counts(self) -> dict[str, int]:
        """Survivor count per stage, read from the upstream output files."""
        return {stage: len(ids) for stage, ids in self.published_survivors.items()}

    def by_composition(self) -> dict[str, HtvsCandidate]:
        return {c.composition: c for c in self.candidates}

    def structures_for(self, composition: str) -> list[EnumeratedStructure]:
        return self.structures.get(composition, [])


def load(root: str | Path | None = None, verify: bool = True) -> HtvsDataset:
    """Read the published funnel into a single candidate table.

    Raises ``MissingDatasetError`` if the data has not been fetched and
    ``DatasetIntegrityError`` if it does not match the pinned commit.
    """
    if verify:
        SOURCE.verify(root)
    base = SOURCE.root(root)

    candidates: dict[str, HtvsCandidate] = {}
    for row in SOURCE.read_csv("data/02_6_e_hull_roost_results.csv", base):
        composition = row["composition"].strip()
        candidates[composition] = HtvsCandidate(
            composition=composition,
            roost_ensemble_mean=to_float(row.get("roost_ensemble_mean")),
            e_hull_from_roost=to_float(row.get("e_hull_from_roost")),
        )

    # Band gaps exist only for the 319 that cleared the thermodynamic filter.
    for row in SOURCE.read_csv("data/03_2_band_gap_roost.csv", base):
        candidate = candidates.get(row["composition"].strip())
        if candidate is not None:
            candidate.roost_band_gap = to_float(row.get("pred_0"))

    # Stability windows exist only for the 306 that also cleared the band-gap filter.
    for row in SOURCE.read_csv("data/04_1_stability_window.csv", base):
        candidate = candidates.get(row["composition"].strip())
        if candidate is not None:
            candidate.upper_bound = to_float(row.get("upper_bound"))
            candidate.total_range = to_float(row.get("total_range"))
            candidate.pretty_formula = (row.get("pretty_formula") or "").strip()

    survivors = {
        stage: [r["composition"].strip() for r in SOURCE.read_csv(survivor_file, base)]
        for stage, (_, survivor_file) in STAGE_FILES.items()
    }

    structures: dict[str, list[EnumeratedStructure]] = {}
    for row in SOURCE.read_csv(STRUCTURE_FILE, base):
        composition = (row.get("composition") or "").strip()
        if not composition:
            continue
        structures.setdefault(composition, []).append(EnumeratedStructure(
            composition=composition,
            spacegroup=(row.get("spacegroup") or "").strip(),
            wyckoffs=(row.get("wyckoffs") or "").strip(),
            species=(row.get("species") or "").strip(),
        ))

    return HtvsDataset(
        candidates=list(candidates.values()),
        published_survivors=survivors,
        structures=structures,
        root=base,
    )
