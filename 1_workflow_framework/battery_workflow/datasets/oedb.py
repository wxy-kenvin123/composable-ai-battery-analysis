"""
The OEDB v2026-05-11 snapshot, split so that no agent can reach a label.

Cases A, B and C replay decisions someone else already made, so it did not matter
much which object held which column. Case D is a live search: the whole result
rests on the optimiser not having seen the answer. That turns "who may read the
labels" from a convention into the central design constraint, which is why this
module hands out three different objects instead of one:

    OEDBSearchSpace   1,872 candidate keys and their features. Constructed by
                      copying the key fields out; the label columns are never
                      stored on it, so an agent cannot reach them by accident or
                      by reflection.
    OEDBOracle        the only holder of labels. Reveals one candidate at a time
                      and remembers what it has revealed, so a repeat query is
                      visible rather than free.
    OEDBEvaluator     the true Pareto front and the hypervolume reference. Starts
                      disabled and refuses every accessor until the oracle has
                      been frozen, so end-of-run scoring cannot leak backwards
                      into the run.

Data is pinned exactly like the other published datasets: a
:class:`~battery_workflow.datasets._source.PublishedSource` with a SHA-256 per
file, resolved through ``BATTAGENT_OEDB_DIR`` or the shared cache, and never
vendored. It is not a git repository, so ``repository`` and ``commit`` carry the
MCP endpoint and the data version instead of a URL and a hash; the MCP specifics
are recorded in full in the run record.

Beyond checksums, :func:`load_oedb_snapshot` re-derives the structural facts the
frozen contract depends on -- row count, the complete 9 x 26 x 8 Cartesian
product, positivity of every objective, and the exact identity of the five rows
with no conductivity. A snapshot that fails any of them raises rather than
quietly changing what the benchmark means.

Stdlib only, matching ``_source.py``. Feature vectors come out as tuples; the
numpy conversion belongs to the surrogate, not to the data layer.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ._source import (
    DatasetIntegrityError,
    MissingDatasetError,
    PublishedSource,
    to_float,
)

#: battery_workflow/datasets/oedb.py -> repository root is three levels up.
_REPO_ROOT = Path(__file__).resolve().parents[2]

CONTRACT_PATH = _REPO_ROOT / "battery_workflow" / "case_d" / "configs" / "sharing_conditions.json"


def load_contract(path: Path | None = None) -> dict[str, Any]:
    """Read the frozen Case D contract.

    Everything configurable about Case D lives in that file, so the loader reads
    it rather than restating any of it. A missing contract is fatal: running the
    search against defaults invented here would bypass the frozen contract.
    """
    target = Path(path) if path else CONTRACT_PATH
    if not target.is_file():
        raise MissingDatasetError(
            f"The Case D contract is missing at {target}.\n"
            "  It is part of the package under battery_workflow/case_d/."
        )
    with open(target, encoding="utf-8") as handle:
        return json.load(handle)


def _source_from_contract(contract: dict[str, Any]) -> PublishedSource:
    data = contract["data"]
    return PublishedSource(
        key="oedb",
        tool="OEDB (Open Electrolyte Database for Batteries)",
        paper_id="n/a; OEDB post-dates the 68-tool audit",
        citation="[@nakamura2026oedb] Nakamura et al., npj Comput. Mater. (2026)",
        doi=data["paper_doi"],
        repository=data["mcp_endpoint"],
        commit=f"data {data['version']}",
        licence=data["licence"],
        redistributable=True,
        env_var=data["env_var"],
        cache_dirname=data["cache_dirname"],
        files=dict(data["files_sha256"]),
    )


@dataclass(frozen=True)
class Observation:
    """What the oracle returns for one revealed candidate.

    ``status`` is ``"ok"`` or ``"data_missing"``. For a ``data_missing`` point
    every objective field is ``None`` even though that row does have a viscosity
    and a diffusivity in the database. Serving the partial row would create a
    second kind of observation that the contract never defined, and the
    approved policy says these points do not enter the model at all.
    """

    candidate_id: int
    status: str
    m1: float | None
    m2: float | None
    m3: float | None
    d_cation: float | None
    d_anion: float | None
    concentration: float | None
    duplicate: bool = False

    @property
    def usable(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class _Record:
    """One Li row, labels included. Held only by the snapshot and the oracle."""

    id: int
    anion: str
    solvent: str
    concentration: float
    ionic_conductivity: float | None
    cation_diffusivity: float
    anion_diffusivity: float
    viscosity: float
    density: float

    @property
    def key(self) -> tuple[str, str, float]:
        return (self.anion, self.solvent, self.concentration)

    def minimize_form(self) -> tuple[float | None, float, float]:
        """(-log10 sigma, -log10 D_Li, +log10 eta); the first is None when blank."""
        m1 = None if self.ionic_conductivity is None else -math.log10(self.ionic_conductivity)
        return (m1, -math.log10(self.cation_diffusivity), math.log10(self.viscosity))

    @property
    def transference(self) -> float:


        return self.cation_diffusivity / (self.cation_diffusivity + self.anion_diffusivity)

    def minimize_form_2obj(self) -> tuple[float | None, float]:


        m1 = None if self.ionic_conductivity is None else -math.log10(self.ionic_conductivity)
        return (m1, -self.transference)


class OEDBSearchSpace:
    """The 1,872 candidate keys, with no label anywhere on the object."""

    def __init__(self, contract: dict[str, Any], records: Iterable[_Record]) -> None:
        universe = contract["universe"]
        self._anions: tuple[str, ...] = tuple(universe["anions_ordered"])
        self._solvents: tuple[str, ...] = tuple(universe["solvents_ordered"])
        scaling = contract["features"]["concentration_scaling"]
        self._c_min = float(scaling["min"])
        self._c_max = float(scaling["max"])
        self._dimension = int(contract["features"]["dimension"])

        # Copy out only the key fields. The records themselves are not retained,
        # so there is no attribute path from a search space to a label.
        keys: dict[int, tuple[str, str, float]] = {}
        for record in records:
            keys[record.id] = record.key
        self._keys = keys
        self._ids: tuple[int, ...] = tuple(sorted(keys))
        self._anion_index = {name: i for i, name in enumerate(self._anions)}
        self._solvent_index = {name: i for i, name in enumerate(self._solvents)}
        self._features = {i: self._encode(keys[i]) for i in self._ids}

    def _encode(self, key: tuple[str, str, float]) -> tuple[float, ...]:
        anion, solvent, concentration = key
        vector = [0.0] * self._dimension
        vector[self._anion_index[anion]] = 1.0
        vector[len(self._anions) + self._solvent_index[solvent]] = 1.0
        span = self._c_max - self._c_min
        vector[-1] = (concentration - self._c_min) / span
        return tuple(vector)

    @property
    def ids(self) -> tuple[int, ...]:
        return self._ids

    @property
    def dimension(self) -> int:
        return self._dimension

    def __len__(self) -> int:
        return len(self._ids)

    def key(self, candidate_id: int) -> tuple[str, str, float]:
        return self._keys[candidate_id]

    def concentration(self, candidate_id: int) -> float:
        return self._keys[candidate_id][2]

    def anion(self, candidate_id: int) -> str:
        return self._keys[candidate_id][0]

    def features(self, candidate_id: int) -> tuple[float, ...]:
        return self._features[candidate_id]

    def feature_matrix(self, ids: Iterable[int] | None = None) -> tuple[tuple[float, ...], ...]:
        chosen = self._ids if ids is None else tuple(ids)
        return tuple(self._features[i] for i in chosen)

    def unseen_ids(self, observed: Iterable[int]) -> tuple[int, ...]:
        seen = set(observed)
        return tuple(i for i in self._ids if i not in seen)


class OEDBOracle:
    """The only object that can turn a candidate id into labels."""

    def __init__(self, contract: dict[str, Any], records: Iterable[_Record]) -> None:
        self._records = {record.id: record for record in records}
        self._missing_ids = frozenset(contract["missing_policy"]["missing_ids"])
        self._revealed: list[int] = []
        self._revealed_set: set[int] = set()
        self._missing_queries: dict[int, int] = {}
        self._duplicate_queries = 0
        self._frozen = False

    def reveal(self, candidate_id: int) -> Observation:
        if self._frozen:
            raise RuntimeError(
                "OEDBOracle.reveal() was called after the campaign froze the oracle. "
                "Scoring has already started; revealing more would make the reported "
                "budget wrong."
            )
        record = self._records.get(candidate_id)
        if record is None:
            raise KeyError(f"{candidate_id!r} is not a candidate in the Li subspace")

        duplicate = candidate_id in self._revealed_set
        if duplicate:
            self._duplicate_queries += 1
        else:
            self._revealed.append(candidate_id)
            self._revealed_set.add(candidate_id)

        if candidate_id in self._missing_ids:
            self._missing_queries[candidate_id] = self._missing_queries.get(candidate_id, 0) + 1
            return Observation(
                candidate_id=candidate_id,
                status="data_missing",
                m1=None,
                m2=None,
                m3=None,
                d_cation=None,
                d_anion=None,
                concentration=None,
                duplicate=duplicate,
            )

        m1, m2, m3 = record.minimize_form()
        return Observation(
            candidate_id=candidate_id,
            status="ok",
            m1=m1,
            m2=m2,
            m3=m3,
            d_cation=record.cation_diffusivity,
            d_anion=record.anion_diffusivity,
            concentration=record.concentration,
            duplicate=duplicate,
        )

    def freeze(self) -> None:
        self._frozen = True

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def revealed_ids(self) -> tuple[int, ...]:
        return tuple(self._revealed)

    @property
    def reveal_count(self) -> int:
        return len(self._revealed)

    @property
    def duplicate_queries(self) -> int:
        return self._duplicate_queries

    @property
    def missing_queries(self) -> dict[int, int]:
        return dict(self._missing_queries)


class OEDBEvaluator:
    """End-of-run scoring against the full label set. Disabled until unlocked.

    The unlock takes the oracle rather than a flag, and refuses unless that
    oracle is frozen. Enabling the evaluator therefore cannot be done half way
    through a campaign without first closing the campaign, which is the property
    the leakage argument needs.
    """

    _LOCKED = (
        "OEDBEvaluator is disabled. It exposes the full label set, so it may only "
        "be enabled after the campaign has finished and the oracle has been frozen: "
        "call enable(oracle)."
    )

    def __init__(self, contract: dict[str, Any], records: Iterable[_Record]) -> None:
        self._contract = contract
        self._records = tuple(records)
        self._enabled = False

    def enable(self, oracle: OEDBOracle) -> None:
        if not isinstance(oracle, OEDBOracle):
            raise TypeError("enable() takes the campaign's OEDBOracle")
        if not oracle.is_frozen:
            raise RuntimeError(
                "Refusing to enable the evaluator while the oracle is still open. "
                "Freeze the oracle first; a run that can still reveal points must "
                "not be able to read the answer key."
            )
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _require(self) -> None:
        if not self._enabled:
            raise RuntimeError(self._LOCKED)

    def truth_rows(self) -> tuple[tuple[int, float, float, float], ...]:
        """(id, m1, m2, m3) for the fully-labelled points only."""
        self._require()
        rows = []
        for record in self._records:
            m1, m2, m3 = record.minimize_form()
            if m1 is None:
                continue
            rows.append((record.id, m1, m2, m3))
        return tuple(sorted(rows))

    def truth_rows_2obj(self) -> tuple[tuple[int, float, float], ...]:


        self._require()
        rows = []
        for record in self._records:
            m1, m4 = record.minimize_form_2obj()
            if m1 is None:
                continue
            rows.append((record.id, m1, m4))
        return tuple(sorted(rows))

    @property
    def frozen_constants(self) -> dict[str, Any]:
        self._require()
        return dict(self._contract["metrics"]["evaluator_only"])


@dataclass
class OEDBSnapshot:
    """A verified OEDB Li subspace, plus the three capability-separated views."""

    contract: dict[str, Any]
    source: PublishedSource
    observed_sha256: dict[str, str]
    resolved_root: str
    _records: tuple[_Record, ...]
    total_rows: int
    checks: dict[str, Any]

    def search_space(self) -> OEDBSearchSpace:
        return OEDBSearchSpace(self.contract, self._records)

    def oracle(self) -> OEDBOracle:
        return OEDBOracle(self.contract, self._records)

    def evaluator(self) -> OEDBEvaluator:
        return OEDBEvaluator(self.contract, self._records)

    def provenance(self) -> dict[str, Any]:
        block = self.source.provenance()
        data = self.contract["data"]
        block["access_method"] = {
            "protocol": "Model Context Protocol",
            "endpoint": data["mcp_endpoint"],
            "server_name": data["mcp_server_name"],
            "server_version": data["mcp_server_version"],
            "protocol_version": data["mcp_protocol_version"],
            "tool": data["export_tool"],
            "filters": data["export_filters"],
            "reported_row_count": data["export_reported_row_count"],
            "note": (
                "OEDB is not a git repository. 'repository' carries the MCP endpoint "
                "and 'commit' the published data version."
            ),
        }
        block["structural_checks"] = dict(self.checks)
        return block

    def summary(self) -> str:
        checks = self.checks
        return (
            f"OEDB {self.contract['data']['version']} ({self.contract['data']['licence']})\n"
            f"  resolved root      : {self.resolved_root}\n"
            f"  rows in export     : {self.total_rows}\n"
            f"  Li candidates      : {checks['li_rows']}"
            f" = {checks['n_anions']} anions x {checks['n_solvents']} solvents"
            f" x {checks['n_concentrations']} concentrations\n"
            f"  duplicate keys     : {checks['duplicate_keys']}\n"
            f"  Cartesian complete : {checks['cartesian_complete']}\n"
            f"  conductivity       : {checks['conductivity_valid']} valid,"
            f" {checks['conductivity_missing']} missing,"
            f" {checks['nonpositive_total']} non-positive across all objectives\n"
            f"  data_missing ids   : {checks['missing_ids']}"
        )


def _fail(message: str) -> None:
    raise DatasetIntegrityError(
        "OEDB snapshot does not match the frozen contract:\n  " + message
    )


def load_oedb_snapshot(
    root: str | Path | None = None,
    verify: bool = True,
    contract: dict[str, Any] | None = None,
) -> OEDBSnapshot:
    """Load, checksum and structurally re-verify the OEDB Li subspace.

    ``verify=False`` skips only the SHA-256 comparison, and is there for the
    integrity test that needs to reach a deliberately corrupted file. The
    structural checks below always run: they are what stops a silently different
    database from redefining the benchmark under a passing test suite.
    """
    contract = contract or load_contract()
    source = _source_from_contract(contract)
    observed = source.verify(root) if verify else {}
    resolved = source.root(root)

    cols = contract["columns"]
    csv_name = contract["data"]["primary_file"]
    with open(source.path(csv_name, root), encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    expected_columns = set(cols.values())
    present = set(rows[0].keys()) if rows else set()
    missing_columns = sorted(expected_columns - present)
    if missing_columns:
        _fail(f"columns absent from {csv_name}: {missing_columns}")

    total_rows = len(rows)
    if total_rows != contract["data"]["export_reported_row_count"]:
        _fail(
            f"export has {total_rows} rows, contract pins "
            f"{contract['data']['export_reported_row_count']}"
        )

    universe = contract["universe"]
    wanted_cation = universe["cation"]
    records: list[_Record] = []
    for row in rows:
        if row[cols["cation"]] != wanted_cation:
            continue
        records.append(
            _Record(
                id=int(row[cols["id"]]),
                anion=row[cols["anion"]],
                solvent=row[cols["solvent"]],
                concentration=float(row[cols["concentration"]]),
                ionic_conductivity=to_float(row[cols["ionic_conductivity"]]),
                cation_diffusivity=float(row[cols["cation_diffusivity"]]),
                anion_diffusivity=float(row[cols["anion_diffusivity"]]),
                viscosity=float(row[cols["viscosity"]]),
                density=float(row[cols["density"]]),
            )
        )
    records.sort(key=lambda r: r.id)

    if len(records) != universe["n_candidates"]:
        _fail(
            f"{wanted_cation} subspace has {len(records)} rows, contract pins "
            f"{universe['n_candidates']}"
        )

    keys = [record.key for record in records]
    duplicate_keys = len(keys) - len(set(keys))
    if duplicate_keys:
        _fail(f"{duplicate_keys} duplicate formulation key(s) in the Li subspace")

    anions = sorted({record.anion for record in records})
    solvents = sorted({record.solvent for record in records})
    concentrations = sorted({record.concentration for record in records})
    if anions != sorted(universe["anions_ordered"]):
        _fail(f"anion set differs from the contract: {anions}")
    if solvents != sorted(universe["solvents_ordered"]):
        _fail(f"solvent set differs from the contract: {solvents}")
    if concentrations != sorted(float(c) for c in universe["concentrations_ordered"]):
        _fail(f"concentration grid differs from the contract: {concentrations}")

    expected_cells = {
        (a, s, c)
        for a in universe["anions_ordered"]
        for s in universe["solvents_ordered"]
        for c in (float(v) for v in universe["concentrations_ordered"])
    }
    observed_cells = set(keys)
    absent = expected_cells - observed_cells
    extra = observed_cells - expected_cells
    if absent or extra:
        _fail(
            f"Cartesian product incomplete: {len(absent)} cell(s) absent, "
            f"{len(extra)} unexpected"
        )

    nonpositive = 0
    for record in records:
        values = [
            record.cation_diffusivity,
            record.anion_diffusivity,
            record.viscosity,
            record.density,
        ]
        if record.ionic_conductivity is not None:
            values.append(record.ionic_conductivity)
        nonpositive += sum(1 for v in values if v <= 0)
    if nonpositive:
        _fail(
            f"{nonpositive} non-positive value(s) among the objective columns; "
            "log10 would be undefined and the contract asserts there are none"
        )

    observed_missing = sorted(
        record.id for record in records if record.ionic_conductivity is None
    )
    pinned_missing = sorted(contract["missing_policy"]["missing_ids"])
    if observed_missing != pinned_missing:
        _fail(
            "the set of rows with no ionic conductivity is not the pinned "
            f"one.\n      contract: {pinned_missing}\n      observed: {observed_missing}"
        )

    valid = len(records) - len(observed_missing)
    if valid != contract["missing_policy"]["truth_set_size"]:
        _fail(
            f"{valid} fully-labelled candidates, contract pins "
            f"{contract['missing_policy']['truth_set_size']}"
        )

    checks = {
        "li_rows": len(records),
        "n_anions": len(anions),
        "n_solvents": len(solvents),
        "n_concentrations": len(concentrations),
        "duplicate_keys": duplicate_keys,
        "cartesian_complete": True,
        "conductivity_valid": valid,
        "conductivity_missing": len(observed_missing),
        "nonpositive_total": nonpositive,
        "missing_ids": observed_missing,
        "checksums_verified": bool(verify),
    }

    return OEDBSnapshot(
        contract=contract,
        source=source,
        observed_sha256=observed,
        resolved_root=str(resolved),
        _records=tuple(records),
        total_rows=total_rows,
        checks=checks,
    )
