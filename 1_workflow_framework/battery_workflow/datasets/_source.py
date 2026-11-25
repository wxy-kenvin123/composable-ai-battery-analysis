"""
Locate, verify, and read the published datasets that the trace replays consume.

The published-data replays in ``battery_workflow/examples`` consume files that were
published by *other* groups alongside their papers. Those files are deliberately
**not** vendored into this repository: two of the three source repositories carry
no licence at all, and maintaining extra copies would also invite data drift.
Instead each dataset is pinned to an upstream commit and a SHA-256 per file, so a
reader who fetches the data themselves can prove they are looking at the same
bytes used by the replay.

Resolution order for a dataset root (first hit wins):

1. the ``root=`` argument, if given;
2. the dataset's own environment variable (e.g. ``BATTAGENT_ML_HTVS_DIR``);
3. ``$BATTAGENT_REPOS_CACHE/<cache_dirname>``;
4. ``<repo>/data/external/<cache_dirname>``.

If none of those contains the required files, loading raises
:class:`MissingDatasetError` with the exact fetch command. Nothing is stubbed,
faked, or silently skipped -- a missing dataset must stop the run, because the
whole point of these cases is that the numbers are not ours to invent.

Stdlib-only, matching ``tool_registry.py``.
"""

from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

#: Historical name retained only so an old environment cannot silently change
#: validation semantics.  Published-data entry points always enforce SHA-256;
#: exploratory callers that knowingly want different bytes must opt out in code
#: with ``load(..., verify=False)`` rather than through ambient process state.
SKIP_CHECKSUM_ENV = "BATTAGENT_SKIP_CHECKSUM"

#: Shared parent directory holding one clone per audited tool.
REPOS_CACHE_ENV = "BATTAGENT_REPOS_CACHE"


class MissingDatasetError(FileNotFoundError):
    """The published data files could not be found on this machine."""


class DatasetIntegrityError(RuntimeError):
    """A data file was found but does not match the pinned checksum."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root() -> Path:
    # battery_workflow/datasets/_source.py -> repository root is three levels up.
    return Path(__file__).resolve().parents[2]


def to_float(value: Any) -> float | None:
    """Parse a CSV cell as a float, mapping blanks and non-numeric cells to None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("na", "nan", "none", "null"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


@dataclass(frozen=True)
class PublishedSource:
    """One upstream repository, pinned to a commit and a set of file checksums."""

    key: str
    tool: str
    paper_id: str
    citation: str
    doi: str
    repository: str
    commit: str
    licence: str
    redistributable: bool
    env_var: str
    cache_dirname: str
    #: repo-relative POSIX path -> expected SHA-256
    files: dict[str, str] = field(default_factory=dict)

    # ----- location -----
    def candidate_roots(self, root: str | Path | None = None) -> list[Path]:
        # An explicitly named root is authoritative: falling back to a different
        # directory would silently answer a question the caller did not ask.
        if root:
            return [Path(root)]
        roots: list[Path] = []
        env = os.environ.get(self.env_var)
        if env:
            roots.append(Path(env))
        cache = os.environ.get(REPOS_CACHE_ENV)
        if cache:
            roots.append(Path(cache) / self.cache_dirname)
        roots.append(_repo_root() / "data" / "external" / self.cache_dirname)
        return roots

    def root(self, root: str | Path | None = None) -> Path:
        probe = next(iter(self.files))
        for candidate in self.candidate_roots(root):
            if (candidate / probe).is_file():
                return candidate
        raise MissingDatasetError(self._not_found_message(root))

    def path(self, relative: str, root: str | Path | None = None) -> Path:
        if relative not in self.files:
            raise KeyError(f"{relative!r} is not a pinned file of source {self.key!r}")
        resolved = self.root(root) / relative
        if not resolved.is_file():
            raise MissingDatasetError(self._not_found_message(root))
        return resolved

    def _not_found_message(self, root: str | Path | None) -> str:
        tried = "\n".join(f"    {p}" for p in self.candidate_roots(root))
        wanted = "\n".join(f"    {name}" for name in self.files)
        return (
            f"Published dataset '{self.key}' ({self.tool}) was not found.\n"
            f"  Searched:\n{tried}\n"
            f"  Required files:\n{wanted}\n"
            f"  Fetch it with:\n"
            f"    git clone {self.repository} <dir>\n"
            f"    git -C <dir> checkout {self.commit}\n"
            f"  Then point {self.env_var} at <dir>, or set {REPOS_CACHE_ENV} to a\n"
            f"  directory containing a clone named '{self.cache_dirname}'.\n"
            f"  Source: {self.citation} (doi:{self.doi}); licence: {self.licence}."
        )

    # ----- integrity -----
    def verify(self, root: str | Path | None = None) -> dict[str, str]:
        """Check every pinned file; return {relative_path: observed_sha256}."""
        base = self.root(root)
        observed: dict[str, str] = {}
        mismatches: list[str] = []
        for relative, expected in self.files.items():
            target = base / relative
            if not target.is_file():
                raise MissingDatasetError(self._not_found_message(root))
            digest = _sha256(target)
            observed[relative] = digest
            if digest != expected:
                mismatches.append(f"    {relative}\n      expected {expected}\n      observed {digest}")
        if mismatches:
            raise DatasetIntegrityError(
                f"Published dataset '{self.key}' does not match the pinned commit "
                f"{self.commit}:\n" + "\n".join(mismatches) + "\n"
                "  Check out the pinned commit. Published trace replays do not "
                "permit an environment-variable checksum bypass."
            )
        return observed

    # ----- reading -----
    def read_csv(self, relative: str, root: str | Path | None = None) -> list[dict[str, str]]:
        with open(self.path(relative, root), encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def iter_csv(self, relative: str, root: str | Path | None = None) -> Iterator[dict[str, str]]:
        with open(self.path(relative, root), encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)

    # ----- provenance -----
    def provenance(self, root: str | Path | None = None) -> dict[str, Any]:
        """The block written into every exported run record for this source."""
        located: str | None
        try:
            located = str(self.root(root))
        except MissingDatasetError:
            located = None
        return {
            "key": self.key,
            "tool": self.tool,
            "audit_paper_id": self.paper_id,
            "citation": self.citation,
            "doi": self.doi,
            "repository": self.repository,
            "commit": self.commit,
            "licence": self.licence,
            "redistributable": self.redistributable,
            "files_sha256": dict(self.files),
            "resolved_root": located,
        }
