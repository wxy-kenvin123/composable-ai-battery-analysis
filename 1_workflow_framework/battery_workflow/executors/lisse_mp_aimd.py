"""Serve the AIMD conductivities stored in LiSSE-MP's released analysis objects.

The release ships each trajectory as a serialized pymatgen ``DiffusionAnalyzer``.
Deserializing one recomputes the authors' MSD statistics from their stored
displacements -- their analysis, on their data. This module tries the running
interpreter first and falls back to ``lisse_mp_worker.py`` under
``BATTAGENT_PYMATGEN_PYTHON`` (default: the repo-local ``.venv_pmg``), the same
two-step pattern the ML-SSE executor uses for its sklearn pin.

Failure contract (single, uniform): whatever goes wrong -- no pymatgen in any
interpreter, a missing fallback executable, a worker crash, malformed worker
output, or a timeout -- the caller receives ``([], reason)``. The reason string
travels into the export (``aimd_analysis_how``) and the feedback stage sets
``aimd_numbers_available=False``, so a degraded run is visible in the record
rather than silent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER = Path(__file__).resolve().parent / "lisse_mp_worker.py"
_DEFAULT_FALLBACK = _REPO_ROOT / ".venv_pmg" / "Scripts" / "python.exe"

#: Deserializing 4 trajectories takes seconds; minutes means a hung interpreter.
_WORKER_TIMEOUT_S = 180

_ROW_KEYS = {
    "path", "formula", "temperature_K", "diffusivity_cm2_s",
    "conductivity_mS_cm", "n_ion_steps", "analyzer", "pymatgen",
    "analysis_diffusion",
}


@dataclass(frozen=True)
class AimdPoint:
    """One published trajectory's transport numbers at one temperature."""

    path: str
    formula: str
    temperature_K: float
    diffusivity_cm2_s: float
    conductivity_mS_cm: float
    n_ion_steps: int
    analyzer: str
    pymatgen: str
    analysis_diffusion: str


def fallback_python() -> Path:
    env = os.environ.get("BATTAGENT_PYMATGEN_PYTHON")
    return Path(env) if env else _DEFAULT_FALLBACK


def _analyze_inprocess(paths: list[Path]) -> list[dict]:
    import importlib
    from importlib.metadata import PackageNotFoundError, version

    importlib.import_module("pymatgen")
    from battery_workflow.datasets.lisse_mp import read_trajectory_dict

    from . import lisse_mp_worker

    cls, module = lisse_mp_worker._analyzer_class()
    pymatgen_version = version("pymatgen")
    try:
        diffusion_version = version("pymatgen-analysis-diffusion")
    except PackageNotFoundError:
        diffusion_version = "builtin"

    rows = []
    for path in paths:
        analyzer = cls.from_dict(read_trajectory_dict(path))
        rows.append({
            "path": str(path),
            "formula": analyzer.structure.composition.reduced_formula,
            "temperature_K": float(analyzer.temperature),
            "diffusivity_cm2_s": float(analyzer.diffusivity),
            "conductivity_mS_cm": float(analyzer.conductivity),
            "n_ion_steps": int(analyzer.disp.shape[1]),
            "analyzer": module,
            "pymatgen": pymatgen_version,
            "analysis_diffusion": diffusion_version,
        })
    return rows


def _analyze_subprocess(paths: list[Path]) -> list[dict]:
    python = fallback_python()
    if not python.is_file():
        raise FileNotFoundError(
            f"pymatgen interpreter not found: {python}. Set BATTAGENT_PYMATGEN_PYTHON "
            "to a python with pymatgen and pymatgen-analysis-diffusion installed."
        )
    payload = json.dumps({"paths": [str(p) for p in paths]})
    proc = subprocess.run(
        [str(python), "-X", "utf8", str(_WORKER)],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(_REPO_ROOT),
        timeout=_WORKER_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "lisse_mp worker failed "
            f"(rc={proc.returncode}): {(proc.stderr or proc.stdout)[-500:]}"
        )
    json_lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("[")]
    if not json_lines:
        raise RuntimeError(
            f"worker produced no JSON row list; stdout tail: {proc.stdout[-300:]!r}"
        )
    rows = json.loads(json_lines[-1])
    if len(rows) != len(paths):
        raise RuntimeError(f"worker returned {len(rows)} rows for {len(paths)} trajectories")
    for row in rows:
        missing = _ROW_KEYS - set(row)
        if missing:
            raise RuntimeError(f"worker row missing keys: {sorted(missing)}")
    return rows


def analyze_trajectories(paths: list[Path]) -> tuple[list[AimdPoint], str]:
    """Return ([AimdPoint...], how). Any failure returns ([], 'unavailable: reason')."""
    if not paths:
        return [], "empty"
    try:
        return [AimdPoint(**row) for row in _analyze_inprocess(paths)], (
            f"inprocess:{sys.executable}"
        )
    except ImportError:
        pass
    except Exception as exc:  # decoding or analyzer failure in-process
        return [], f"unavailable: {exc}"
    try:
        rows = _analyze_subprocess(paths)
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired,
            json.JSONDecodeError, TypeError) as exc:
        return [], f"unavailable: {exc}"
    return [AimdPoint(**row) for row in rows], f"subprocess:{fallback_python()}"
