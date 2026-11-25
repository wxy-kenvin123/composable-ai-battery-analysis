"""Deserialize published DiffusionAnalyzer trajectories in a pymatgen interpreter.

Reads JSON from stdin: {"paths": ["...800_nteps_100.json", ...]}
Writes one JSON line to stdout: a list of rows, each

    {"path", "formula", "temperature_K", "diffusivity_cm2_s",
     "conductivity_mS_cm", "n_ion_steps", "analyzer", "pymatgen",
     "analysis_diffusion"}

The numbers are produced by the authors' own serialized analysis objects: the
class recomputes their MSD statistics from their stored displacements on
``from_dict``, which is the same operation their notebook performs on load.
This worker adds no smoothing choice, no fit, and no extrapolation of its own.
The environment that ran the deserialization is part of every row, because the
numbers depend on it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The shared trajectory decoder lives in the dataset module; the worker must
# not re-implement the unwrap.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from battery_workflow.datasets.lisse_mp import read_trajectory_dict  # noqa: E402


def _analyzer_class():
    try:
        from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer
        return DiffusionAnalyzer, "pymatgen.analysis.diffusion.analyzer"
    except ImportError:
        from pymatgen.analysis.diffusion_analyzer import DiffusionAnalyzer
        return DiffusionAnalyzer, "pymatgen.analysis.diffusion_analyzer"


def main() -> int:
    from importlib.metadata import PackageNotFoundError, version

    pymatgen_version = version("pymatgen")
    try:
        diffusion_version = version("pymatgen-analysis-diffusion")
    except PackageNotFoundError:
        diffusion_version = "builtin"

    payload = json.loads(sys.stdin.read())
    cls, module = _analyzer_class()

    rows = []
    for path in payload["paths"]:
        analyzer = cls.from_dict(read_trajectory_dict(Path(path)))
        rows.append({
            "path": path,
            "formula": analyzer.structure.composition.reduced_formula,
            "temperature_K": float(analyzer.temperature),
            "diffusivity_cm2_s": float(analyzer.diffusivity),
            "conductivity_mS_cm": float(analyzer.conductivity),
            "n_ion_steps": int(analyzer.disp.shape[1]),
            "analyzer": module,
            "pymatgen": pymatgen_version,
            "analysis_diffusion": diffusion_version,
        })

    print(json.dumps(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
