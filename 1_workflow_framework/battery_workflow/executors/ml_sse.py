"""Execute the released ML-SSE conductivity model on composition strings.

The shipped ``model.joblib`` is a ``TECRF`` ensemble of scikit-learn
RandomForest regressors pickled without the later ``missing_go_to_left`` tree
field. sklearn 1.8 cannot unpickle it; sklearn 1.0.2 can, and reproduces the
authors' ``new_formula_prediction.csv`` to numerical noise (Task 0 probe).

This module therefore tries an in-process load and, on that dtype error, runs
``ml_sse_worker.py`` under ``BATTAGENT_ML_SSE_PYTHON`` (default: the
current interpreter). One call predicts a whole batch so the 21 MB
forest is loaded once.

The prediction is log10 of ionic conductivity in S/cm, matching the upstream
notebook. It is a composition-level model output, not a measured value, and it
cannot see crystal structure.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER = Path(__file__).resolve().parent / "ml_sse_worker.py"
_DEFAULT_FALLBACK = Path(sys.executable)

CONVENTION_LOG10_S_CM = -4.0  # 1e-4 S/cm; convention, not a published criterion


@dataclass(frozen=True)
class MlSsePrediction:
    formula: str
    log10_sigma: float | None
    sigma_S_cm: float | None
    above_convention_1e4: bool | None
    ood_elements: tuple[str, ...]
    executed: bool
    error: str = ""


def fallback_python() -> Path:
    env = os.environ.get("BATTAGENT_ML_SSE_PYTHON")
    return Path(env) if env else _DEFAULT_FALLBACK


def _predict_inprocess(repo: Path, formulas: list[str]) -> list[float]:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    previous = Path.cwd()
    os.chdir(repo)
    try:
        import joblib
        import pandas as pd

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(repo / "model.joblib")
            dummy = pd.DataFrame({"formula": formulas, "target": [0.0] * len(formulas)})
            return [float(x) for x in model.predict(dummy)]
    finally:
        os.chdir(previous)


def _predict_subprocess(repo: Path, formulas: list[str]) -> list[float]:
    python = fallback_python()
    if not python.is_file():
        raise FileNotFoundError(
            f"ML-SSE fallback interpreter not found: {python}. "
            "Set BATTAGENT_ML_SSE_PYTHON to a sklearn 1.0.x python."
        )
    payload = json.dumps({"repo": str(repo), "formulas": formulas})
    proc = subprocess.run(
        [str(python), "-X", "utf8", str(_WORKER)],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(_REPO_ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "ML-SSE worker failed "
            f"(rc={proc.returncode}): {(proc.stderr or proc.stdout)[-500:]}"
        )
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("[")][-1]
    values = json.loads(line)
    if len(values) != len(formulas):
        raise RuntimeError(
            f"ML-SSE worker returned {len(values)} predictions for {len(formulas)} formulae"
        )
    return [float(x) for x in values]


def predict_log10(repo: Path, formulas: list[str]) -> tuple[list[float], str]:
    """Return (log10 sigma values, how they were produced)."""
    if not formulas:
        return [], "empty"
    try:
        return _predict_inprocess(repo, formulas), f"inprocess:{sys.executable}"
    except ValueError as exc:
        message = str(exc)
        if "incompatible dtype" not in message and "missing_go_to_left" not in message:
            raise
        return _predict_subprocess(repo, formulas), f"subprocess:{fallback_python()}"


def predict_formulas(
    repo: Path,
    formulas: list[str],
    training_elements: set[str] | None = None,
) -> tuple[list[MlSsePrediction], str]:
    values, how = predict_log10(repo, formulas)
    rows: list[MlSsePrediction] = []
    for formula, log10_sigma in zip(formulas, values):
        ood = ()
        if training_elements is not None:
            ood = tuple(sorted(_elements(formula) - training_elements))
        sigma = 10 ** log10_sigma
        rows.append(MlSsePrediction(
            formula=formula,
            log10_sigma=log10_sigma,
            sigma_S_cm=sigma,
            above_convention_1e4=log10_sigma >= CONVENTION_LOG10_S_CM,
            ood_elements=ood,
            executed=True,
        ))
    return rows, how


def _elements(formula: str) -> set[str]:
    import re
    return set(re.findall(r"[A-Z][a-z]?", formula))
