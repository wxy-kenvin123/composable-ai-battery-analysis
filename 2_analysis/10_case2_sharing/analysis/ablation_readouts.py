"""Recompute and verify paired module effects in Supplementary Table 16.

By default, verifies every ``<variant>_trajectory.csv`` present in ``results/``
(requiring at least the main setting ``ablation_ne_conductivity_scale``) against
``results/ablation_readouts.csv`` and writes checked output under ``rerun/``.
Pass ``--trajectories`` with all five variant trajectories to recompute all five rows.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SEC = Path(__file__).resolve().parents[1]
RESULTS = SEC / "results"
VARIANTS = (
    "ablation_prior_mix",
    "ablation_prior_mix_no_concentration",
    "ablation_ne_conductivity_scale",
    "ablation_ei_concentration_scale",
    "ablation_ei_stokes_einstein_scale",
)
MODULES = {"KM": "full_shared_no_km", "MS": "full_shared_no_mechanism"}
SEEDS = range(300, 360)
BUDGET = range(1, 101)
BOOTSTRAP_SEED, RESAMPLES = 20260919, 10000


def estimate(differences: np.ndarray) -> str:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(differences), size=(RESAMPLES, len(differences)))
    lower, upper = np.percentile(differences[indices].mean(axis=1), [2.5, 97.5])
    return f"{differences.mean():+.4f} ({lower:+.4f} to {upper:+.4f})"


def trajectories(frame: pd.DataFrame, arm: str) -> np.ndarray:
    sub = frame.loc[frame["arm"] == arm]
    matrix = sub.pivot(index="seed", columns="budget_used", values="hv_regret_normalized").sort_index()
    if list(matrix.index) != list(SEEDS) or list(matrix.columns) != list(BUDGET) or matrix.isna().any().any():
        raise ValueError(f"{arm} requires all 60 seeds and evaluations 1 through 100")
    return matrix.to_numpy()


def readouts(directory: Path) -> pd.DataFrame:
    rows = []
    for name in VARIANTS:
        traj = directory / f"{name}_trajectory.csv"
        if not traj.exists():
            continue
        frame = pd.read_csv(traj, encoding="utf-8-sig")
        frame = frame.loc[frame["protocol"] == "p80"]
        full = trajectories(frame, "full_shared")
        row = {"variant": name}
        for module, arm in MODULES.items():
            diff = trajectories(frame, arm) - full
            row[f"{module}@1-25"] = estimate(diff[:, :25].mean(axis=1))
            row[f"{module}@25"] = estimate(diff[:, 24])
            row[f"{module}@100"] = estimate(diff[:, 99])
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No <variant>_trajectory.csv files found in {directory}")
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, default=RESULTS)
    parser.add_argument("--reference", type=Path, default=RESULTS / "ablation_readouts.csv")
    parser.add_argument("--output", type=Path, default=SEC / "rerun" / "ablation_readouts_checked.csv")
    args = parser.parse_args()
    result = readouts(args.trajectories)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8")
    print(result.to_string(index=False))
    if args.reference.exists():
        ref = pd.read_csv(args.reference, dtype=str).set_index("variant")
        got = result.astype(str).set_index("variant")
        if "ablation_ne_conductivity_scale" not in got.index:
            print("FAIL: main setting ablation_ne_conductivity_scale trajectory missing", file=sys.stderr)
            return 1
        for variant, row in got.iterrows():
            if variant not in ref.index or not row.equals(ref.loc[variant]):
                print(f"FAIL: {variant} does not match {args.reference}", file=sys.stderr)
                return 1
        print(f"PASS: {len(got)} variant row(s) match {args.reference}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
