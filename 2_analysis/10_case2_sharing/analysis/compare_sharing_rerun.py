"""Compare a sharing-conditions rerun with the distributed 60-seed results.

    python analysis/compare_sharing_rerun.py <directory-with-run_record.json>

Seed-level results, summary trajectories and ordered selection hashes must
agree with the distributed results. Aggregate tables and evaluation-level rounds
are also compared when complete/reference files are present.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


REFERENCE = Path(__file__).resolve().parents[1] / "results" / "sharing_conditions"


def record(directory: Path) -> dict:
    return json.loads((directory / "run_record.json").read_text(encoding="utf-8"))


def table(directory: Path, filename: str, campaigns: set, by_point: bool, subset: bool = False):
    with (directory / filename).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = {}
        for row in reader:
            campaign = (row["protocol"], row["arm"], int(row["seed"]))
            if campaign not in campaigns:
                if subset:
                    continue
                raise ValueError(f"{filename} contains a campaign absent from the run record: {campaign}")
            key = campaign + (int(row["budget_used"]),) if by_point else campaign
            if key in rows:
                raise ValueError(f"{filename} contains a duplicate key {key}")
            rows[key] = row
        return tuple(reader.fieldnames or ()), rows


def check(path: Path, reference: Path) -> None:
    rerun, archived = record(path), record(reference)
    arms, seeds, protocols = rerun["arms"], rerun["seeds"], rerun["protocols"]
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(archived["seeds"]):
        raise ValueError("unexpected seed range")
    if rerun["case"] != archived["case"] or arms != archived["arms"] or protocols != archived["protocols"]:
        raise ValueError("experiment, protocol or arm list differs")
    campaigns = {(p, a, s) for p in protocols for a in arms for s in seeds}
    n = rerun["campaign_total_budget"]
    if n != archived["campaign_total_budget"]:
        raise ValueError("campaign budget differs")

    for filename, by_point in (("seed_summary.csv", False), ("trajectory.csv", True)):
        headers, mine = table(path, filename, campaigns, by_point)
        ref_headers, expected = table(reference, filename, campaigns, by_point, subset=True)
        expected_keys = ({key + (i,) for key in campaigns for i in range(1, n + 1)}
                         if by_point else campaigns)
        if headers != ref_headers or set(mine) != expected_keys or set(expected) != expected_keys:
            raise ValueError(f"{filename} differs in columns or campaign coverage")
        for key in sorted(expected_keys):
            if mine[key] != expected[key]:
                col = next(c for c in headers if mine[key][c] != expected[key][c])
                raise ValueError(f"{filename}: {key}, {col}: {mine[key][col]!r} != {expected[key][col]!r}")
        print(f"{filename}: {len(mine)} rows, all cells identical")

    rounds_headers, rounds_mine = table(path, "rounds.csv", campaigns, by_point=True)
    expected_pts = {key + (i,) for key in campaigns for i in range(1, n + 1)}
    if set(rounds_mine) != expected_pts:
        raise ValueError("rounds.csv differs in campaign coverage")
    if (reference / "rounds.csv").exists():
        ref_headers, ref_rounds = table(reference, "rounds.csv", campaigns, by_point=True, subset=True)
        if rounds_headers != ref_headers or set(ref_rounds) != expected_pts:
            raise ValueError("rounds.csv differs in columns or campaign coverage")
        for key in sorted(expected_pts):
            if rounds_mine[key] != ref_rounds[key]:
                col = next(c for c in rounds_headers if rounds_mine[key][c] != ref_rounds[key][c])
                raise ValueError(f"rounds.csv: {key}, {col}: {rounds_mine[key][col]!r} != {ref_rounds[key][col]!r}")
        print(f"rounds.csv: {len(rounds_mine)} rows, all cells identical")

    selected = {f"{a}/{s}": [rounds_mine[(p, a, s, i)]["candidate_id"] for i in range(1, n + 1)]
                for p, a, s in campaigns}
    calculated = {key: hashlib.sha256(",".join(ids).encode("ascii")).hexdigest()
                  for key, ids in selected.items()}
    if calculated != rerun["selection_hashes"]:
        raise ValueError("rerun selection hashes differ from its own ordered evaluations")
    if any(archived["selection_hashes"].get(key) != value for key, value in calculated.items()):
        raise ValueError("selection hashes differ from the distributed run")
    if set(seeds) == set(archived["seeds"]):
        for name in ("arm_comparison.csv", "comparisons.csv"):
            with (path / name).open(encoding="utf-8-sig", newline="") as a, (
                    reference / name).open(encoding="utf-8-sig", newline="") as b:
                if list(csv.reader(a)) != list(csv.reader(b)):
                    raise ValueError(f"{name} differs for the complete seed set")
    print(f"PASS: {len(calculated)} campaign hashes and {len(campaigns) * n} ordered evaluations")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rerun", type=Path)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    try:
        check(args.rerun, args.reference)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")


if __name__ == "__main__":
    main()
