"""Fetch the upstream repositories that Case 1 reads, at the pinned commits, and verify every pinned file.

The three repositories are not redistributed with this package: ML-HTVS-Li-SE, LiSSE-MP and ML-SSE declare
no licence. Each is cloned from its public repository, checked out at the commit the framework pins, and
every file the framework reads is checked against its SHA-256 (battery_workflow/datasets/*.py).
Clones are made with core.autocrlf=false: the checksums are those of the files as stored in the repositories, and
git's line-ending conversion on Windows (core.autocrlf=true) would change them.

Usage
-----
    python tools/fetch_external_data.py --cache DIR            # clone from GitHub (needs git and network)
    python tools/fetch_external_data.py --cache DIR --verify   # only verify an existing cache
    python tools/fetch_external_data.py --cache DIR --mirror M # clone from local copies M/<cache_dirname>

Then set BATTAGENT_REPOS_CACHE=DIR before running reproduce_case1.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from battery_workflow.datasets import lisse_mp, ml_htvs, ml_sse  # noqa: E402

SOURCES = [ml_htvs.SOURCE, lisse_mp.SOURCE, ml_sse.SOURCE]


def git(*args: str) -> None:
    subprocess.run(["git", *args], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True, help="directory that will hold one clone per repository")
    ap.add_argument("--verify", action="store_true", help="do not clone; only verify the pinned files")
    ap.add_argument("--mirror", type=Path, help="clone from local copies instead of the public URLs")
    a = ap.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    failed = 0
    for src in SOURCES:
        dest = a.cache / src.cache_dirname
        print(f"\n{src.tool}: {src.repository} @ {src.commit} (licence: {src.licence})")
        if not a.verify:
            try:
                if dest.exists():
                    print(f"  {dest} exists; checking out the pinned commit")
                else:
                    url = str(a.mirror / src.cache_dirname) if a.mirror else src.repository
                    git("clone", "--quiet", "--config", "core.autocrlf=false", url, str(dest))
                git("-C", str(dest), "checkout", "--quiet", src.commit)
            except subprocess.CalledProcessError as exc:
                failed += 1
                print(f"  FAILED: git exited with {exc.returncode} ({' '.join(exc.cmd[1:3])} ...)")
                continue
        try:
            observed = src.verify(dest)
            print(f"  verified {len(observed)} pinned files")
        except Exception as exc:  # noqa: BLE001 - report every source before failing
            failed += 1
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            print("  If this clone was made with core.autocrlf=true, delete it and run this script again.")
    print("\nall pinned files verified" if not failed else f"\n{failed} source(s) failed verification")
    print(f"set BATTAGENT_REPOS_CACHE={a.cache.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
