#!/usr/bin/env python
"""Initialize test fixtures for CORGIAS smoke tests.

Run directly to create all fixtures, or import ensure_fixtures() from other scripts.

Usage:
    python script/test_init.py [-c CORES] [fixture ...]

Available fixtures:
    ML_result   ASR (ML)  on full OG table   — used by profiling_smoke_cli.py
    MP_result   ASR (MP)  on full OG table   — used by profiling_smoke_cli.py
    ML_sub      ASR (ML)  on random-10-col subset  — primary for asa secondary test
    ML_others   ASR (ML)  on remaining cols (--test 50)  — secondary for asa secondary test
    MP_sub      ASR (MP)  on random-10-col subset  — primary for sev secondary test
    MP_others   ASR (MP)  on remaining cols (--test 50)  — secondary for sev secondary test

OG table splits (og_sub.csv / og_others.csv) are written to test/ and reused across runs.
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
TEST = ROOT / "test"

OG_TABLE = SAMPLES / "archaea_COG_table99.csv"
TREE = SAMPLES / "archaea_hq90.tre"

OG_SUB = TEST / "og_sub.csv"
OG_OTHERS = TEST / "og_others.csv"
SUB_N = 10  # number of random columns in the subset

# fixture_num_tests: None means use the value passed to ensure_fixtures()
FIXTURES: dict[str, dict] = {
    "ML_result": {"method": "ML", "og_table": OG_TABLE,  "work_dir": TEST / "ML_result", "fixture_num_tests": None},
    "MP_result": {"method": "MP", "og_table": OG_TABLE,  "work_dir": TEST / "MP_result", "fixture_num_tests": None},
    "ML_sub":    {"method": "ML", "og_table": OG_SUB,    "work_dir": TEST / "ML_sub",    "fixture_num_tests": None},
    "ML_others": {"method": "ML", "og_table": OG_OTHERS, "work_dir": TEST / "ML_others", "fixture_num_tests": 50},
    "MP_sub":    {"method": "MP", "og_table": OG_SUB,    "work_dir": TEST / "MP_sub",    "fixture_num_tests": None},
    "MP_others": {"method": "MP", "og_table": OG_OTHERS, "work_dir": TEST / "MP_others", "fixture_num_tests": 50},
}

_SPLIT_DEPENDENT = {"ML_sub", "ML_others", "MP_sub", "MP_others"}


def ensure_og_splits(seed: int = 42) -> None:
    """Write og_sub.csv (random SUB_N cols) and og_others.csv to test/."""
    if OG_SUB.exists() and OG_OTHERS.exists():
        print(f"  og_sub / og_others: already exist, skipping")
        return
    TEST.mkdir(exist_ok=True)
    df = pl.read_csv(OG_TABLE)
    index_col = df.columns[0]
    og_cols = df.columns[1:]
    random.seed(seed)
    sub_cols = random.sample(og_cols, SUB_N)
    other_cols = [c for c in og_cols if c not in sub_cols]
    df.select([index_col] + sub_cols).write_csv(OG_SUB)
    df.select([index_col] + other_cols).write_csv(OG_OTHERS)
    print(f"  og_sub ({SUB_N} OGs) and og_others ({len(other_cols)} OGs): written to {TEST}/")


def _build_fixture(name: str, cores: int, num_tests: int) -> bool:
    cfg = FIXTURES[name]
    work_dir: Path = cfg["work_dir"]
    method: str = cfg["method"]
    og_table: Path = cfg["og_table"]
    n = cfg["fixture_num_tests"] if cfg["fixture_num_tests"] is not None else num_tests

    if work_dir.exists():
        print(f"  {name}: already exists, skipping")
        return True

    print(f"  {name}: running corgias asr (method={method}, --test {n}) ...")
    cmd = [
        "corgias", "asr",
        "-t", str(TREE),
        "-d", str(og_table),
        "--work_dir", str(work_dir),
        "-m", method,
        "-c", str(cores),
        "--test", str(n),
    ]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"  {name}: done")
        return True
    print(f"  {name}: FAILED (exit {result.returncode})")
    return False


def ensure_fixtures(fixtures: list[str] | None = None,
                    cores: int = 4, num_tests: int = 10) -> bool:
    """Ensure requested fixtures exist, creating them via corgias asr if needed.

    Args:
        fixtures: fixture names to ensure (default: all).
        cores: CPU cores passed to corgias asr.
        num_tests: OGs to process for fixtures without a per-fixture override.

    Returns:
        True if all fixtures are available.
    """
    if fixtures is None:
        fixtures = list(FIXTURES.keys())

    unknown = [f for f in fixtures if f not in FIXTURES]
    if unknown:
        print(f"Unknown fixture(s): {unknown}. Available: {list(FIXTURES.keys())}")
        return False

    if any(f in _SPLIT_DEPENDENT for f in fixtures):
        ensure_og_splits()

    TEST.mkdir(exist_ok=True)
    return all(_build_fixture(name, cores, num_tests) for name in fixtures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize CORGIAS test fixtures")
    parser.add_argument("-c", "--cores", default=4, type=int)
    parser.add_argument("-n", "--num_tests", default=10, type=int,
                        help="OGs to process for fixtures without a fixed --test value")
    parser.add_argument("fixtures", nargs="*", default=list(FIXTURES.keys()),
                        help="Fixtures to create (default: all)")
    args = parser.parse_args()

    print("Initializing test fixtures ...")
    ok = ensure_fixtures(args.fixtures, args.cores, args.num_tests)
    sys.exit(0 if ok else 1)
