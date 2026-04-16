#!/usr/bin/env python
"""Initialize test fixtures for CORGIAS smoke tests.

Run directly to create all fixtures, or import ensure_fixtures() from other scripts.

Usage:
    python script/test_init.py [-c CORES] [-n NUM_TESTS] [fixture ...]

Available fixtures:
    ML_result   ASR output with ML method (required by asa / asa_secondary)
    MP_result   ASR output with MP method (required by sev / sev_secondary)
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
TEST = ROOT / "test"

OG_TABLE = SAMPLES / "archaea_COG_table99.csv"
TREE = SAMPLES / "archaea_hq90.tre"

FIXTURES = {
    "ML_result": {"method": "ML", "work_dir": TEST / "ML_result"},
    "MP_result": {"method": "MP", "work_dir": TEST / "MP_result"},
}


def _build_fixture(name: str, method: str, work_dir: Path,
                   cores: int, num_tests: int) -> bool:
    if work_dir.exists():
        print(f"  {name}: already exists, skipping")
        return True
    print(f"  {name}: running corgias asr (method={method}, --test {num_tests}) ...")
    cmd = [
        "corgias", "asr",
        "-t", str(TREE),
        "-d", str(OG_TABLE),
        "--work_dir", str(work_dir),
        "-m", method,
        "-c", str(cores),
        "--test", str(num_tests),
    ]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"  {name}: done")
        return True
    else:
        print(f"  {name}: FAILED (exit {result.returncode})")
        return False


def ensure_fixtures(fixtures: list[str] | None = None,
                    cores: int = 4, num_tests: int = 10) -> bool:
    """Create fixtures that do not yet exist.

    Args:
        fixtures: list of fixture names to ensure (default: all).
        cores: number of CPU cores passed to corgias asr.
        num_tests: number of OGs processed (--test); must be >= the value
                   used in downstream smoke tests.

    Returns:
        True if all requested fixtures are available.
    """
    if fixtures is None:
        fixtures = list(FIXTURES.keys())

    unknown = [f for f in fixtures if f not in FIXTURES]
    if unknown:
        print(f"Unknown fixture(s): {unknown}. Available: {list(FIXTURES.keys())}")
        return False

    TEST.mkdir(exist_ok=True)
    return all(
        _build_fixture(name, **FIXTURES[name], cores=cores, num_tests=num_tests)
        for name in fixtures
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize CORGIAS test fixtures")
    parser.add_argument("-c", "--cores", default=4, type=int)
    parser.add_argument("-n", "--num_tests", default=10, type=int,
                        help="OGs to process per fixture (--test value)")
    parser.add_argument("fixtures", nargs="*", default=list(FIXTURES.keys()),
                        help="Fixtures to create (default: all)")
    args = parser.parse_args()

    print("Initializing test fixtures ...")
    ok = ensure_fixtures(args.fixtures, args.cores, args.num_tests)
    sys.exit(0 if ok else 1)
