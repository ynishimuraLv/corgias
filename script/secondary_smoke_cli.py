#!/usr/bin/env python
"""Smoke test for secondary dataset mode (-og2 / -a2) of corgias profiling."""

import math
import subprocess
import sys
import tempfile
import argparse
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
TEST = ROOT / "test"

sys.path.insert(0, str(Path(__file__).parent))
from test_init import ensure_fixtures

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_dir', default=None,
                    help='Output directory (default: temp dir)')
parser.add_argument('-c', '--cores', default=4, type=int)
parser.add_argument('-n', '--num_tests', default=5, type=int,
                    help='Number of OGs per dataset (--test value)')
args = parser.parse_args()

N = args.num_tests  # OGs per dataset

# ── Ensure test fixtures exist ─────────────────────────────────────────────
print("Checking test fixtures ...")
if not ensure_fixtures(['ML_result', 'MP_result'], cores=args.cores, num_tests=N * 2):
    sys.exit(1)
print()

# ── Prepare split OG tables ────────────────────────────────────────────────
og_table = SAMPLES / "archaea_COG_table99.csv"
tree = SAMPLES / "archaea_hq90.tre"
asr_ml = TEST / "ML_result"
asr_mp = TEST / "MP_result"

output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="corgias_secondary_"))
output_dir.mkdir(parents=True, exist_ok=True)

df = pl.read_csv(og_table)
# index col + first N OG cols → primary; index col + next N OG cols → secondary
primary_path = output_dir / "og_primary.csv"
secondary_path = output_dir / "og_secondary.csv"
df[:, :N + 1].write_csv(primary_path)
df[:, [0] + list(range(N + 1, N * 2 + 1))].write_csv(secondary_path)

print(f"Output dir : {output_dir}")
print(f"OGs per dataset: {N}")
print()

# ── Expected row counts ────────────────────────────────────────────────────
# cross = N * N,  secondary allvall = C(N, 2)
expected_rows = N * N + math.comb(N, 2)

# ── Helpers ────────────────────────────────────────────────────────────────
results = []

def run_case(name, cmd, out_path=None, expected=None):
    cmd = [str(x) for x in cmd]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if expected is not None and out_path is not None:
            actual = len(pl.read_csv(out_path))
            if actual != expected:
                results.append((name, f"FAIL (rows: expected {expected}, got {actual})"))
                return
        results.append((name, "OK"))
    except subprocess.CalledProcessError as e:
        results.append((name, f"FAIL (exit {e.returncode})"))
    except Exception as e:
        results.append((name, f"FAIL ({e})"))

def run_should_fail(name, cmd):
    cmd = [str(x) for x in cmd]
    ret = subprocess.run(cmd, capture_output=True)
    if ret.returncode != 0:
        results.append((name, "OK (rejected as expected)"))
    else:
        results.append((name, "FAIL (should have been rejected)"))

# ── Common base args ───────────────────────────────────────────────────────
base = ['corgias', 'profiling', '-c', args.cores, '--test', N]

# ── Test cases ─────────────────────────────────────────────────────────────

run_case('naive_secondary',
         base + ['-m', 'naive',
                 '-og', primary_path, '-og2', secondary_path,
                 '-o', output_dir / 'naive_secondary.csv'],
         out_path=output_dir / 'naive_secondary.csv',
         expected=expected_rows)

run_case('rle_secondary',
         base + ['-m', 'rle',
                 '-og', primary_path, '-og2', secondary_path, '-t', tree,
                 '-o', output_dir / 'rle_secondary.csv'],
         out_path=output_dir / 'rle_secondary.csv',
         expected=expected_rows)

run_case('cwa_secondary',
         base + ['-m', 'cwa',
                 '-og', primary_path, '-og2', secondary_path, '-t', tree,
                 '-o', output_dir / 'cwa_secondary.csv'],
         out_path=output_dir / 'cwa_secondary.csv',
         expected=expected_rows)

run_case('cotr_secondary',
         base + ['-m', 'cotr',
                 '-og', primary_path, '-og2', secondary_path, '-t', tree,
                 '-o', output_dir / 'cotr_secondary.csv'],
         out_path=output_dir / 'cotr_secondary.csv',
         expected=expected_rows)

run_case('asa_secondary',
         base + ['-m', 'asa',
                 '-a', asr_ml, '-a2', asr_ml, '-t', tree,
                 '-o', output_dir / 'asa_secondary.csv'])

run_case('sev_secondary',
         base + ['-m', 'sev',
                 '-a', asr_mp, '-a2', asr_mp, '-t', tree,
                 '-o', output_dir / 'sev_secondary.csv'])

# ── Validation rejection cases ─────────────────────────────────────────────
run_should_fail('reject_og2_with_asa',
                base + ['-m', 'asa',
                        '-og2', secondary_path, '-a', asr_ml, '-t', tree,
                        '-o', output_dir / 'reject1.csv'])

run_should_fail('reject_a2_with_naive',
                base + ['-m', 'naive',
                        '-og', primary_path, '-a2', asr_ml,
                        '-o', output_dir / 'reject2.csv'])

run_should_fail('reject_query_with_og2',
                base + ['-m', 'naive',
                        '-og', primary_path, '-og2', secondary_path,
                        '-q', 'COG0001',
                        '-o', output_dir / 'reject3.csv'])

# ── Report ─────────────────────────────────────────────────────────────────
print(f"{'Case':<30} {'Result'}")
print('-' * 55)
failures = 0
for name, status in results:
    print(f"{name:<30} {status}")
    if 'FAIL' in status:
        failures += 1

print()
if failures:
    print(f"{failures} case(s) FAILED.")
    sys.exit(1)
else:
    print("All cases passed.")
