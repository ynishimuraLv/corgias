import os
import sys
import argparse
import subprocess
from pathlib import Path

import polars as pl

try:
    import cupy  # noqa: F401
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))
from test_init import ensure_fixtures

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
TEST = ROOT / "test"
ANSWER_DIR = TEST / "answer"

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_dir')
parser.add_argument('-c', '--cores', default=5, type=int)
parser.add_argument('-n', '--num_tests', default=5, type=int)
args = parser.parse_args()

print("Checking test fixtures ...")
if not ensure_fixtures(cores=args.cores, num_tests=args.num_tests * 2):
    sys.exit(1)
print()
body = ['corgias', 'profiling', '-c', args.cores, '--test', args.num_tests, '--log-file', f'{args.output_dir}/log.txt']

og_table = SAMPLES / "archaea_COG_table99.csv"
tree = SAMPLES / "archaea_hq90.tre"
query = ['-q', 'COG0003']

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

OUT = Path(args.output_dir)

# Column pairs that must be swapped together when OG1/OG2 are exchanged
SWAP_TF_FT   = [('TF', 'FT')]
SWAP_CHANGE  = [('num_change1', 'num_change2')]

CASES = [
    {
        'name': 'naive',
        'cmd': body + ['-m', 'naive', '-og', og_table,
                       '-o', OUT / 'naive.csv'],
        'output_path': OUT / 'naive.csv',
        'answer_method': 'naive', 'swap_pairs': SWAP_TF_FT,
    },
    {
        'name': 'naive_query',
        'cmd': body + ['-m', 'naive', '-og', og_table,
                       '-o', OUT / 'naive_COG0003.csv', '-q', 'COG0003'] + query,
        'output_path': OUT / 'naive_COG0003.csv',
        'answer_method': 'naive', 'swap_pairs': SWAP_TF_FT,
    },
    {
        'name': 'rle',
        'cmd': body + ['-m', 'rle', '-og', og_table, '-t', tree,
                       '-o', OUT / 'rle.csv'],
        'output_path': OUT / 'rle.csv',
        'answer_method': 'rle', 'swap_pairs': SWAP_TF_FT,
    },
    {
        'name': 'rle_query',
        'cmd': body + ['-m', 'rle', '-og', og_table, '-t', tree,
                       '-o', OUT / 'rle_query.csv'] + query,
        'output_path': OUT / 'rle_query.csv',
        'answer_method': 'rle', 'swap_pairs': SWAP_TF_FT,
    },
    {
        'name': 'cwa',
        'cmd': body + ['-m', 'cwa', '-og', og_table, '-t', tree,
                       '-o', OUT / 'cwa.csv'],
        'output_path': OUT / 'cwa.csv',
        'answer_method': 'cwa', 'swap_pairs': SWAP_TF_FT,
    },
    {
        'name': 'cwa_query',
        'cmd': body + ['-m', 'rle', '-og', og_table, '-t', tree,
                       '-o', OUT / 'cwa_query.csv'] + query,
        'output_path': OUT / 'cwa_query.csv',
        'answer_method': 'rle', 'swap_pairs': SWAP_TF_FT,
    },
    {
        'name': 'asa',
        'cmd': body + ['-m', 'asa', '-t', tree,
                       '-a', str(TEST / "ML_result"),
                       '-o', OUT / 'asa.csv'],
        'output_path': OUT / 'asa.csv',
        'answer_method': 'asa', 'swap_pairs': SWAP_TF_FT, 'float_tol': 5e-3,
    },
    {
        'name': 'asa_query',
        'cmd': body + ['-m', 'asa', '-t', tree,
                       '-a', str(TEST / "ML_result"),
                       '-o', OUT / 'asa_query.csv'] + query,
        'output_path': OUT / 'asa_query.csv',
        'answer_method': 'asa', 'swap_pairs': SWAP_TF_FT, 'float_tol': 5e-3,
    },
    {
        'name': 'cotr',
        'cmd': body + ['-m', 'cotr', '-t', tree, '-og', og_table,
                       '-o', OUT / 'cotr.csv'],
        'output_path': OUT / 'cotr.csv',
        'answer_method': 'cotr', 'swap_pairs': SWAP_CHANGE,
    },
    {
        'name': 'cotr_query',
        'cmd': body + ['-m', 'cotr', '-t', tree, '-og', og_table,
                       '-o', OUT / 'cotr_query.csv'] + query,
        'output_path': OUT / 'cotr_query.csv',
        'answer_method': 'cotr', 'swap_pairs': SWAP_CHANGE,
    },
    {
        'name': 'sev',
        'cmd': body + ['-m', 'sev', '-t', tree,
                       '-a', str(TEST / "MP_result"),
                       '-o', OUT / 'sev.csv'],
        'output_path': OUT / 'sev.csv',
        'answer_method': 'sev', 'swap_pairs': SWAP_CHANGE,
    },
    {
        'name': 'sev_query',
        'cmd': body + ['-m', 'sev', '-t', tree,
                       '-a', str(TEST / "MP_result"),
                       '-o', OUT / 'sev_query.csv'] + query,
        'output_path': OUT / 'sev_query.csv',
        'answer_method': 'sev', 'swap_pairs': SWAP_CHANGE,
    },
]

if CUPY_AVAILABLE:
    CASES += [
        {
            'name': 'naive_gpu',
            'cmd': body + ['--gpu', '-m', 'naive', '-og', og_table,
                           '-o', OUT / 'naive_gpu.csv'],
            'output_path': OUT / 'naive_gpu.csv',
            'answer_method': 'naive', 'swap_pairs': SWAP_TF_FT,
        },
        {
            'name': 'naive_gpu_query',
            'cmd': body + ['--gpu', '-m', 'naive', '-og', og_table,
                           '-o', OUT / 'naive_gpu_COG0003.csv'] + query,
            'output_path': OUT / 'naive_gpu_COG0003.csv',
            'answer_method': 'naive', 'swap_pairs': SWAP_TF_FT,
        },
    ]


def normalize_pairs(df: pl.DataFrame, swap_pairs: list) -> pl.DataFrame:
    """Canonicalize so OG1 < OG2 lexicographically, swapping associated column pairs."""
    needs_swap = pl.col('OG1') > pl.col('OG2')
    exprs = [
        pl.when(needs_swap).then(pl.col('OG2')).otherwise(pl.col('OG1')).alias('OG1'),
        pl.when(needs_swap).then(pl.col('OG1')).otherwise(pl.col('OG2')).alias('OG2'),
    ]
    for a, b in swap_pairs:
        exprs += [
            pl.when(needs_swap).then(pl.col(b)).otherwise(pl.col(a)).alias(a),
            pl.when(needs_swap).then(pl.col(a)).otherwise(pl.col(b)).alias(b),
        ]
    return df.with_columns(exprs)


def check_answer(computed_path: Path, answer_path: Path,
                 swap_pairs: list, float_tol: float | None = None) -> tuple[bool, str]:
    """Compare computed result against answer file. Returns (ok, message).

    Handles OG1/OG2 swaps: when the pair appears in reversed order in either
    file, swap_pairs columns (e.g. TF/FT or num_change1/num_change2) are also
    exchanged before comparison.
    """
    computed = pl.read_csv(computed_path)
    og_set = set(computed['OG1'].to_list() + computed['OG2'].to_list())

    # Load only rows relevant to our OG set (both OGs must appear in the result)
    answer = pl.read_csv(answer_path).filter(
        pl.col('OG1').is_in(og_set) & pl.col('OG2').is_in(og_set)
    )

    computed_norm = normalize_pairs(computed, swap_pairs)
    answer_norm   = normalize_pairs(answer,   swap_pairs)

    value_cols = [c for c in computed_norm.columns if c not in ('OG1', 'OG2')]
    answer_renamed = answer_norm.rename({c: f'_ans_{c}' for c in value_cols})
    joined = computed_norm.join(answer_renamed, on=['OG1', 'OG2'], how='left')

    sentinel = f'_ans_{value_cols[0]}'
    missing = joined.filter(pl.col(sentinel).is_null())
    if len(missing) > 0:
        eg = f"{missing[0, 'OG1']}/{missing[0, 'OG2']}"
        return False, f"{len(missing)} pair(s) not found in answer (e.g. {eg})"

    for col in value_cols:
        c_vals = joined[col]
        a_vals = joined[f'_ans_{col}']
        if float_tol is not None and c_vals.dtype in (pl.Float32, pl.Float64):
            diff  = (c_vals - a_vals).abs()
            scale = a_vals.abs().clip(lower_bound=1.0)
            if (diff / scale > float_tol).any():
                worst = (diff / scale).max()
                return False, f"column '{col}' max relative error {worst:.2e} > {float_tol}"
        else:
            mismatches = (c_vals != a_vals).sum()
            if mismatches > 0:
                return False, f"column '{col}': {mismatches} mismatch(es)"

    return True, f"answer matches ({len(computed_norm)} pairs)"


def run_case(case):
    try:
        subprocess.run([str(x) for x in case["cmd"]], check=True)
    except subprocess.CalledProcessError:
        print("FAILED:", case["name"])
        return

    out_path = case.get("output_path")
    answer_method = case.get("answer_method")
    if out_path and answer_method and ANSWER_DIR.exists():
        answer_path = ANSWER_DIR / f"{answer_method}.csv"
        if answer_path.exists():
            ok, msg = check_answer(
                out_path, answer_path,
                case["swap_pairs"],
                float_tol=case.get("float_tol"),
            )
            print("OK" if ok else "FAILED (answer check)", f"— {msg}")
            return

    print("OK")


if __name__ == "__main__":
    for case in CASES:
        print(f"Running {case['name']}...")
        run_case(case)
        print("\n\n")
