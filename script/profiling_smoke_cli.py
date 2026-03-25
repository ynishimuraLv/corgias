import os
import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
TEST = ROOT / "test"

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_dir')
parser.add_argument('-c', '--cores', default=5)
parser.add_argument('-n', '--num_tests', default=5)
args = parser.parse_args()
body = ['corgias', 'profiling', '-c', args.cores, '--test', args.num_tests, '--log-file', f'{args.output_dir}/log.txt']

og_table = SAMPLES / "archaea_COG_table99.csv"
tree = SAMPLES / "archaea_hq90.tre"
query = ['-q', 'COG0003']

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

CASES = [
    {
        'name': 'naive',
        'cmd':body + ['-m', 'naive', '-og', og_table,
                      '-o', f"{args.output_dir}/naive.csv"],
    },
    {
        'name': 'naive_query',
        'cmd':body + ['-m', 'naive', '-og', og_table,
                      '-o', f"{args.output_dir}/naive_COG0003.csv", '-q', 'COG0003'
                     ] + query
    },
    {
        'name': 'rle',
        'cmd':body + ['-m', 'rle', '-og', og_table, '-t', tree,
                      '-o', f"{args.output_dir}/rle.csv"],
    },
    {
        'name': 'rle_query',
        'cmd':body + ['-m', 'rle', '-og', og_table, '-t', tree,
                      '-o', f"{args.output_dir}/rle_query.csv"]
                   + query
    },
    {
        'name': 'cwa',
        'cmd':body + ['-m', 'cwa', '-og', og_table, '-t', tree,
                      '-o', f"{args.output_dir}/cwa.csv"],
    },
    {
        'name': 'cwa_query',
        'cmd':body + ['-m', 'rle', '-og', og_table, '-t', tree,
                      '-o', f"{args.output_dir}/cwa_query.csv"]
                   + query
    },
    {
        'name': 'asa',
        'cmd':body + ['-m', 'asa', '-t', tree,
                      '-a', str(TEST / "ML_result"),
                      '-o', f"{args.output_dir}/asa.csv"],
    },
    {
        'name': 'asa_query',
        'cmd':body + ['-m', 'asa', '-t', tree,
                      '-a', str(TEST / "ML_result"),
                      '-o', f"{args.output_dir}/asa_query.csv"]
                   + query
    },
    {
        'name': 'cotr',
        'cmd':body + ['-m', 'cotr', '-t', tree, '-og', og_table,
                      '-o', f"{args.output_dir}/cotr.csv"]
    },
    {
        'name': 'cotr_query',
        'cmd':body + ['-m', 'cotr', '-t', tree, '-og', og_table,
                      '-o', f"{args.output_dir}/cotr_query.csv"]
                   + query
    },
    {
        'name': 'sev',
        'cmd':body + ['-m', 'sev', '-t', tree,
                      '-a', str(TEST / "MP_result"),
                      '-o', f"{args.output_dir}/sev.csv"],
    },
    {
        'name': 'sev_query',
        'cmd':body + ['-m', 'sev', '-t', tree,
                      '-a', str(TEST / "MP_result"),
                      '-o', f"{args.output_dir}/sev_query.csv"]
                   + query
    },
]

def run_case(case):
    try:
        subprocess.run(case["cmd"], check=True)
        print("OK")
    except subprocess.CalledProcessError:
        print("FAILED:", case["name"])
        
if __name__ == "__main__":
    for case in CASES:
        print(f"Running {case['name']}...")
        run_case(case)
        print("\n\n")
