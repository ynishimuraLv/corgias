"""
Benchmark: current naive_cpu (4x np.dot on int arrays with Pool)
        vs dsyrk approach  (2x dsyrk + 1x dgemm + transpose, float64)

Usage:
    python script/bench_dsyrk.py [--repeats N] [--ogs N] [--genomes N]
"""

import argparse
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.linalg.blas import dsyrk


# ── current approach ──────────────────────────────────────────────────────────

def current_naive_cpu(df_T, df, df_T_flipped, df_flipped, cores=4):
    jobs = [(df_T.values, df.values),
            (df_T.values, df_flipped.values),
            (df_T_flipped.values, df.values),
            (df_T_flipped.values, df_flipped.values)]
    cores = min(cores, 4)
    with Pool(processes=cores) as pool:
        tt, tf, ft, ff = pool.starmap(np.dot, jobs)
    return tt, tf, ft, ff


# ── new approach ──────────────────────────────────────────────────────────────

def dsyrk_naive_cpu(df_vals: np.ndarray, df_flipped_vals: np.ndarray):
    """
    tt = A.T @ A  (symmetric) → dsyrk, upper triangle only
    ff = B.T @ B  (symmetric) → dsyrk, upper triangle only
    tf = A.T @ B  (full needed: upper tri of tf + lower tri of tf = upper tri of ft)
    ft = tf.T     (free)
    """
    a = df_vals.astype(np.float64)
    b = df_flipped_vals.astype(np.float64)
    tt = dsyrk(1.0, a, trans=1, lower=0)   # upper triangle of A.T @ A
    ff = dsyrk(1.0, b, trans=1, lower=0)   # upper triangle of B.T @ B
    tf = a.T @ b                            # float64 BLAS dgemm
    ft = tf.T
    return tt, tf, ft, ff


# ── helpers ───────────────────────────────────────────────────────────────────

def make_data(n_genomes, n_ogs, seed=42):
    rng = np.random.default_rng(seed)
    vals = rng.integers(0, 2, size=(n_genomes, n_ogs), dtype=np.int64)
    df = pd.DataFrame(vals)
    df_flipped = df.replace({0: 1, 1: 0})
    df_T = df.T
    df_T_flipped = df_flipped.T
    return df, df_flipped, df_T, df_T_flipped, vals


def upper_idx(n):
    return np.triu_indices(n, k=1)


def bench(label, fn, n_repeats):
    times = []
    result = None
    for i in range(n_repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    best = min(times)
    print(f"  {label:<30s}  best={best:.3f}s  ({n_repeats} runs, times: {[f'{t:.3f}' for t in times]})")
    return result, best


def verify(tt1, tf1, ft1, ff1, tt2, tf2, ft2, ff2, n_ogs):
    idx = upper_idx(n_ogs)
    assert np.allclose(tt1[idx], tt2[idx]), "tt upper triangle mismatch"
    assert np.allclose(ff1[idx], ff2[idx]), "ff upper triangle mismatch"
    assert np.allclose(tf1[idx], tf2[idx]), "tf upper triangle mismatch"
    assert np.allclose(ft1[idx], ft2[idx]), "ft upper triangle mismatch"
    print("  correctness: OK (upper triangles match)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ogs", type=int, default=3308, help="number of OGs (cols)")
    parser.add_argument("--genomes", type=int, default=1921, help="number of genomes (rows)")
    parser.add_argument("--cores", type=int, default=4)
    args = parser.parse_args()

    print(f"Data: {args.genomes} genomes × {args.ogs} OGs  (cores={args.cores})")
    print()

    df, df_flipped, df_T, df_T_flipped, vals = make_data(args.genomes, args.ogs)
    df_flipped_vals = 1 - vals

    print("Running current (4x np.dot int, multiprocessing Pool)…")
    (tt1, tf1, ft1, ff1), t_cur = bench(
        "current",
        lambda: current_naive_cpu(df_T, df, df_T_flipped, df_flipped, cores=args.cores),
        args.repeats,
    )

    print("Running dsyrk (2x dsyrk + 1x dgemm float64, no Pool)…")
    (tt2, tf2, ft2, ff2), t_new = bench(
        "dsyrk",
        lambda: dsyrk_naive_cpu(vals, df_flipped_vals),
        args.repeats,
    )

    print()
    verify(tt1, tf1, ft1, ff1, tt2, tf2, ft2, ff2, args.ogs)
    print()
    print(f"  speedup: {t_cur / t_new:.2f}x")
    print()

    # ── also benchmark calculate_k (cotr/sev) ────────────────────────────────
    print("── calculate_k (cotr/sev symmetric case) ──")
    n_nodes = args.genomes - 1
    rng = np.random.default_rng(0)
    t_mat = rng.integers(-1, 2, size=(args.ogs, n_nodes)).astype(np.int64)
    t_mat_f = t_mat.astype(np.float64)

    print("Running current (np.dot int)…")
    _, t_k_cur = bench(
        "current calculate_k",
        lambda: np.dot(t_mat, t_mat.T),
        args.repeats,
    )

    print("Running dsyrk (float64)…")
    _, t_k_new = bench(
        "dsyrk calculate_k",
        lambda: dsyrk(1.0, t_mat_f, trans=0, lower=0),
        args.repeats,
    )

    print()
    print(f"  speedup: {t_k_cur / t_k_new:.2f}x")


if __name__ == "__main__":
    main()
