import logging
import math
import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import NDArray
from multiprocessing import Pool

from .common import load_og_table, uppermatrix2vector, flatten_indices, log_secondary_mode
from .gpu_utils import cp, block_dot

logger = logging.getLogger(__name__)

def run_naive(args):
    df = load_og_table(args.og_table, query=args.query)
    if args.test:
        df = df.iloc[:, :args.test]

    if args.og_table2:
        df2 = load_og_table(args.og_table2)
        if args.test:
            df2 = df2.iloc[:, :args.test]
        n1, n2 = len(df.columns), len(df2.columns)
        log_secondary_mode(logger, n1, n2)
        df2_flipped, _, _ = prepare_matrices(df2)
        _, df1_T, df1_T_flipped = prepare_matrices(df)
        backend = "GPU" if args.gpu else "CPU"
        logger.info(f"Computing co-/anti-occurrence matrix ({n1}x{n2} cross) on {backend}.")
        if args.gpu:
            tt, tf, ft, ff = naive_gpu(df2, df2_flipped, df1_T, df1_T_flipped, args.num_blocks)
        else:
            tt, tf, ft, ff = naive_cpu(df2, df2_flipped, df1_T, df1_T_flipped, args.cores)
        return pl.concat([
            naivecross2matrix(tt, tf, ft, ff, list(df.columns), list(df2.columns)),
            _naive_allvall(df2, args),
        ])

    n = len(df.columns)
    num_pairs = n - 1 if args.query else math.comb(n, 2)
    logger.info(f"Processing {num_pairs} OG pairs.")
    df_flipped, df_T, df_T_flipped = prepare_matrices(df)
    if args.query:
        idx = df.columns.get_loc(args.query)
        df = df.loc[:, args.query]
        df_T.drop(args.query, inplace=True)
        df_flipped = df_flipped.iloc[:, idx]
        df_T_flipped.drop(args.query, inplace=True)
    backend = "GPU" if args.gpu else "CPU"
    logger.info(f"Computing co-/anti-occurrence matrix on {backend}.")
    if args.gpu:
        tt, tf, ft, ff = naive_gpu(df, df_flipped, df_T, df_T_flipped, args.num_blocks)
    else:
        tt, tf, ft, ff = naive_cpu(df, df_flipped, df_T, df_T_flipped, args.cores)

    og_names = list(df_T.index)
    if args.query:
        return pl.DataFrame({'OG1':[args.query]*len(og_names), 'OG2':og_names,
                             'TT':tt, 'TF':tf, 'FT':ft, 'FF':ff})
    else:
        return naivecount2matrix(tt, tf, ft, ff, og_names)


def prepare_matrices(df: pd.DataFrame
                     ) -> tuple[pd.DataFrame, pd.DataFrame,
                                pd.DataFrame]:
    df_flipped = df.replace({0:1, 1:0})
    df_T = df.T
    df_T_flipped = df_T.replace({0:1, 1:0})

    return df_flipped, df_T, df_T_flipped


def naive_gpu(df: pd.DataFrame, df_T: pd.DataFrame, df_flipped: pd.DataFrame, 
                  df_T_flipped: pd.DataFrame, num_blocks: int = 0
                 ) -> tuple[NDArray[np.int64], NDArray[np.int64],
                         NDArray[np.int64], NDArray[np.int64]]:
    
    # cp.int16で十分かは後で考える
    df = cp.asarray(df, dtype=cp.int16)
    df_T = cp.asarray(df_T, dtype=cp.int16)
    df_flipped = cp.asarray(df_flipped, dtype=cp.int16)
    df_T_flipped = cp.asarray(df_T_flipped, dtype=cp.int16)
    if num_blocks == 0:
        tt = cp.asnumpy(cp.dot(df_T, df))
        tf = cp.asnumpy(cp.dot(df_T, df_flipped))
        ft = cp.asnumpy(cp.dot(df_T_flipped, df))
        ff = cp.asnumpy(cp.dot(df_T_flipped, df_flipped))
    else:
        block_size = df_T.shape[0] // num_blocks
        tt = block_dot(df_T, df, block_size)
        tf = block_dot(df_T, df_flipped, block_size)
        ft = block_dot(df_T_flipped, df, block_size)
        ff = block_dot(df_T_flipped, df_flipped, block_size)

    return tt, tf, ft, ff


def naive_cpu(df: pd.DataFrame, df_flipped: pd.DataFrame, df_T: pd.DataFrame, 
                  df_T_flipped: pd.DataFrame, cores: int) -> tuple[int, int, int, int]:
    jobs = [(df_T, df), (df_T, df_flipped),
            (df_T_flipped, df), (df_T_flipped, df_flipped)]
    if cores > 4:
        cores = 4
    with Pool(processes=cores) as process:
        tt, tf, ft, ff = process.starmap(np.dot, jobs)

    return tt, tf, ft, ff


def _naive_allvall(df: pd.DataFrame, args) -> pl.DataFrame:
    df_flipped, df_T, df_T_flipped = prepare_matrices(df)
    if args.gpu:
        tt, tf, ft, ff = naive_gpu(df, df_flipped, df_T, df_T_flipped, args.num_blocks)
    else:
        tt, tf, ft, ff = naive_cpu(df, df_flipped, df_T, df_T_flipped, args.cores)
    return naivecount2matrix(tt, tf, ft, ff, list(df.columns))


def naivecross2matrix(tt: NDArray[np.int64], tf: NDArray[np.int64],
                      ft: NDArray[np.int64], ff: NDArray[np.int64],
                      og1_names: list[str], og2_names: list[str]) -> pl.DataFrame:
    n1, n2 = len(og1_names), len(og2_names)
    OG1 = [og1_names[i] for i in range(n1) for _ in range(n2)]
    OG2 = og2_names * n1
    return pl.DataFrame({'OG1': OG1, 'OG2': OG2,
                         'TT': tt.flatten(), 'TF': tf.flatten(),
                         'FT': ft.flatten(), 'FF': ff.flatten()})


def naivecount2matrix(tt: NDArray[np.int64], tf: NDArray[np.int64],
                      ft: NDArray[np.int64], ff: NDArray[np.int64],
                      og_names: list[str]) -> pl.DataFrame:
    indices = flatten_indices(tt)
    og_names = pl.DataFrame(og_names).with_row_count('index')
    og_names = og_names.with_columns(pl.col('index').cast(pl.Int64))
    og_names = og_names.rename({'column_0':'OG'})
    indices = indices.join(
                    og_names, left_on='column_0', right_on='index'
                 ).rename({'OG':'OG1'}).join(
                    og_names, left_on='column_1', right_on='index'
                 ).rename({'OG':'OG2'}).select('OG1', 'OG2')

    tt = uppermatrix2vector(tt)
    tf = uppermatrix2vector(tf)
    ft = uppermatrix2vector(ft)
    ff = uppermatrix2vector(ff)
    df = pl.DataFrame({'TT':tt, 'TF':tf, 'FT':ft, 'FF':ff})

    df = pl.concat([indices, df], how='horizontal')

    return df

