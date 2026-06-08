import logging
import math
import numpy as np
import polars as pl
import ete3 as et
from multiprocessing import Pool
from numpy.typing import NDArray
from tqdm import tqdm
from .common import load_og_table, flatten_indices, list_asr_trees, uppermatrix2vector, log_secondary_mode, pastml_attr
from scipy.linalg.blas import dsyrk as _dsyrk
from .gpu_utils import cp, block_dot, _gpu_dtype, _gpu_syrk

logger = logging.getLogger(__name__)


def run_cotr(args):
    df = load_og_table(args.og_table, query=args.query)
    if args.test:
        df = df.iloc[:, :args.test]
    tree = et.Tree(args.tree, format=1)
    order = [leaf.name for leaf in tree.get_leaves()]
    df = df.loc[order]
    num_genomes = len(order) - 1

    if args.og_table2:
        df2 = load_og_table(args.og_table2)
        if args.test:
            df2 = df2.iloc[:, :args.test]
        df2 = df2.loc[order]
        count1 = _count_transitions(df, args)
        count2 = _count_transitions(df2, args)
        og_names1, t1, num_trans1 = _unpack_count(count1)
        og_names2, t2, num_trans2 = _unpack_count(count2)
        n1, n2 = len(og_names1), len(og_names2)
        log_secondary_mode(logger, n1, n2)
        backend = "GPU" if args.gpu else "CPU"
        logger.info(f"Computing transition matrix ({n1}x{n2} cross + {n2}x{n2}) on {backend}.")
        k_cross = calculate_k(t1, t2.T, gpu=args.gpu, num_blocks=args.num_blocks, symmetric=False)
        k2     = calculate_k(t2, t2.T, gpu=args.gpu, num_blocks=args.num_blocks)
        return pl.concat([
            transition_cross2df(k_cross, og_names1, og_names2, num_trans1, num_trans2, num_genomes),
            transition_count2df(k2, og_names2, num_trans2, None, num_genomes, args),
        ])

    n = len(df.columns)
    logger.info(f"Counting transitions for {n} OGs.")
    count = _count_transitions(df, args)
    og_names, t, t_T, num_transition, num_transition_query = prepare_matrix(count, args)
    backend = "GPU" if args.gpu else "CPU"
    logger.info(f"Computing transition matrix on {backend}.")
    k = calculate_k(t, t_T, gpu=args.gpu, num_blocks=args.num_blocks)
    return transition_count2df(k, og_names, num_transition, num_transition_query, num_genomes, args)


def count_transition(og:str, row: NDArray[np.int64]
                     ) -> tuple[str, NDArray[np.int64], int]:
    row = np.array(row)
    shifted = row.copy()[1:]
    transition = shifted - row[:-1]
    for i in range(len(transition) -1):
        if transition[i] + transition[i+1] == 0:
            transition[i+1] = 0

    num_transition = np.count_nonzero(transition)

    return og, transition, num_transition


def calculate_k(df: np.ndarray, df_T: np.ndarray,
                gpu: bool = False, num_blocks: int = 0,
                symmetric: bool = True) -> NDArray:
    if df_T.ndim == 1:
        symmetric = False
    if gpu:
        if symmetric and num_blocks == 0:
            k = cp.asnumpy(_gpu_syrk(cp.asarray(df, dtype=cp.float32), 'N'))
        elif num_blocks == 0:
            dtype = _gpu_dtype(df.shape[1])
            k = cp.asnumpy(cp.dot(cp.asarray(df, dtype=dtype),
                                   cp.asarray(df_T, dtype=dtype)))
        else:
            block_size = df.shape[0] // num_blocks
            k = block_dot(df, df_T, block_size)
    elif symmetric:
        k = _dsyrk(1.0, df.astype(np.float64), trans=0, lower=0)
    else:
        k = np.dot(df.astype(np.float64),
                   df_T.astype(np.float64) if df_T.ndim > 1 else df_T)
    return k


def transition_count2df(k: np.ndarray, og_names: list[str],  num_transition: np.ndarray, 
                        num_transition_query: np.ndarray, N: int, args) -> pl.DataFrame:
    if num_transition_query:
        return pl.DataFrame({'OG1':[args.query]*len(og_names), 'OG2':og_names,
                             'num_change1':[num_transition_query]*len(og_names),
                             'num_change2':num_transition.flatten(), 'k':k.astype(np.int64), 'N':[N]*len(og_names)})
    else:
        indices = flatten_indices(k)
        og_names = pl.DataFrame(og_names).with_row_count('index').select(
                pl.col('index').cast(pl.Int64), pl.col('column_0').alias('OG'))
        num_transition = pl.DataFrame(num_transition).with_row_count('index').select(
                    pl.col('index').cast(pl.Int64), pl.col('column_0').alias('num_transition'))

        indices = indices.join(
                og_names, left_on='column_0', right_on='index'
                ).rename({'OG':'OG1'}).join(
                og_names, left_on='column_1', right_on='index'
                ).rename({'OG':'OG2'}).join(
                num_transition, left_on='column_0', right_on='index'
                ).rename({'num_transition':'num_change1'}).join(
                num_transition, left_on='column_1', right_on='index'
                ).rename({'num_transition':'num_change2'}).select(
                ['OG1', 'OG2', 'num_change1', 'num_change2']
                )
        k = uppermatrix2vector(k)
        k = pl.DataFrame({'k': k.astype(np.int64)})
        result = pl.concat([indices, k], how='horizontal')
        df = result.with_columns(pl.lit(N, dtype=pl.Int64).alias('N'))

    return df


def run_sev(args):
    trees = list_asr_trees(args.asr_folder, args.tree, query=args.query)
    if args.test:
        trees = trees[:args.test]
    tree = et.Tree(args.tree, format=1)
    num_internal_nodes = len(tree.get_leaves()) - 1

    if args.asr_folder2:
        trees2 = list_asr_trees(args.asr_folder2, args.tree)
        if args.test:
            trees2 = trees2[:args.test]
        count1 = _count_changes(trees, args)
        count2 = _count_changes(trees2, args)
        og_names1, t1, num_trans1 = _unpack_count(count1)
        og_names2, t2, num_trans2 = _unpack_count(count2)
        n1, n2 = len(og_names1), len(og_names2)
        log_secondary_mode(logger, n1, n2)
        backend = "GPU" if args.gpu else "CPU"
        logger.info(f"Computing transition matrix ({n1}x{n2} cross + {n2}x{n2}) on {backend}.")
        k_cross = calculate_k(t1, t2.T, gpu=args.gpu, num_blocks=args.num_blocks)
        k2 = calculate_k(t2, t2.T, gpu=args.gpu, num_blocks=args.num_blocks, symmetric=True)
        return pl.concat([
            transition_cross2df(k_cross, og_names1, og_names2, num_trans1, num_trans2, num_internal_nodes),
            transition_count2df(k2, og_names2, num_trans2, None, num_internal_nodes, args),
        ])

    n = len(trees)
    logger.info(f"Counting state changes for {n} OGs.")
    count = _count_changes(trees, args)
    og_names, t, t_T, num_transition, num_transition_query = prepare_matrix(count, args)
    backend = "GPU" if args.gpu else "CPU"
    logger.info(f"Computing transition matrix on {backend}.")
    k = calculate_k(t, t_T, gpu=args.gpu, num_blocks=args.num_blocks)
    return transition_count2df(k, og_names, num_transition, num_transition_query, num_internal_nodes, args)


def count_change(tree: et.TreeNode, og: str
                 ) -> tuple[str, NDArray[np.int64], int]:
    tree = et.Tree(tree, format=1)
    transition = []
    attr = pastml_attr(og)
    for node in tree.traverse():
        if not node.is_leaf():
            parent_state = getattr(node, attr)
            for child in node.get_children():
                child_state = getattr(child, attr)
                try:
                    transition.append(int(float(child_state)) - int(float(parent_state)))
                except ValueError:
                    transition.append(0)

    num_transition = np.count_nonzero(transition)

    return og, transition, num_transition


def _count_transitions(df, args) -> list:
    ogs = ((i, row) for i, row in df.T.iterrows())
    with Pool(processes=args.cores) as pool:
        with tqdm(total=len(df.columns), disable=args.quiet) as pbar:
            futures = [pool.apply_async(count_transition, og, callback=lambda _: pbar.update())
                       for og in ogs]
            return [f.get() for f in futures]


def _count_changes(trees, args) -> list:
    with Pool(processes=args.cores) as pool:
        with tqdm(total=len(trees), disable=args.quiet) as pbar:
            futures = [pool.apply_async(count_change, tree, callback=lambda _: pbar.update())
                       for tree in trees]
            return [f.get() for f in futures]


def _unpack_count(count: list) -> tuple[list, NDArray, NDArray]:
    og_names = [c[0] for c in count]
    t_matrix = np.vstack([c[1] for c in count])
    num_transitions = np.array([c[2] for c in count])
    return og_names, t_matrix, num_transitions


def transition_cross2df(k: NDArray, og_names1: list, og_names2: list,
                        num_trans1: NDArray, num_trans2: NDArray, N: int) -> pl.DataFrame:
    n1, n2 = len(og_names1), len(og_names2)
    OG1 = [og_names1[i] for i in range(n1) for _ in range(n2)]
    OG2 = og_names2 * n1
    nc1 = [int(num_trans1[i]) for i in range(n1) for _ in range(n2)]
    nc2 = [int(x) for x in num_trans2] * n1
    return pl.DataFrame({'OG1': OG1, 'OG2': OG2,
                         'num_change1': nc1, 'num_change2': nc2,
                         'k': k.flatten().astype(np.int64),
                         'N': [N] * len(OG1)},
                        schema={'OG1': pl.Utf8, 'OG2': pl.Utf8,
                                'num_change1': pl.Int64, 'num_change2': pl.Int64,
                                'k': pl.Int64, 'N': pl.Int64})


def prepare_matrix(count, args):
    og_names = [ sublist[0] for sublist in count ]
    t_matrix = np.vstack([ sublist[1] for sublist in count ])
    num_transition = np.vstack([ sublist[2] for sublist in count ])
    num_transition_query = None
    
    if args.query:
        df = t_matrix[1:]
        df_T = t_matrix[0]
        num_transition_query = num_transition[0][0]
        num_transition = num_transition[1:]
        og_names.pop(0)
    else:
        df = t_matrix
        df_T = t_matrix.T

    return og_names, df, df_T, num_transition, num_transition_query


