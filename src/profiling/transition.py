import numpy as np
import polars as pl
import ete3 as et
from multiprocessing import Pool
from numpy.typing import NDArray
from .common import load_og_table, flatten_indices, list_asr_trees
from .gpu_utils import cp, block_dot


def run_cotr(args):
    df = load_og_table(args)
    tree = et.Tree(args.tree, format=1)
    order = [ leaf.name for leaf in tree.get_leaves() ]
    df = df.loc[order]
    ogs = ((i, row) for i, row in df.T.iterrows())
    with Pool(processes=args.cores) as process:
        count = process.starmap_async(count_transition, ogs).get()
    num_genomes = len(order) - 1
    og_names, df, df_T, num_transition, num_transition_query = prepare_matrix(count, args)
    k = calculate_k(df, df_T, gpu=args.gpu, num_blocks=args.num_blocks)
    result = transition_count2df(k, og_names, num_transition,
                                 num_transition_query, num_genomes, args)
    
    return result


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
                gpu: bool = False, num_blocks: int = 0) -> NDArray[np.int64]:
    if gpu:
        if num_blocks == 0:
            df = cp.asarray(df, dtype=cp.int16)
            df_T = cp.asarray(df_T, dtype=cp.int16)
            k = cp.asnumpy(cp.dot(df, df_T))
        else:
            block_size = df.shape[0] // num_blocks
            k = block_dot(df,df_T, block_size)
    else:
        k = np.dot(df, df_T)

    return k


def transition_count2df(k: np.ndarray, og_names: list[str],  num_transition: np.ndarray, 
                        num_transition_query: np.ndarray, N: int, args) -> pl.DataFrame:
    if num_transition_query:
        return pl.DataFrame({'OG1':[args.query]*len(og_names), 'OG2':og_names,
                             'k':k, 'num_change1':[num_transition_query]*len(og_names), 
                             'num_change2':num_transition.flatten(), 'N':[N]*len(og_names)})
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
        k = pl.DataFrame({'k':k})
        result = pl.concat([indices, k], how='horizontal')
        df = result.with_columns(N).rename({'literal':'N'})

    return df


def run_sev(args):
    trees = list_asr_trees(args)
    with Pool(processes=args.cores) as process:
        result = process.starmap_async(count_change, trees)
        count = result.get()
    tree = et.Tree(args.tree, format=1)
    num_internal_nodes = len(tree.get_leaves()) - 1
    og_names, df, df_T, num_transition, num_transition_query = prepare_matrix(count, args)
    k = calculate_k(df, df_T, gpu=args.gpu, num_blocks=args.num_blocks)
    result = transition_count2df(k, og_names, num_transition, 
                                 num_transition_query, num_internal_nodes, args)
    
    return result


def count_change(tree: et.TreeNode, og: str
                 ) -> tuple[str, NDArray[np.int64], int]:
    tree = et.Tree(tree, format=1)
    transition = []
    for node in tree.traverse():
        if not node.is_leaf():
            parent_state = getattr(node, og)
            for child in node.get_children():
                child_state = getattr(child, og)
                try:
                    transition.append(int(float(child_state)) - int(float(parent_state)))
                except ValueError:
                    transition.append(0)

    num_transition = np.count_nonzero(transition)

    return og, transition, num_transition


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


