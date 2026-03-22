#!/usr/bin/env python
import os
import pathlib
import sys
import logging
import ete3 as et
import numpy as np
import pandas as pd
import polars as pl
from itertools import combinations
from itertools import groupby
from numpy.typing import NDArray
from collections import Counter
from multiprocessing import Pool

from src.config import CUPY_AVAILABLE, cp


def gene_count(genes: str) -> int:
    if genes == '*' or not genes:
        num = 0
    else:
        try:
            num = genes.count(',') + 1
        except:
            num = 0
    return num


def count2bin(count: int) -> bool:
    if count >= 1:
        return True
    else:
        return False


def load_og_table(args):
    df = pl.read_csv(args.og_table).to_pandas()
    index = df.columns[0]
    df.set_index(index, inplace=True)
    if args.test != 0:
        df = df.iloc[:, :args.test]
        
    return df


def run_naive(args):
    df = load_og_table(args)
    if args.gpu:
        tt, tf, ft, ff = naive_gpu(df, args.num_blocks)
    else:
        tt, tf, ft, ff = naive_cpu(df, args.cores)
    og_names = list(df.columns)
    result = naivecount2matrix(tt, tf, ft, ff, og_names)

    return result


def naive_gpu(df: pd.DataFrame, num_blocks: int = 0
              ) -> tuple[NDArray[np.int64], NDArray[np.int64],
                         NDArray[np.int64], NDArray[np.int64]]:
    df_flipped, df_T, df_T_flipped = prepare_matrices(df)
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


def prepare_matrices(df: pd.DataFrame
                     ) -> tuple[pd.DataFrame, pd.DataFrame,
                                pd.DataFrame]:
    df_flipped = df.replace({0:1, 1:0})
    df_T = df.T
    df_T_flipped = df_T.replace({0:1, 1:0})

    return df_flipped, df_T, df_T_flipped


def naive_cpu(df: pd.DataFrame, cores: int) -> tuple[int, int, int, int]:
    df_flipped, df_T, df_T_flipped = prepare_matrices(df)
    jobs = [(df_T, df), (df_T, df_flipped),
            (df_T_flipped, df), (df_T_flipped, df_flipped)]
    if cores > 4:
        cores = 4
    with Pool(processes=cores) as process:
        tt, tf, ft, ff = process.starmap(np.dot, jobs)

    return tt, tf, ft, ff


if CUPY_AVAILABLE:
    def block_dot(df1: cp.ndarray, df2: cp.ndarray, block_size: int):
        M, K = df1.shape
        _, N = df2.shape
        result = np.zeros((M, N), dtype=np.int32)
        for i in range(0, M, block_size):
            for j in range(0, N, block_size):
                block_C = cp.zeros((block_size, block_size), dtype=cp.int32)
                for k in range(0, K, block_size):
                    block_C += cp.dot(
                        cp.asarray(df1[i:i+block_size, k:k+block_size], dtype=cp.int32),
                        cp.asarray(df2[k:k+block_size, j:j+block_size], dtype=cp.int32)
                    )
                result[i:i+block_size, j:j+block_size] += cp.asnumpy(block_C)

        return result


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


def uppermatrix2vector(matrix: NDArray[np.int64]):
    rows, _ = matrix.shape
    upper_triangle_indices = np.triu_indices(rows, k=1)
    upper_triangle = matrix[upper_triangle_indices]
    return upper_triangle


def flatten_indices(df: NDArray[np.int64]) -> pl.DataFrame:
    rows, _ = df.shape
    upper_indices = np.triu_indices(rows, k = 1)
    indices = pl.DataFrame(np.vstack(upper_indices).T)

    return indices


class RLE_CWA:
    def __init__(self, df: pd.DataFrame, method: str,
                 tree: et.Tree, cores: int):
        self.df = df.map(count2bin)
        self.method = method
        self.tree = tree
        self.cores = cores
        if method == 'rle':
            order = [ leaf.name for leaf in self.tree.get_leaves() ]
            self.df = self.df.loc[order]


    def rle(self, og1: str, og2: str
            ) -> tuple[str, str, int, int, int, int]:
        z = self.convert2traits(og1, og2)
        reduction = [ i[0] for i in groupby(z) ]
        count = Counter(reduction)
        return og1, og2, count[1], count[2], count[3], count[0]


    def convert2traits(self, og1: str, og2: str
                       ) -> pd.Series: # pd.Series[int]
        z = pd.Series(np.zeros(self.df.shape[0]), index=self.df.index, dtype=int)
        col1 = self.df.loc[:, og1]
        col2 = self.df.loc[:, og2]
        z[col1 & col1] = 1
        z[col1 & ~col2] = 2
        z[~col1 & col2] = 3
        return z


    def cwa(self, og1: str, og2: str
            ) -> tuple[str, str, int, int, int, int]:
        z = self.convert2traits(og1, og2)
        for leaf in self.tree.get_leaves():
            leaf.trait = str(z[leaf.name])

        remove = set()
        for node in self.tree.traverse(strategy='postorder'):
            if not node.is_leaf():
                child1, child2 = node.get_children()
                if child1.trait == child2.trait:
                    node.trait = child1.trait
                else:
                    node.trait = 0
                    for child in [child1, child2]:
                        if not child.is_leaf() and child.trait != 0:
                            leaves = [ leaf.name for leaf in child.get_leaves() ][1:]
                            remove |= set(leaves)
        z = z.loc[z.index.difference(set(remove))]
        count = z.value_counts()
        return og1, og2, count.get(1, 0), count.get(2, 0), count.get(3, 0), count.get(0, 0)


    def run_paralell(self):
        pairs = combinations(self.df.columns, 2)
        if self.method == 'rle':
            order = [ leaf.name for leaf in self.tree.get_leaves() ]
            self.df = self.df.loc[order]
            run_method = self.rle
        elif self.method == 'cwa':
            self.tree.resolve_polytomy()
            run_method = self.cwa

        with Pool(processes=self.cores) as process:
            result = process.starmap_async(run_method, pairs).get()

        return result


weighted_schema = { "OG1": pl.Utf8, "OG2": pl.Utf8,
                    "TT": pl.Float64, "TF": pl.Float64,
                    "FT": pl.Float64, "FF": pl.Float64 }

def run_rle_cwa(args):
    df = load_og_table(args)
    tree = et.Tree(args.tree, format=1)
    profiler = RLE_CWA(df, args.method, tree, cores=args.cores)
    result = profiler.run_paralell()
    return pl.DataFrame(result, schema = weighted_schema, orient='row')


def asa(tree_og1: tuple[str, str],
        tree_og2: tuple[str, str],
        ignore_branch: bool = False):
    tree1 = et.Tree(tree_og1[0], format=1)
    tree2 = et.Tree(tree_og2[0], format=1)
    og1 = tree_og1[1]
    og2 = tree_og2[1]
    merged_tree = merge_tree(tree1, og1, tree2, og2)
    if ignore_branch:
        result = count_by_ancestral_state(merged_tree)
    else:
        result = correct_by_ancestral_state(merged_tree)

    return og1, og2, result['1'], result['2'], result['3'], result['0']


def merge_tree(tree1: et.Tree, og1: str,
               tree2: et.Tree, og2: str):
    tree = tree1.copy()
    for node, node1, node2 in zip(tree.traverse(),
                                  tree1.traverse(),
                                  tree2.traverse()):
        node.trait = mix_trait(getattr(node1, og1), getattr(node2, og2))
    return tree


def mix_trait(og1: str, og2: str):
#    if og1 is None or og2 is None:
#        return '4'
    if og1 == '0' and og2 == '0':
        return '0'
    elif og1 == '1' and og2 == '1':
        return '1'
    elif og1 == '1' and og2 == '0':
        return '2'
    elif og1 == '0' and og2 == '1':
        return '3'
    else:
        return '4'


def correct_genomes(node: et.Tree) -> float:
    if node.num_child == 1:
        return 1
    else:
        return node.num_child * (node.pathlength / node.denominator)


def count_by_ancestral_state(tree: et.Tree):
    result = { str(i):0 for i in range(4) }
    for node in tree.traverse(strategy='postorder'):
        node.state = node.trait
        if node.is_leaf():
            node.num_child = 1
        else:
            node.num_child = 0
            if node.state in '0123':
                for child in node.get_children():
                    if node.state == child.state:
                        node.num_child = 1
                    else:
                        if child.num_child:
                            result[child.state] += 1
            else:
                for child in node.get_children():
                    if child.num_child:
                        result[child.state] += 1

    if tree.num_child:
        result[tree.state] += 1

    return result


def correct_by_ancestral_state(tree: et.Tree):
    result = { str(i):0 for i in range(4) }
    for node in tree.traverse(strategy='postorder'):
        node.state = node.trait
        if node.is_leaf():
            node.num_child = 1
            node.pathlength = node.dist
            node.denominator = node.dist
        else:
            node.num_child = 0
            node.pathlength = 0
            node.denominator = 0
            if node.state in '0123':
                for child in node.get_children():
                    if node.state == child.state:
                        node.num_child += child.num_child
                        node.pathlength += child.pathlength
                        node.denominator += child.denominator
                    else:
                        if child.num_child:
                            result[child.state] += correct_genomes(child)
                if node.num_child:
                    node.denominator += node.dist * node.num_child
                    node.pathlength += node.dist
            else:
                for child in node.get_children():
                    if child.num_child:
                        result[child.state] += correct_genomes(child)

    if tree.num_child:
        tree.denominator += tree.dist * tree.num_child
        tree.pathlength += tree.dist
        result[tree.state] += correct_genomes(tree)

    return result


def prepare_trees(args):
    tree_name = pathlib.Path(args.tree).stem
    tree_name = 'named.tree_' + tree_name + '.nwk'
    trees = ((f'{args.asr_folder}/{folder}/{tree_name}', folder)
            for folder in os.listdir(args.asr_folder)
            if os.path.exists(f'{args.asr_folder}/{folder}/{tree_name}'))
    if args.test != 0:
        trees = list(trees)[:args.test]

    return trees


def run_asa(args):
    trees = prepare_trees(args)
    pairs = ((tree1, tree2, args.ignore_branch) for tree1, tree2
            in combinations(trees, 2))
    with Pool(processes=args.cores) as process:
        result = pl.DataFrame(process.starmap_async(asa, pairs).get(),
                            schema = weighted_schema,
                            orient='row')

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


def run_transition(count, gpu: bool, num_blocks: int, N: int):
    og_names = [ sublist[0] for sublist in count ]
    t_matrix = np.vstack([ sublist[1] for sublist in count ])
    num_transition = np.vstack([ sublist[2] for sublist in count ])
    k = calculate_k(t_matrix, gpu, num_blocks)
    result = transition_count2df(k, num_transition, og_names, N)

    return result


def run_cotr(args):
    df = load_og_table(args)
    tree = et.Tree(args.tree, format=1)
    order = [ leaf.name for leaf in tree.get_leaves() ]
    df = df.loc[order]
    ogs = ((i, row) for i, row in df.T.iterrows())
    with Pool(processes=args.cores) as process:
        count = process.starmap_async(count_transition, ogs).get()
    num_genomes = len(order) - 1
    
    return run_transition(count, args.gpu, args.num_blocks, num_genomes)


def transition_count2df(k: NDArray[np.int64], num_transition: int, og_names: list[str], N: int
                 ) -> pl.DataFrame:
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


def count_change(tree: et.TreeNode, og: str
                 ) -> tuple[str, NDArray[np.int64], int]:
    tree = et.Tree(tree, format=1)
    transition = []
#    internal_node = []
    for node in tree.traverse():
        if not node.is_leaf():
            parent_state = getattr(node, og)
            for child in node.get_children():
                child_state = getattr(child, og)
                try:
                    transition.append(int(float(child_state)) - int(float(parent_state)))
                except ValueError:
                    transition.append(0)
#            internal_node.append(transition[-1] + transition[-2])

#    num_transition = np.count_nonzero(internal_node)
    num_transition = np.count_nonzero(transition)

    return og, transition, num_transition


def calculate_k(t_matrix: pl.DataFrame, gpu: bool = False,
                num_blocks: int = 0) -> NDArray[np.int64]:
    if gpu:
        if num_blocks == 0:

            df = cp.asarray(t_matrix, dtype=cp.int16)
            df_T = cp.asarray(t_matrix.transpose(), dtype=cp.int16)
            k = cp.asnumpy(cp.dot(df, df_T))
        else:
            block_size = t_matrix.shape[0] // num_blocks
            k = block_dot(t_matrix, t_matrix.transpose(), block_size)
    else:
        k = np.dot(t_matrix, t_matrix.transpose())

    return k


def run_sev(args):
    trees = prepare_trees(args) 
    with Pool(processes=args.cores) as process:
        result = process.starmap_async(count_change, trees)
        count = result.get()
    tree = et.Tree(args.tree, format=1)
    num_internal_nodes = len(tree.get_leaves()) - 1
    return run_transition(count, args.gpu, args.num_blocks,
                            num_internal_nodes)


def validate_args(args):
    if args.method == 'naive' and not args.og_table:
        print('An ortholog table is required when using naive method',
            file=sys.stderr)
        sys.exit(1)
    elif args.method in ['rle', 'cwa']:
        if (not args.tree) or (not args.og_table):
            print('An ortholog table and a phylogenetic tree are '
                f'required when using {args.method} method',
                file=sys.stderr)
            sys.exit(1)
    elif args.method in ['asa', 'sev'] and not args.asr_folder:
        print('The results of ancestral state reconstruction are '
            f'required when using {args.method} method',
            file=sys.stderr)
    elif args.method in ['asa', 'sev'] and not args.tree:
        print('A phylogenetic tree is required '
            f'when using {args.method} method',
            file=sys.stderr)
        sys.exit(1)


logger = logging.getLogger(__name__)
METHOD_RUNNERS = {
    "naive": run_naive,
    "rle": run_rle_cwa,
    "cwa": run_rle_cwa,
    "cotr": run_cotr,
    "asa": run_asa,
    "sev": run_sev,
}

def run_profiling(args, options):
    logger.info("Starting phylogenetic profiling")
    if not CUPY_AVAILABLE:
        args.gpu = False
        args.num_blocks = 0
    print('test')
    validate_args(args)

    method_runner = METHOD_RUNNERS[args.method]
    result = method_runner(args)

    result.write_csv(args.output)
