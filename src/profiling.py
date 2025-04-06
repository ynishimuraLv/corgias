#!/usr/bin/env python
import os
import sys
from numpy.typing import NDArray
import numpy as np
import cupy as cp
import pandas as pd
import polars as pl
from multiprocessing import Pool
from collections import Counter
from itertools import groupby
from itertools import combinations
import ete3 as et


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


def run_naive(df: pd.DataFrame, gpu: bool, num_blocks: int, cores: int):
    if gpu:
        tt, tf, ft, ff = naive_gpu(df, num_blocks)
    else:
        tt, tf, ft, ff = naive_cpu(df, cores)
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
        self.df = df.applymap(count2bin)
        self.method = method
        self.tree = tree
        self.cores = cores


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


def asa(tree_og1: tuple[str, str],
        tree_og2: tuple[str, str],
        ignore_branch: bool = False):
    tree1 = et.Tree(tree_og1[0], format=1)
    tree2 = et.Tree(tree_og2[0], format=1)
    og1 = tree_og1[1]
    og2 = tree_og2[1]
    merged_tree = merge_tree(tree1, og1, tree2, og2)
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
             ).rename({'num_transition':'num_transition1'}).join(
             num_transition, left_on='column_1', right_on='index'
             ).rename({'num_transition':'num_transition2'}).select(
             ['OG1', 'OG2', 'num_transition1', 'num_transition2']
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


