import os
import logging
import pathlib
import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import NDArray

logger = logging.getLogger(__name__)
weighted_schema = { "OG1": pl.Utf8, "OG2": pl.Utf8,
                    "TT": pl.Float64, "TF": pl.Float64,
                    "FT": pl.Float64, "FF": pl.Float64 }

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


def load_og_table(path, query=None):
    df = pl.read_csv(path).to_pandas()
    index = df.columns[0]
    df.set_index(index, inplace=True)
    if query:
        if query not in df.columns:
            logger.error(f"{query} is not found in the ortholog table")
            raise ValueError(f"{query} is not found in the ortholog table")
        q = df[[query]]
        rest = df.drop(columns=query)
        df = pd.concat([q, rest], axis=1)
    return df


def list_asr_trees(asr_folder, tree, query=None):
    tree_name = f'named.tree_{pathlib.Path(tree).stem}.nwk'
    if query:
        query_tree = [(f'{asr_folder}/{query}/{tree_name}', query)]
        other_folders = [f for f in os.listdir(asr_folder) if f != query]
        other_folders = [(f'{asr_folder}/{f}/{tree_name}', f)
                         for f in other_folders
                         if os.path.exists(f'{asr_folder}/{f}/{tree_name}')]
        return query_tree + other_folders
    else:
        return [(f'{asr_folder}/{f}/{tree_name}', f)
                for f in os.listdir(asr_folder)
                if os.path.exists(f'{asr_folder}/{f}/{tree_name}')]


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
