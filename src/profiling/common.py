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


def load_og_table(args):
    df = pl.read_csv(args.og_table).to_pandas()
    index = df.columns[0]
    df.set_index(index, inplace=True)
    if args.query:
        if args.query not in df.columns:
            logger.error(f"{args.query} is not found in the ortholog table")
            raise ValueError(f"{args.query} is not found in the ortholog table")
        query = df[[args.query]]
        rest = df.drop(columns=args.query)
        df = pd.concat([query, rest], axis=1)
    if args.test != 0:
        return df.iloc[:, :args.test]
    else:
        return df


def list_asr_trees(args):
    tree_name = pathlib.Path(args.tree).stem
    tree_name = f'named.tree_{tree_name}.nwk'
    
    if args.query:
        query_tree = [(f'{args.asr_folder}/{args.query}/{tree_name}', args.query)]
        other_folders = [ folder for folder in os.listdir(args.asr_folder)
                        if folder != args.query]
        other_folders =  [ (f'{args.asr_folder}/{folder}/{tree_name}', folder)
                            for folder in other_folders 
                            if (os.path.exists(f'{args.asr_folder}/{folder}/{tree_name}'))]
        folders = query_tree + other_folders
    else:
        folders = [ (f'{args.asr_folder}/{folder}/{tree_name}', folder)
                        for folder in os.listdir(args.asr_folder)
                    if os.path.exists(f'{args.asr_folder}/{folder}/{tree_name}') ]

    if args.test != 0:
        return folders[:args.test]
    else:
        return folders


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
