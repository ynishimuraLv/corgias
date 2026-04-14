import logging
import math
import numpy as np
import pandas as pd
import polars as pl
import ete3 as et
from itertools import groupby, combinations
from collections import Counter
from multiprocessing import Pool
from tqdm import tqdm
from .common import load_og_table, count2bin, weighted_schema

logger = logging.getLogger(__name__)


class RLE_CWA:
    def __init__(self, df: pd.DataFrame, method: str,
                 tree: et.Tree, cores: int, query: str, quiet: bool = False):
        self.df = df.map(count2bin)
        self.method = method
        self.tree = tree
        self.cores = cores
        self.query = query
        self.quiet = quiet
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
        if self.query:
            pairs = [(self.query, og) for og in self.df.columns if og != self.query]
        else:
            pairs = combinations(self.df.columns, 2)
        n = len(self.df.columns)
        num_pairs = n - 1 if self.query else math.comb(n, 2)
        logger.info(f"Processing {num_pairs} OG pairs.")
        if self.method == 'rle':
            order = [ leaf.name for leaf in self.tree.get_leaves() ]
            self.df = self.df.loc[order]
            run_method = self.rle
        elif self.method == 'cwa':
            self.tree.resolve_polytomy()
            run_method = self.cwa

        with Pool(processes=self.cores) as pool:
            with tqdm(total=num_pairs, disable=self.quiet) as pbar:
                futures = [pool.apply_async(run_method, pair, callback=lambda _: pbar.update())
                           for pair in pairs]
                result = [f.get() for f in futures]

        return result


def run_rle_cwa(args):
    df = load_og_table(args)
    tree = et.Tree(args.tree, format=1)
    profiler = RLE_CWA(df, args.method, tree, cores=args.cores, query=args.query, quiet=args.quiet)
    result = profiler.run_paralell()
    return pl.DataFrame(result, schema = weighted_schema, orient='row')