#!/usr/bin/env python

import os
import pathlib
import shutil
import sys
from datetime import datetime
from itertools import combinations
from multiprocessing import Pool

import ete3 as et
import pandas as pd
import polars as pl
from statsmodels.stats.multitest import multipletests

import src.asr as asr
import src.calstat as calstat
import src.parser as parser
import src.profiling as profiling

try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def run_asr(args, options):
    df = pl.read_csv(args.data).to_pandas()
    index_col = df.columns[int(args.id_index)]
    df.set_index(index_col, inplace=True)
    df_type_check = df.dtypes.apply(pd.api.types.is_integer_dtype).all()
    if not df_type_check:
        sys.exit('Input data includes non Integer columns')
    else:
        df = df.where(df == 0, 1)
        if args.test:
            df = df.iloc[:, :int(args.test)]

    if not args.tmp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmpdir = pathlib.Path(f'tmp_{timestamp}')
    else:
        tmpdir = pathlib.Path(args.tmp)
    tmpdir.mkdir()

    pathlib.Path(args.work_dir).mkdir(exist_ok=True, parents=True)
    jobs: list[str] = []
    for col in df.columns:
        file = str(tmpdir.joinpath(col))
        df.loc[:, col].to_csv(file)
        jobs.append((file, ))

    asr_runner = asr.PastMLRunner(args.tree, args.prediction_method, args.work_dir, args.cores)
    asr_runner.set_pastml_command(options)
    firstfile = jobs[0][0]
    returncode, _, _ = asr_runner.run_pastml(firstfile)

    if returncode != 0:
        if not args.keep:
            shutil.rmtree(tmpdir)
        print('Something went wrong with pastml. See options by pastml --help')
        sys.exit(f'''
                    Your command was interpreted as:
                     {asr_runner.command + ['-d', firstfile]}
                 ''')

    asr_runner.run_parallel(jobs)
    if not args.keep:
        shutil.rmtree(tmpdir)


def run_profiling(args, options):
    if not CUPY_AVAILABLE:
        args.gpu = False
        args.num_blocks = 0

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

    weighted_schema = { "OG1": pl.Utf8, "OG2": pl.Utf8,
                        "TT": pl.Float64, "TF": pl.Float64,
                        "FT": pl.Float64, "FF": pl.Float64
                    }

    if args.method in ['naive', 'rle', 'cwa', 'cotr']:
        df = pl.read_csv(args.og_table).to_pandas()
        index = df.columns[0]
        df.set_index(index, inplace=True)
        if args.test != 0:
            df = df.iloc[:, :args.test]

        # df = df.applymap(count2bin) フォーマットチェックは一旦置いておく
        if args.method == 'naive':
            result = profiling.run_naive(df, args.gpu, args.num_blocks, args.cores)

        elif args.method in ['rle', 'cwa']:
            tree = et.Tree(args.tree, format=1)
            profiler = profiling.RLE_CWA(df, args.method, tree, cores=args.cores)
            result = profiler.run_paralell()
            result = pl.DataFrame(result, schema = weighted_schema,
                                orient='row')

        elif args.method == 'cotr':
            tree = et.Tree(args.tree, format=1)
            order = [ leaf.name for leaf in tree.get_leaves() ]
            df = df.loc[order]
            ogs = ((i, row) for i, row in df.T.iterrows())
            with Pool(processes=args.cores) as process:
                count = process.starmap_async(profiling.count_transition, ogs).get()

            num_genomes = len(order) - 1

            result = profiling.run_transition(count, args.gpu, args.num_blocks, num_genomes)

    else: #elif args.method == 'asa' or args.method == 'sev':
        tree_name = pathlib.Path(args.tree).stem
        tree_name = 'named.tree_' + tree_name + '.nwk'
        trees = ((f'{args.asr_folder}/{folder}/{tree_name}', folder)
                for folder in os.listdir(args.asr_folder)
                if os.path.exists(f'{args.asr_folder}/{folder}/{tree_name}'))
        if args.test != 0:
            trees = list(trees)[:args.test]

        if args.method == 'asa':
            pairs = ((tree1, tree2, args.ignore_branch) for tree1, tree2
                    in combinations(trees, 2))
            with Pool(processes=args.cores) as process:
                result = pl.DataFrame(process.starmap_async(profiling.asa, pairs).get(),
                                    schema = weighted_schema,
                                    orient='row')

        else: #  args.method == 'sev':
            with Pool(processes=args.cores) as process:
                result = process.starmap_async(profiling.count_change, trees)
                count = result.get()
            tree = et.Tree(args.tree, format=1)
            num_internal_nodes = len(tree.get_leaves()) - 1
            result = profiling.run_transition(count, args.gpu, args.num_blocks,
                                                num_internal_nodes)

    result.write_csv(args.output)


def run_stat(args, options):
    weighted_method = ['naive', 'rle', 'cwa', 'asa']
    transition_method = ['cotr', 'sev']

    df = pl.read_csv(args.input)
    if args.method in weighted_method:
        if args.direction == 'correlation':
            direction = 'greater'
        elif args.direction == 'anti-correlation':
            direction = 'less'
        else:
            direction = 'two-sided'
        df = df.with_columns(pl.lit(direction).alias('alternative'))
        rows = df.select('OG1', 'OG2', 'TT', 'TF', 'FT', 'FF', 'alternative').iter_rows()
        with Pool(processes=args.cores) as process:
            result = pl.DataFrame(process.starmap_async(calstat.run_test4weighted, rows).get(),
                                    schema=['OG1', 'OG2', 'odds', 'pvalue'], orient='row')
    elif args.method in transition_method:
        if args.direction == 'correlation':
            df = df.filter(pl.col('k') > 0)
        elif args.direction == 'anti-correlation':
            df = df.filter(pl.col('k') < 0)
        rows = df.iter_rows()
        with Pool(processes=args.cores) as process:
            result = pl.DataFrame(process.starmap_async(calstat.run_test4transition, rows).get(),
                                    schema=['OG1', 'OG2', 'direction', 'pvalue'],
                                    orient='row')

    qvalues = multipletests(result['pvalue'], method=args.statistical_test,
                            alpha=args.threthold)
    result = result.with_columns(pl.Series('qvalue', qvalues[1]),
                                    pl.Series('signif', qvalues[0]))
    if args.only_signif:
        result = result.filter(pl.col('signif'))
    result = result.sort(by='qvalue')
    result.write_csv(args.output)


def main():
    args, options = parser.parse_arguments()

    if args.subparser_name == 'asr':
        run_asr(args, options)
    elif args.subparser_name == 'profiling':
        run_profiling(args, options)
    elif args.subparser_name == 'stat':
        run_stat(args, options)


if __name__ == '__main__':
    main()
