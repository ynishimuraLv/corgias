#!/usr/bin/env python

import os
import sys
import shutil
import pathlib
import argparse
import pandas as pd
import polars as pl
from datetime import datetime
from itertools import combinations
from multiprocessing import Pool
from statsmodels.stats.multitest import multipletests

import ete3 as et
import src.asr as asr
import src.profiling as profiling
import src.calstat as calstat

try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

def main():

    parent_parser = argparse.ArgumentParser(
        add_help=False,
        description = 'CORGIAS'
    )
    
    parser = argparse.ArgumentParser(parents=[parent_parser])
    subparsers = parser.add_subparsers(title='Sub-commands', dest='subparser_name',
                                       parser_class = argparse.ArgumentParser)
    
    subparser_name2parser = {}
    
    def new_subparser(subparsers, parser_name, parser_description):
        subpar = subparsers.add_parser(parser_name, description = parser_description,
                                       help = parser_description, 
                                       formatter_class = argparse.RawTextHelpFormatter,
                                       parents=[parent_parser])
        subparser_name2parser = subpar
        return subpar
    
    asr_description = '\tPrepare trees with ancestral presence/absence states of ortholog for ASA or SEV profiling. \n' \
        '\tThe ortholog table should be a CSV-like file but each ortholog is assmued to be evolved independently. \n' \
        '\tExample usage:\n' \
        '\t\tcorgias asr -t tree.nwk -d orthologs.csv -i 0 -s "," -o pastml_result -c 4 --predictio_method ML\n\n'\
        '\tNote: Recostruction should be performed by a maximum-likelihood (DOWNPASS) and maximum-parsimony method (ACCTRAN)\n' \
        '\t      for ASA and SEV, respectively.\n'
    profiling_description = '\tPerform phylogenetic profiling using a ortholog table (naive, rle, cwa, cotr)\n' \
                            '\ta species tree (rle, cwa, cotr) and/or, ancestral state reconstruction results (asa, sev)\n' \
                            '\tExample usages: \n' \
                            '\t\tcorgias profiling -m naive -og orthologs.csv -o naive_out.csv -c -4 --gpu -nb 4\n' \
                            '\t\tcorgias profiling -m rle -og orthologs.csv -t tree.nwk -o rle_out.csv -c 4 \n' \
                            '\t\tcorgias profiling -m cwa -og orthologs.csv -t tree.nwk -o cwa_out.csv -c 4 \n' \
                            '\t\tcorgias profiling -m asa -a pastml_result_folder -t tree.nwk -o asa_out.csv -c 4 \n' \
                            '\t\tcorgias profiling -m cotr -og orthologs.csv -t tree.nwk -o cotr_out.csv -c 4 \n' \
                            '\t\tcorgias profiling -m sev --a pastml_result_folder -t tree.nwk -o sev_out.csv -c 4 \n\n' \
                            '\tNote: with --test 5, Run test will start using five orthologs. \n'
    stat_description = '\tConduct statistical tests for phylogenetic profiling results.\n' \
                       '\tExample usage:\n' \
                       '\t\tcorgias stat -i profiling_result.csv -m naive -o stat_out.csv -c 4 \n' 

    asr_parser = new_subparser(subparsers, 'asr', asr_description)
    asr_parser.add_argument('-t', '--tree', required=True)
    asr_parser.add_argument('-d', '--data', required=True)
    asr_parser.add_argument('-i', '--id_index', default=0)
    asr_parser.add_argument('-s', '--separator', default=',')
    asr_parser.add_argument('--work_dir', required=True)
    asr_parser.add_argument('-c', '--cores', default=1)
    asr_parser.add_argument('--test', type=int, default=0)
    asr_parser.add_argument('--tmp')
    asr_parser.add_argument('--keep', action='store_true', default=False)
    
    profiling_parser = new_subparser(subparsers, 'profiling', profiling_description)
    profiling_parser.add_argument('-m', '--method', choices=['naive', 'rle', 'cwa', 'asa', 'cotr', 'sev'], required=True)
    profiling_parser.add_argument('-og', '--og_table')
    profiling_parser.add_argument('-a', '--asr_folder')
    profiling_parser.add_argument('-o', '--output', required=True)
    profiling_parser.add_argument('-t', '--tree')
    profiling_parser.add_argument('-c', '--cores', type=int, default=1)
    profiling_parser.add_argument('--ignore_branch', action='store_true', default=False)
    if CUPY_AVAILABLE:
        profiling_parser.add_argument('--gpu', action='store_true', default=False)
        profiling_parser.add_argument('-nb', '--num_blocks', type=int, default=0)
    profiling_parser.add_argument('--test', type=int, default=0)
    
    stat_parser = new_subparser(subparsers, 'stat', stat_description)
    stat_parser.add_argument('-i', '--input', required=True)
    stat_parser.add_argument('-m', '--method', required=True,
                             choices=['naive', 'rle', 'cwa', 'asa', 'cotr', 'sev'])
    stat_parser.add_argument('-o', '--output')
    stat_parser.add_argument('-d', '--direction',
                             choices=['both', 'correlation', 'anti-correlation'],
                             default='both')
    stat_parser.add_argument('-c', '--cores', type=int, default=1)
    stat_parser.add_argument('-t', '--threthold', type=float, default=0.05)
    stat_parser.add_argument('-s', '--statistical_test',
                             choices=['bonferroni', 'sidak', 'holm-sidak', 'simes-hochberg',
                                      'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'],
                             default='fdr_bh')
    stat_parser.add_argument('--only_signif', action='store_true', default=False)

    args, options = parser.parse_known_args()
    
    if args.subparser_name == 'asr':
        df = pl.read_csv(args.data, separator=args.separator).to_pandas()
        index_col = df.columns[args.id_index]
        df.set_index(index_col, inplace=True)
        df_type_check = df.dtypes.apply(pd.api.types.is_integer_dtype).all()
        if not df_type_check:
            sys.exit('Input data includes non Integer columns')
        else:
            df = df.where(df == 0, 1)
            if args.test:
                df = df.iloc[:, :args.test]
        
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
        
        asr_runner = asr.PastMLRunner(args.tree, args.separator, args.work_dir, args.cores)
        asr_runner.set_pastml_command(options)
        
        returncode, _, _ = asr_runner.run_pastml(jobs.pop(0)[0])
        
        if returncode != 0:
            if not args.keep:
                shutil.rmtree(tmpdir)
            sys.exit('Something went wrong with pastml. See options by pastml --help')
            
        asr_runner.run_parallel(jobs)
        if not args.keep:
            shutil.rmtree(tmpdir)
        
    
    elif args.subparser_name == 'profiling':
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
        elif args.method in ['asa', 'sim'] and not args.asr_folder:
            print('The results of ancestral state reconstruction are '
                f'required when using {args.method} method',
                file=sys.stderr)
        elif args.method in ['asa', 'sim'] and not args.tree:
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
                 
    elif  args.subparser_name == 'stat':
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

if __name__ == '__main__':
    main()
    