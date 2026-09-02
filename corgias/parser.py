#!/usr/bin/env python

import argparse
from corgias.config import CUPY_AVAILABLE


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a valid positive integer")
    return ivalue

def non_negative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"{value} is not a valid non-negative integer")
    return ivalue

def valid_float(value: str):
    fvalue = float(value)
    if not (0 < fvalue <= 1):
        raise argparse.ArgumentTypeError(f"{value} is not a valid float. It should be larger than 0 and smaller than 1")
    return fvalue

def parse_arguments():
    parent_parser = argparse.ArgumentParser(
        add_help=False,
        description = 'CORGIAS'
    )
    parent_parser.add_argument("-l", "--log-file", default=None)
    parent_parser.add_argument("--verbose", action="store_true")
    parent_parser.add_argument("--quiet", action="store_true")

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
        '\tThe ortholog table should be a CSV file but each ortholog is assmued to be evolved independently. \n' \
        '\tExample usage:\n' \
        '\t\tcorgias asr -t tree.nwk -d orthologs.csv -i 0 --work_dir pastml_result -c 4 --prediction_method ML\n\n'\
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
    asr_parser.add_argument('-m', '--prediction_method', choices=['MPPA','MAP','JOINT','DOWNPASS','ACCTRAN',
                                                                  'DELTRAN', 'ML', 'MP'],
                            default='ML')
    asr_parser.add_argument('-d', '--data', required=True)
    asr_parser.add_argument('-i', '--id_index', default=0, type=int)
    asr_parser.add_argument('--work_dir', required=True)
    asr_parser.add_argument('-c', '--cores', default=1)
    asr_parser.add_argument('--test', type=positive_int, default=0)
    asr_parser.add_argument('--tmp')
    asr_parser.add_argument('--keep', action='store_true', default=False)

    profiling_parser = new_subparser(subparsers, 'profiling', profiling_description)
    profiling_parser.add_argument('-m', '--method', choices=['naive', 'rle', 'cwa', 'asa', 'cotr', 'sev'], required=True)
    profiling_parser.add_argument('-og', '--og_table')
    profiling_parser.add_argument('-og2', '--og_table2', default=None,
                                  help='Secondary OG table for cross-comparison (naive, rle, cwa, cotr)')
    profiling_parser.add_argument('-a', '--asr_folder')
    profiling_parser.add_argument('-a2', '--asr_folder2', default=None,
                                  help='Secondary ASR result folder for cross-comparison (asa, sev)')
    profiling_parser.add_argument('-o', '--output', required=True)
    profiling_parser.add_argument('-t', '--tree')
    profiling_parser.add_argument('-c', '--cores', type=positive_int, default=1)
    profiling_parser.add_argument('--ignore_branch', action='store_true', default=False)
    if CUPY_AVAILABLE:
        profiling_parser.add_argument('--gpu', action='store_true', default=False)
        profiling_parser.add_argument('-nb', '--num_blocks', type=positive_int, default=0)
    profiling_parser.add_argument('-q', '--query', default=None)
    profiling_parser.add_argument('--test', type=positive_int, default=0)
    profiling_parser.add_argument('--legacy_n', action='store_true', default=False,
                                  help='sev only: use the old dataset-wide constant N '
                                       '(number of internal nodes) instead of the per-pair '
                                       'active-node-union N. Ignored for other methods.')

    stat_parser = new_subparser(subparsers, 'stat', stat_description)
    stat_parser.add_argument('-i', '--input', required=True)
    stat_parser.add_argument('-i2', '--input2', default=None,
                             help='Secondary profiling result (raw) for combined correction with primary stat result (-i)')
    stat_parser.add_argument('-m', '--method', required=True,
                             choices=['naive', 'rle', 'cwa', 'asa', 'cotr', 'sev'])
    stat_parser.add_argument('-o', '--output', required=True)
    stat_parser.add_argument('-d', '--direction',
                             choices=['both', 'correlation', 'anti-correlation'],
                             default='both')
    stat_parser.add_argument('-c', '--cores', type=positive_int, default=1)
    stat_parser.add_argument('-t', '--threthold', type=valid_float, default=0.05)
    stat_parser.add_argument('-s', '--statistical_test',
                             choices=['bonferroni', 'sidak', 'holm-sidak', 'simes-hochberg',
                                      'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'],
                             default='fdr_bh')
    stat_parser.add_argument('--only_signif', action='store_true', default=False)
    stat_parser.add_argument('--min_k', type=non_negative_int, default=None,
                             help='Minimum number of co-transition events (|k|) required to keep a pair '
                                  '(cotr/sev only). Applied after multiple testing correction.')
    stat_parser.add_argument('--min_phi', type=float, default=None,
                             help='Minimum |phi| (effect size, independent of dataset size) required to '
                                  'keep a pair. Applied after multiple testing correction.')

    args, options = parser.parse_known_args()

    return args, options


