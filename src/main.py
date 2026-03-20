#!/usr/bin/env python

import src.asr as asr
import src.calstat as calstat
import src.parser as parser
import src.profiling as profiling


def main():
    args, options = parser.parse_arguments()

    if args.subparser_name == 'asr':
        asr.run_asr(args, options)
    elif args.subparser_name == 'profiling':
        profiling.run_profiling(args, options)
    elif args.subparser_name == 'stat':
        calstat.run_stat(args, options)


if __name__ == '__main__':
    main()
