#!/usr/bin/env python
import sys
import logging
from src.logconfig import setup_logger
import src.asr as asr
import src.calstat as calstat
import src.parser as parser
import src.profiling as profiling


def main():
    args, options = parser.parse_arguments()
    setup_logger(args.log_file, args.verbose, args.quiet)

    logging.info(f"CORGIAS started")

    if args.subparser_name == 'asr':
        asr.run_asr(args, options)
    elif args.subparser_name == 'profiling':
        profiling.run_profiling(args, options)
    elif args.subparser_name == 'stat':
        calstat.run_stat(args, options)


if __name__ == '__main__':
    main()
