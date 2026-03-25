#!/usr/bin/env python
import logging
from src.logconfig import setup_logger
import src.parser as parser
from src.asr import run_asr
from src.calstat import run_stat
from src.profiling.runner import run_profiling


def main():
    args, options = parser.parse_arguments()
    setup_logger(args.log_file, args.verbose, args.quiet)

    logging.info("CORGIAS started")

    if args.subparser_name == 'asr':
        run_asr(args, options)
    elif args.subparser_name == 'profiling':
        run_profiling(args, options)
    elif args.subparser_name == 'stat':
        run_stat.run_stat(args, options)


if __name__ == '__main__':
    main()
