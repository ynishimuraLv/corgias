#!/usr/bin/env python
import logging
from pathlib import Path
from corgias.logconfig import setup_logger
import corgias.parser as parser
from corgias.asr import run_asr
from corgias.calstat import run_stat
from corgias.profiling.runner import run_profiling


def main():
    args, options = parser.parse_arguments()

    if args.log_file is None:
        if args.subparser_name == 'asr':
            args.log_file = f"{Path(args.work_dir).name}.log"
        elif args.subparser_name == 'profiling':
            args.log_file = f"{Path(args.output).stem}.log"
        elif args.subparser_name == 'stat':
            args.log_file = f"{Path(args.output).stem}.log"

    setup_logger(args.log_file, args.verbose, args.quiet)

    logging.info("CORGIAS started")

    if args.subparser_name == 'asr':
        run_asr(args, options)
    elif args.subparser_name == 'profiling':
        run_profiling(args, options)
    elif args.subparser_name == 'stat':
        run_stat(args, options)


if __name__ == '__main__':
    main()
