#!/usr/bin/env python
import sys
import pathlib
import shutil
import subprocess
import logging
import pandas as pd
import polars as pl
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm



class PastMLRunner:
    def __init__(self, tree: str, method: str, work_dir: str, cores: int):
        self.tree = tree
        if method == 'ML':
            self.method = 'MPPA'
        elif method == 'MP':
            self.method = 'ACCTRAN'
        else:
            self.method = method
        self.work_dir = work_dir
        self.cores = int(cores)
        self.command = ['pastml']

    def set_pastml_command(self, options: list[str]):
        self.command += ['-t', self.tree, '-s', ",", '--prediction_method', self.method] + options


    def run_pastml(self, file: str):
        logger = logging.getLogger(__name__)
        
        outfile = file.split('/')[-1]
        command = self.command + ['-d', file, '--work_dir', f'{self.work_dir}/{outfile}']
        logger.debug(f"Running Ancestral state reconstruction of {file}")
        logger.debug(f"Executing: {' '.join(command)}")

        result = subprocess.run(command,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        logger.debug(f"PastML finished with return code {result.returncode} for {file}")
        return result.returncode, result.stdout, result.stderr


    def run_parallel(self, files: list[str], quiet: bool = False):
        logger = logging.getLogger(__name__)
        logger.info(f'Starting parallel ASR with {self.cores} cores for {len(files)} jobs.')
        with Pool(processes=self.cores) as pool:
            with tqdm(total=len(files), disable=quiet) as pbar:
                futures = [pool.apply_async(self.run_pastml, f, callback=lambda _: pbar.update())
                           for f in files]
                [f.get() for f in futures]


logger = logging.getLogger(__name__)

def run_asr(args, options):
    logger.info("Starting ancestral state reconstruction")
    df = pl.read_csv(args.data).to_pandas()
    index_col = df.columns[int(args.id_index)]
    df.set_index(index_col, inplace=True)
    logger.info(f'Loaded data with {df.shape[0]} rows and {df.shape[1]} columns.')
    df_type_check = df.dtypes.apply(pd.api.types.is_integer_dtype)
    if not df_type_check.all():
        non_int_cols = df_type_check[~df_type_check].index.tolist()
        logger.error(f'Input data includes non Integer columns: {non_int_cols}')
        raise ValueError('Input data must be integer')
  
    df = df.where(df == 0, 1)
    if args.test:
        logger.info(f'Test mode: using first {args.test} columns.')
        df = df.iloc[:, :int(args.test)]

    if not args.tmp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmpdir = pathlib.Path(f'tmp_{timestamp}')
    else:
        tmpdir = pathlib.Path(args.tmp)
    tmpdir.mkdir()
    logger.debug(f'Temporary directory created at {tmpdir.resolve()}')

    pathlib.Path(args.work_dir).mkdir(exist_ok=True, parents=True)
    jobs: list[str] = []
    for col in df.columns:
        file = str(tmpdir.joinpath(col))
        df.loc[:, col].to_csv(file)
        jobs.append((file, ))
    logger.debug(f'Prepared {len(jobs)} jobs for ASR.')

    asr_runner = PastMLRunner(args.tree, args.prediction_method, args.work_dir, args.cores)
    asr_runner.set_pastml_command(options)
    firstfile = jobs[0][0]
    logger.info('Running pastml check with the first data.')
    returncode, stdout, stderr = asr_runner.run_pastml(firstfile)

    if returncode != 0:
        logger.error('PastML failed.')
        logger.error(f'stdout:\n{stdout}\n')
        logger.error(f'stderr:\n{stderr}\n')
        
        if not args.keep:
            shutil.rmtree(tmpdir)
            logger.debug('Temporary directory removed.')
        
        raise RuntimeError(
            f'PastMl failed\n'
            f'Your command was interpreted as:\n'
            f' {" ".join(asr_runner.command + ["-d", firstfile])}\n'
            f'Please check the error message above and see options by pastml --help'
        )
    else:
        logger.info('PastML check passed successfully.')

    logger.info(f'Running ASR in parrallel with {args.cores} CPUs.')
    asr_runner.run_parallel(jobs, quiet=args.quiet)

    logger.info('ASR completed successfully.')
    
    if not args.keep:
        shutil.rmtree(tmpdir)
        logger.debug('Temporary directory removed.')