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
        outfile = file.split('/')[-1]
        command = self.command + ['-d', file, '--work_dir', f'{self.work_dir}/{outfile}']
        result = subprocess.run(command, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr


    def run_parallel(self, files: list[str]):
        with Pool(processes=self.cores) as process:
            process.starmap(self.run_pastml, files)


logger = logging.getLogger(__name__)

def run_asr(args, options):
    logger.info("Starting ancestral state reconstruction")
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

    asr_runner = PastMLRunner(args.tree, args.prediction_method, args.work_dir, args.cores)
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
