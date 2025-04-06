#!/usr/bin/env python

import subprocess
from multiprocessing import Pool

class PastMLRunner:
    def __init__(self, tree: str, separator: str, work_dir: str, cores: int):
        self.tree = tree
        self.separator = separator
        self.work_dir = work_dir
        self.cores = int(cores)
        self.command = ['pastml']
        
        
    def set_pastml_command(self, options: list[str]):
        self.command += ['-t', self.tree, '-s', self.separator] + options
    
    
    def run_pastml(self, file: str):
        outfile = file.split('/')[-1]
        command = self.command + ['-d', file, '--work_dir', f'{self.work_dir}/{outfile}']
        print(command)
        result = subprocess.run(command, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    
    
    def run_parallel(self, files: list[str]):
        with Pool(processes=self.cores) as process:
            process.starmap(self.run_pastml, files)