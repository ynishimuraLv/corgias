# Corgias

<div align="center">
<p align="center">
    <img src="CORGIAS.png?raw=true?" alt="corgias-logo" width="300">
   <br> This logo was generated with the aid from AI (DALL·E).
</p>
<h1>CORGIAS</h1>
<h3>CORrelated Genes Identifier by considering Ancestral State</h3>
</div>

CORGIAS is a phylogenetic profiling tool for a large-scale dataset comprising of thousands and tens of thousands of orthologs and genomes. As co- and anti-correlated orthologs are expexted to be functionally related, CORGIAS can help functional annotation of orthologs, especially those showing no sequence similarity to functionally known genes.

## Installation

Corgias can be installed on Linux system.

### Download

```bash
git clone https://github.com/ynishimuraLv/corgias.git
cd corgias
```

### Create the virtual environment (Optional, but recommended)
``` bash
python -m venv .venv
. .venv/bin/activate
```

### For System with GPU compatible CUDA 12.x
```bash
python -m pip install .[gpu]
```

### For System without a compatible GPU (CPU-only)
```bash
python -m pip install .
```

## Usage

CORGIAS provides three subcommand.
1. **`profiling`** calculate values necessary for evaluating correlation of all ortholog pairs. Two our new profiling methods.
2. **`asr`** performs ancestral state of ortholog presence/absence for Ancestral steate adjustment (ASA) and Simultaneous Evolutionary Test (SEV).
3. **`stat`** performs statistical tests using values calculated by `profiling`.

### Ancestral state construction (ASR)

`asr` reuqires a rooted species tree in Newick format and an ortholog table in CSV or CSV-like format (separator can be sepcified with `-s/--separator` option). This subcommand acts as a wrapper for running  `pastml` in parallel.

```
$corgias asr -h
usage: corgias asr [-h] -t TREE -d DATA [-i ID_INDEX] [-s SEPARATOR] --work_dir WORK_DIR [-c CORES] [--test TEST] [--tmp TMP] [--keep]

	Prepare trees with ancestral presence/absence states of ortholog for ASA or SEV profiling.
	The ortholog table should be a CSV-like file but each ortholog is assmued to be evolved independently.
	Example usage:
		corgias asr -t tree.nwk -d orthologs.csv -i 0 -s "," -o pastml_result -c 4 --prediction_method ML
```

#### Options
| Option                  | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `-h, --help`            | Show the help message and exit.                                            |
| `-t, --tree TREE`       | Path to the species tree file (Newick format).                             |
| `-d, --data DATA`       | Path to the ortholog table file (CSV or CSV-like format).                              |
| `-i, --id_index ID_INDEX` | Column index for ortholog IDs (0-based).                                 |
| `-s, --separator SEPARATOR` | Separator for the ortholog table (default:  `,`).                          |
| `--work_dir WORK_DIR`   | Directory for output files.                                                |
| `-c, --cores CORES`     | Number of CPU cores to use.                                                |
| `--test TEST`           | Number of orthologs to process in test mode.                               |
| `--tmp TMP`             | Temporary directory for intermediate files.                               |
| `--keep`                | Keep intermediate files.                                                  |
| `--prediction_method`   | Method to                                            |

**Note**: Recostruction should be performed by a maximum-likelihood (DOWNPASS) and maximum-parsimony method (ACCTRAN) for ASA and SEV, respectively.
In addition to the above, `pastml` options can be acceptable (for example, `--upload_to_itol`).

### Phylogenetic profiling

The `profiling` subcommand supports six methods:

1. **naive**
2. **run length encoding (rle)**
3. **cladewise adjustment (cwa)**
4. **ancestral state adjustment (asa)**
5. **cotransitions (cotr)**
6. **simultaneous evolution test (SEV)**

| Method | Required Inputs                                                                 |
|--------|---------------------------------------------------------------------------------|
| `naive` | Ortholog table (CSV format).                                                  |
| `rle`, `cwa`, `cotr` | Ortholog table (CSV format) and rooted species tree (Newick format). |
| `asa`, `sev` | Output folder from `asr` and rooted species tree (Newick format).        |

It is highly recommended to run with `--test` with a small number before running with a large dataset.
```bash
$corgias profiling -h
usage: corgias profiling [-h] -m {naive,rle,cwa,asa,cotr,sev} [-og OG_TABLE] [-a ASR_FOLDER] -o OUTPUT [-t TREE] [-c CORES] [--ignore_branch] [--gpu]  [-nb NUM_BLOCKS] [--test TEST]

	Perform phylogenetic profiling using a ortholog table (naive, rle, cwa, cotr)
	a species tree (rle, cwa, cotr) and/or, ancestral state reconstruction results (asa, sev)
	Example usages:
		corgias profiling -m naive -og orthologs.csv -o naive_out.csv -c -4 --gpu -nb 4
		corgias profiling -m rle -og orthologs.csv -t tree.nwk -o rle_out.csv -c 4
		corgias profiling -m cwa -og orthologs.csv -t tree.nwk -o cwa_out.csv -c 4
		corgias profiling -m asa -a pastml_result_folder -t tree.nwk -o asa_out.csv -c 4
		corgias profiling -m cotr -og orthologs.csv -t tree.nwk -o cotr_out.csv -c 4
		corgias profiling -m sev --a pastml_result_folder -t tree.nwk -o sev_out.csv -c 4

  Note: with --test 5, Run test will start using five orthologs.
```
#### Options
| Option                  | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `-h, --help`            | Show the help message and exit.                                            |
| `-m, --method {naive,rle,cwa,asa,cotr,sev}` | Profiling method to use.                              |
| `-og, --og_table OG_TABLE` | Path to the ortholog table file (CSV format).                           |
| `-a, --asr_folder ASR_FOLDER` | Path to the `asr` output folder.                                     |
| `-o, --output OUTPUT`   | Output file name.                                                          |
| `-t, --tree TREE`       | Path to the species tree file (Newick format).                             |
| `-c, --cores CORES`     | Number of CPU cores to use.                                                |
| `--ignore_branch`       | Ignore branch lengths in the tree.                                         |
| `--gpu`                 | Use GPU for computation.                                                  |
| `-nb, --num_blocks NUM_BLOCKS` | Number of blocks for GPU computation.                               |
| `--test TEST`           | Number of orthologs to process in test mode.


### Statictical test
The `stat` subcommand performs statistical tests on the results generated by `profiling`.
```
corgias stat [-h] -i INPUT -m {naive,rle,cwa,asa,cotr,sev} [-o OUTPUT] [-d {both,correlation,anti-correlation}] [-c CORES] [-t THRETHOLD] [-s {bonferroni,sidak,holm-sidak,simes-hochberg,hommel,fdr_bh,fdr_by,fdr_tsbh,fdr_tsbky}] [--only_signif]

	Conduct statistical tests for phylogenetic profiling results.
	Example usage:
		corgias stat -i profiling_result.csv -m naive -o stat_out.csv -c 4
```
#### Options

| Option                  | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `-h, --help`            | Show the help message and exit.                                            |
| `-i, --input INPUT`     | Input file generated by `profiling`.                                       |
| `-m, --method {naive,rle,cwa,asa,cotr,sev}` | Profiling method used.                                |
| `-o, --output OUTPUT`   | Output file name.                                                          |
| `-d, --direction {both,correlation,anti-correlation}` | Direction of correlation to test.            |
| `-c, --cores CORES`     | Number of CPU cores to use.                                                |
| `-t, --threthold THRETHOLD` | Significance threshold.                                                |
| `-s, --statistical_test {bonferroni,sidak,holm-sidak,simes-hochberg,hommel,fdr_bh,fdr_by,fdr_tsbh,fdr_tsbky}` | Statistical test method. |
| `--only_signif`         | Output only significant results.

## License

This project is licensed under the GPL3.0 License. See the `LICENSE` file for details.

## Citation

If you use CORGIAS in your research, please the paper below:

```
Yuki Nishimura, Kimiho Omae, Kento Tominnaga, Wataru Iwasaki.
CORGIAS: identifying correlated gene pairs by considering evolutionary history in a large-scale prokaryotic genome dataset
bioRxiv, 2025, https://doi.org/10.1101/2025.05.07.652372
```

The results in this paper can be reproduced by using the code [here](https://github.com/ynishimuraLv/CORGIAS_data.git)

## Contact

Yuki Nishimura (The University of Tokyo) yuki-nishimura@g.ecc.u-tokyo.ac.jp