# Usage

CORGIAS provides three subcommand.
1. **`asr`** performs ancestral state of ortholog presence/absence for Ancestral steate adjustment (ASA) and Simultaneous Evolutionary Test (SEV).
2. **`profiling`** calculate values necessary for evaluating correlation of all ortholog pairs. Six methods are available, and ASA and SEV are our proposed methods.
3. **`stat`** performs statistical tests using values calculated by `profiling`.

## Ancestral state construction (ASR)

`asr` reuqires a rooted species tree in Newick format and an ortholog table in CSV or CSV format. This subcommand acts as a wrapper for running  `pastml` in parallel.

```
$corgias asr -h
usage: corgias asr [-h] -t TREE [-m {MPPA,MAP,JOINT,DOWNPASS,ACCTRAN,DELTRAN,ML,MP}] -d DATA
                   [-i ID_INDEX] --work_dir WORK_DIR [-c CORES] [--test TEST] [--tmp TMP] [--keep]

	Prepare trees with ancestral presence/absence states of ortholog for ASA or SEV profiling.
	The ortholog table should be a CSV file but each ortholog is assmued to be evolved independently.
	Example usage:
		corgias asr -t tree.nwk -d orthologs.csv -i 0 -s "," -o pastml_result -c 4 --prediction_method ML
```

### Options
| Option                  | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `-h, --help`            | Show the help message and exit.                                            |
| `-t, --tree TREE`       | Path to the species tree file (Newick format).                             |
| `-m, --prediction_method METHOD`       | Method for ASR.                             |
| `-d, --data DATA`       | Path to the ortholog table file (CSV format).                              |
| `-i, --id_index ID_INDEX` | Column index for ortholog IDs (0-based).                                 |
| `-s, --separator SEPARATOR` | Separator for the ortholog table (default:  `,`).                      |
| `--work_dir WORK_DIR`   | Directory for output files.                                                |
| `-c, --cores CORES`     | Number of CPU cores to use. It should be positive integer.                 |
| `--test TEST`           | Number of orthologs to process in test mode.                               |
| `--tmp TMP`             | Temporary directory for intermediate files.                               |
| `--keep`                | Keep intermediate files.                                                  |

**Note**: Reconstruction should be performed by a maximum-likelihood (MPPA) and maximum-parsimony method (ACCTRAN) for ASA and SEV, respectively. Therefore, ML and MP are replaced with MPPA and ACCTRAN.
In addition to the above, `pastml` options can be acceptable (for example, `--upload_to_itol`).

## Phylogenetic profiling

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
**Note:** Input options that are not supported by each method will be ignored even if specified.

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
### Options
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

**Note:** `--gpu` and `--num_blocks` are available only when CORGIAS installed with `.[gpu]` and when selected methods is naive, cotr or sev.

### Output
Depending on the selected method, the output CSV file contains the following columns:

**Weighted methods (`naive`, `rle`, `cwa`, `asa`)**

| OG1   | OG2   | TT   | TF   | FT   | FF   |
|-------|-------|------|------|------|------|
| OG0001| OG0002| 12.0 | 3.0  | 2.0  | 8.0  |

- **OG1, OG2**: Ortholog pair IDs
- **TT**: Number of genomes where both OG1 and OG2 are present
- **TF**: Number of genomes where OG1 is present and OG2 is absent
- **FT**: Number of genomes where OG1 is absent and OG2 is present
- **FF**: Number of genomes where both OG1 and OG2 are absent

**Transition methods (`cotr`, `sev`)**

| OG1   | OG2   |num_change1|num_change2| k   |   N   |
|-------|-------|-----------|-----------|-----|-------|
| OG0001| OG0002| 23         | 46       |  5  |  100  |

- **OG1, OG2**: Ortholog pair IDs
- **num_change1, numchnage2**: Number of presence/absence changes of OG1 and OG2
- **k**: Number of concordance(positive)/discordance(negative) changes
- **n**: Number of genomes (`cotr`) or internal nodes in the tree


## Statictical test
The `stat` subcommand performs statistical tests on the results generated by `profiling`.
```
corgias stat [-h] -i INPUT -m {naive,rle,cwa,asa,cotr,sev} [-o OUTPUT] [-d {both,correlation,anti-correlation}] [-c CORES] [-t THRETHOLD] [-s {bonferroni,sidak,holm-sidak,simes-hochberg,hommel,fdr_bh,fdr_by,fdr_tsbh,fdr_tsbky}] [--only_signif]

	Conduct statistical tests for phylogenetic profiling results.
	Example usage:
		corgias stat -i profiling_result.csv -m naive -o stat_out.csv -c 4
```
### Options

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

###S Output
The output CSV file from the `stat` subcommand contains:

| OG1   | OG2   | odds | pvalue | qvalue | signif |
|-------|-------|------|--------|--------|--------|
| OG0001| OG0002| 2.5  | 0.01   | 0.02   | True   |

- **odds**: Odds ratio
- **pvalue**: P-value from statistical test
- **qvalue**: Adjusted p-value (multiple testing correction)
- **signif**: Whether the result is significant (True/False)