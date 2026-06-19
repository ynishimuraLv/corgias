# Corgias
<div align="center">
<p align="center">
    <img src="CORGIAS.png?raw=true?" alt="corgias-logo" width="300">
   <br> This logo was generated with the aid from AI (DALL·E).
</p>
<h1>CORGIAS</h1>
<h3>CORrelated Genes Identifier by considering Ancestral State</h3>
</div>

[![DOI](https://zenodo.org/badge/961329653.svg)](https://doi.org/10.5281/zenodo.15583904)

CORGIAS is a phylogenetic profiling tool for a large-scale dataset comprising of thousands and tens of thousands of orthologs and genomes. As co- and anti-correlated orthologs are expexted to be functionally related, CORGIAS can help functional annotation of orthologs, especially those showing no sequence similarity to functionally known genes.


## Platform Support
| Feature | Linux | macOS |
|---|---|---|
| Core functionality | ✓ | ✓ |
| Polars acceleration | ✓ | ✓ |
| CuPy GPU acceleration | ✓ (NVIDIA GPU required) | ✗ |

## Installation

### pip
```bash
# CPU-only
pip install corgias

# With GPU support (NVIDIA CUDA 12.x, Linux only)
pip install corgias[gpu]
```

### From source
```bash
git clone https://github.com/ynishimuraLv/corgias.git
cd corgias

# Create the virtual environment (Optional, but recommended)
python -m venv .venv
. .venv/bin/activate

# For System with GPU compatible CUDA 12.x
# This option is not available on macOS.
python -m pip install .[gpu]

# For System without a compatible GPU (CPU-only)
python -m pip install .
```

## Usage at a glance

**Detailed usage is discreibed [here](./USAGE.md)**

### Ancestral state reconstruction

```bash
# Reconstruct by a maximum-likelihood method
corgias asr -t samples/archaea_hq90.tre -d samples/archaea_COG_table99.csv -i 0  \
            --work_dir ML_result -c 4 --prediction_method ML --test 5

# Reconstruct by a maximum-parsimony method
corgias asr -t samples/archaea_hq90.tre -d samples/archaea_COG_table99.csv -i 0  \
            --work_dir MP_result -c 4 --prediction_method MP --test 5
```

### Phylogenetic profiling
```bash
# naime method
corgias profiling -m naive -og samples/archaea_COG_table99.csv \
                  -o naive_test.csv -c 4 --test 5

# run lenth encoding (RLE)
corgias profiling -m rle -og samples/archaea_COG_table99.csv \
                  -t samples/archaea_hq90.tre -o rle_test.csv -c 4 --test 5

# clade-wise adjustment (CWA)
corgias profiling -m cwa -og samples/archaea_COG_table99.csv \
                  -t samples/archaea_hq90.tre -o cwa_test.csv -c 4  --test 5

# ancestral state adjustment (ASA)
corgias profiling -m asa -a ML_result -t samples/archaea_hq90.tre -o asa_test.csv -c 4 --test 5

# cotransitions (cotr)
corgias profiling -m cotr -og samples/archaea_COG_table99.csv -t samples/archaea_hq90.tre -o cotr_test.csv -c 4 --test 5

# simultaneous evolution test (SEV)
corgias profiling -m sev -a MP_result -t samples/archaea_hq90.tre -o sev_test.csv -c 4 --test 5
```

### Statistical test
```bash
corgias stat -i naive_test.csv -m naive -o stat_naive.csv -c 4
corgias stat -i rle_test.csv -m rle -o stat_rle.csv -c 4
corgias stat -i cwa_test.csv -m cwa -o stat_cwa.csv -c 4
corgias stat -i asa_test.csv -m asa -o stat_asa.csv -c 4
corgias stat -i cotr_test.csv -m cotr -o stat_cotr.csv -c 4
corgias stat -i sev_test.csv -m sev -o stat_sev.csv -c 4
```

## License

This project is licensed under the GPL3.0 License. See the `LICENSE` file for details.

## Citation

If you use CORGIAS in your research, please the paper below:

```
Yuki Nishimura, Kimiho Omae, Kento Tominnaga, Wataru Iwasaki.
CORGIAS: identifying correlated gene pairs by considering evolutionary history in a large-scale prokaryotic genome dataset
NAR Genomics and Bioinformatics, 2025, https://doi.org/10.1093/nargab/lqaf182
```

The results in this paper can be reproduced by using the code [here](https://github.com/ynishimuraLv/CORGIAS_data.git)

## Contact

Yuki Nishimura (The University of Tokyo) yuki-nishimura@g.ecc.u-tokyo.ac.jp