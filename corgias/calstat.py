import logging
import math
import polars as pl
from multiprocessing import Pool
from tqdm import tqdm

from scipy import stats
from statsmodels.stats.multitest import multipletests


def phi_coefficient(a: float, b: float, c: float, d: float) -> float:
    """Effect size (phi coefficient / Matthews correlation coefficient) for a 2x2 table [[a,b],[c,d]].

    Unlike the p-value, phi does not shrink as the number of genomes/branches (N) grows,
    so it stays comparable across pairs and datasets of very different sizes.
    """
    a, b, c, d = float(a), float(b), float(c), float(d)
    denom = math.sqrt((a + b) * (a + c) * (b + d) * (c + d))
    if denom == 0:
        return 0.0
    return (a * d - b * c) / denom


def run_test4weighted(OG1: str, OG2: str, tt: float, tf: float,
                      ft: float, ff: float, direction: str):
    odds, pvalue = stats.fisher_exact(
        [[tt, tf],
         [ft, ff]],
         alternative = direction
    )
    phi = phi_coefficient(tt, tf, ft, ff)

    return OG1, OG2, odds, phi, pvalue

def run_test4transition(OG1: str, OG2: str, t1: int,
                        t2: int, k: int, n: int):
    direction = k
    k = abs(k)
    _, pvalue = stats.fisher_exact(
        [[k, t1-k],
         [t2-k, n-t1-t2+k]],
         alternative='greater'
    )
    phi = phi_coefficient(k, t1-k, t2-k, n-t1-t2+k)
    if direction < 0:
        phi = -phi

    return OG1, OG2, direction, phi, pvalue


def _apply_weighted(row):
    return run_test4weighted(*row)

def _apply_transition(row):
    return run_test4transition(*row)


logger = logging.getLogger(__name__)

WEIGHTED_METHODS    = ['naive', 'rle', 'cwa', 'asa']
TRANSITION_METHODS  = ['cotr', 'sev']


def _compute_pvalues(df: pl.DataFrame, method: str, direction: str,
                     cores: int, quiet: bool) -> pl.DataFrame:
    """Run Fisher's exact tests on a raw profiling DataFrame.

    Returns a DataFrame with columns (OG1, OG2, odds, phi, pvalue) for weighted
    methods, or (OG1, OG2, direction, phi, pvalue) for transition methods.
    `phi` is the phi coefficient (effect size), independent of dataset size.
    """
    if method in WEIGHTED_METHODS:
        dir_map = {'correlation': 'greater', 'anti-correlation': 'less', 'both': 'two-sided'}
        alt = dir_map[direction]
        df = df.with_columns(pl.lit(alt).alias('alternative'))
        n = df.shape[0]
        rows = df.select('OG1', 'OG2', 'TT', 'TF', 'FT', 'FF', 'alternative').iter_rows()
        chunksize = max(1, n // (cores * 10))
        logger.info(f'Running statistical tests of {method} results with {direction} direction for {n} pairs.')
        with Pool(processes=cores) as pool:
            with tqdm(total=n, disable=quiet) as pbar:
                results = []
                for r in pool.imap(_apply_weighted, rows, chunksize=chunksize):
                    results.append(r)
                    pbar.update()
        return pl.DataFrame(results, schema=['OG1', 'OG2', 'odds', 'phi', 'pvalue'], orient='row')

    elif method in TRANSITION_METHODS:
        if direction == 'correlation':
            df = df.filter(pl.col('k') > 0)
        elif direction == 'anti-correlation':
            df = df.filter(pl.col('k') < 0)
        n = df.shape[0]
        rows = df.iter_rows()
        chunksize = max(1, n // (cores * 10))
        logger.info(f'Running statistical tests of {method} results with {direction} direction for {n} pairs.')
        with Pool(processes=cores) as pool:
            with tqdm(total=n, disable=quiet) as pbar:
                results = []
                for r in pool.imap(_apply_transition, rows, chunksize=chunksize):
                    results.append(r)
                    pbar.update()
        return pl.DataFrame(results, schema=['OG1', 'OG2', 'direction', 'phi', 'pvalue'], orient='row')


def run_stat(args, options):
    logger.info("Conducting statistical tests for phylogenetic profiling results")

    if args.input2:
        primary = pl.read_csv(args.input)
        if 'qvalue' not in primary.columns or 'pvalue' not in primary.columns:
            raise ValueError(
                f"'-i {args.input}' does not look like a stat result file "
                "(expected columns: pvalue, qvalue, signif). "
                "When using -i2, -i must be the output of a previous 'corgias stat' run."
            )
        if 'signif' in primary.columns and primary['signif'].all():
            logger.warning(
                "Primary stat result may have been filtered with --only_signif. "
                "Combined multiple testing correction requires all pairs to be present."
            )
        primary = primary.drop(['qvalue', 'signif'])
        df2 = pl.read_csv(args.input2)
        logger.info(f"Secondary mode: {len(primary)} primary pairs + {len(df2)} secondary pairs to test.")
        secondary = _compute_pvalues(df2, args.method, args.direction, args.cores, args.quiet)
        result = pl.concat([primary, secondary])
    else:
        df = pl.read_csv(args.input)
        result = _compute_pvalues(df, args.method, args.direction, args.cores, args.quiet)

    logger.info(f'Adjusting p-values with {args.statistical_test} method and threshold of {args.threthold}.')
    qvalues = multipletests(result['pvalue'], method=args.statistical_test,
                            alpha=args.threthold)
    result = result.with_columns(pl.Series('qvalue', qvalues[1]),
                                    pl.Series('signif', qvalues[0]))

    if args.min_k is not None:
        if args.method not in TRANSITION_METHODS:
            raise ValueError('--min_k only applies to transition methods (cotr, sev).')
        logger.info(f'Filtering results to keep pairs with at least {args.min_k} co-transition events.')
        result = result.filter(pl.col('direction').abs() >= args.min_k)

    if args.min_phi is not None:
        logger.info(f'Filtering results to keep pairs with |phi| >= {args.min_phi}.')
        result = result.filter(pl.col('phi').abs() >= args.min_phi)

    if args.only_signif:
        logger.info('Filtering results to include only significant pairs.')
        result = result.filter(pl.col('signif'))
    result = result.sort(by='qvalue')
    result.write_csv(args.output)
    logger.info(f'Statistical analysis completed. Results saved to {args.output}.')
