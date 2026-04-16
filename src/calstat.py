import logging
import polars as pl
from multiprocessing import Pool
from tqdm import tqdm

from scipy import stats
from statsmodels.stats.multitest import multipletests


def run_test4weighted(OG1: str, OG2: str, tt: float, tf: float,
                      ft: float, ff: float, direction: str):
    odds, pvalue = stats.fisher_exact(
        [[tt, tf],
         [ft, ff]],
         alternative = direction
    )

    return OG1, OG2, odds, pvalue

def run_test4transition(OG1: str, OG2: str, t1: int,
                        t2: int, k: int, n: int):
    direction = k
    k = abs(k)
    _, pvalue = stats.fisher_exact(
        [[k, t1-k],
         [t2-k, n-t1-t2+k]],
         alternative='greater'
    )

    return OG1, OG2, direction, pvalue


logger = logging.getLogger(__name__)

WEIGHTED_METHODS    = ['naive', 'rle', 'cwa', 'asa']
TRANSITION_METHODS  = ['cotr', 'sev']


def _compute_pvalues(df: pl.DataFrame, method: str, direction: str,
                     cores: int, quiet: bool) -> pl.DataFrame:
    """Run Fisher's exact tests on a raw profiling DataFrame.

    Returns a DataFrame with columns (OG1, OG2, odds, pvalue) for weighted
    methods, or (OG1, OG2, direction, pvalue) for transition methods.
    """
    if method in WEIGHTED_METHODS:
        dir_map = {'correlation': 'greater', 'anti-correlation': 'less', 'both': 'two-sided'}
        alt = dir_map[direction]
        df = df.with_columns(pl.lit(alt).alias('alternative'))
        rows = df.select('OG1', 'OG2', 'TT', 'TF', 'FT', 'FF', 'alternative').iter_rows()
        logger.info(f'Running statistical tests of {method} results with {direction} direction for {df.shape[0]} pairs.')
        with Pool(processes=cores) as pool:
            with tqdm(total=df.shape[0], disable=quiet) as pbar:
                futures = [pool.apply_async(run_test4weighted, row, callback=lambda _: pbar.update())
                           for row in rows]
                return pl.DataFrame([f.get() for f in futures],
                                    schema=['OG1', 'OG2', 'odds', 'pvalue'], orient='row')

    elif method in TRANSITION_METHODS:
        if direction == 'correlation':
            df = df.filter(pl.col('k') > 0)
        elif direction == 'anti-correlation':
            df = df.filter(pl.col('k') < 0)
        rows = df.iter_rows()
        logger.info(f'Running statistical tests of {method} results with {direction} direction for {df.shape[0]} pairs.')
        with Pool(processes=cores) as pool:
            with tqdm(total=df.shape[0], disable=quiet) as pbar:
                futures = [pool.apply_async(run_test4transition, row, callback=lambda _: pbar.update())
                           for row in rows]
                return pl.DataFrame([f.get() for f in futures],
                                    schema=['OG1', 'OG2', 'direction', 'pvalue'], orient='row')


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
    if args.only_signif:
        logger.info('Filtering results to include only significant pairs.')
        result = result.filter(pl.col('signif'))
    result = result.sort(by='qvalue')
    result.write_csv(args.output)
    logger.info(f'Statistical analysis completed. Results saved to {args.output}.')
