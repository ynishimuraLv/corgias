import logging
import polars as pl
from multiprocessing import Pool

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

def run_stat(args, options):
    logger.info("Conducting statistical tests for phylogenetic profiling results")
    weighted_method = ['naive', 'rle', 'cwa', 'asa']
    transition_method = ['cotr', 'sev']

    df = pl.read_csv(args.input)
    if args.method in weighted_method:
        if args.direction == 'correlation':
            direction = 'greater'
        elif args.direction == 'anti-correlation':
            direction = 'less'
        else:
            direction = 'two-sided'
        df = df.with_columns(pl.lit(direction).alias('alternative'))
        rows = df.select('OG1', 'OG2', 'TT', 'TF', 'FT', 'FF', 'alternative').iter_rows()
        logger.info(f'Running statistical tests of {args.method} results with {args.direction} direction for {df.shape[0]} pairs.')
        with Pool(processes=args.cores) as process:
            result = pl.DataFrame(process.starmap_async(run_test4weighted, rows).get(),
                                    schema=['OG1', 'OG2', 'odds', 'pvalue'], orient='row')
    elif args.method in transition_method:
        if args.direction == 'correlation':
            df = df.filter(pl.col('k') > 0)
        elif args.direction == 'anti-correlation':
            df = df.filter(pl.col('k') < 0)
        rows = df.iter_rows()
        logger.info(f'Running statistical tests of {args.method} results with {args.direction} direction for {df.shape[0]} pairs.')
        with Pool(processes=args.cores) as process:
            result = pl.DataFrame(process.starmap_async(run_test4transition, rows).get(),
                                    schema=['OG1', 'OG2', 'direction', 'pvalue'],
                                    orient='row')

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