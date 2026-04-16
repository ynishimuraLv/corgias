import logging
import sys
from .naive import run_naive
from .rle_cwa import run_rle_cwa
from .asa import run_asa
from .transition import run_cotr, run_sev
from .gpu_utils import CUPY_AVAILABLE

logger = logging.getLogger(__name__)


METHOD_RUNNERS = {
    "naive": run_naive,
    "rle": run_rle_cwa,
    "cwa": run_rle_cwa,
    "cotr": run_cotr,
    "asa": run_asa,
    "sev": run_sev,
}

def validate_args(args):
    if (args.og_table2 or args.asr_folder2) and args.query:
        raise ValueError('Secondary dataset mode (-og2/-a2) cannot be combined with -q/--query')
    if args.og_table2 and args.method in ['asa', 'sev']:
        raise ValueError('-og2 is not valid for asa/sev; use -a2 instead')
    if args.asr_folder2 and args.method in ['naive', 'rle', 'cwa', 'cotr']:
        raise ValueError('-a2 is not valid for naive/rle/cwa/cotr; use -og2 instead')

    if args.method == 'naive' and not args.og_table:
        raise ValueError('An ortholog table is required when using naive method')
    elif args.method in ['rle', 'cwa']:
        if (not args.tree) or (not args.og_table):
            raise ValueError('An ortholog table and a phylogenetic tree are '
                             f'required when using {args.method} method')
    elif args.method in ['asa', 'sev'] and not args.asr_folder:
        raise ValueError('The results of ancestral state reconstruction are '
                         f'required when using {args.method} method')
    elif args.method in ['asa', 'sev'] and not args.tree:
        raise ValueError('A phylogenetic tree is required '
                         f'when using {args.method} method')


def run_profiling(args, options):
    logger.info("Starting phylogenetic profiling")
    if not CUPY_AVAILABLE:
        args.gpu = False
        args.num_blocks = 0
    try:
        validate_args(args)
    except ValueError as e:
        logger.error(f"Invalid arguments: {e}")
        sys.exit(1)

    method_runner = METHOD_RUNNERS[args.method]
    logger.info(f"Selected method: {args.method}")
    result = method_runner(args)
    result.write_csv(args.output)