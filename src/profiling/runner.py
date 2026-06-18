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


def _log_params(args):
    parts = [f"method={args.method}", f"output={args.output}", f"cores={args.cores}"]
    for attr in ('og_table', 'og_table2', 'asr_folder', 'asr_folder2', 'tree', 'query'):
        val = getattr(args, attr, None)
        if val:
            parts.append(f"{attr}={val}")
    if getattr(args, 'gpu', False):
        parts.append(f"gpu=True")
        nb = getattr(args, 'num_blocks', 0)
        if nb:
            parts.append(f"num_blocks={nb}")
    if getattr(args, 'test', 0):
        parts.append(f"test={args.test}")
    logger.info("Parameters: " + ", ".join(parts))


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

    _log_params(args)
    method_runner = METHOD_RUNNERS[args.method]
    try:
        result = method_runner(args)
    except MemoryError:
        if getattr(args, 'gpu', False):
            logger.error(
                "GPU out of memory. Re-run with -nb/--num_blocks (e.g. -nb 4) "
                "to split the computation into smaller blocks."
            )
            sys.exit(1)
        raise
    result.write_csv(args.output)
