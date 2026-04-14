import logging
import math
import polars as pl
import ete3 as et
from itertools import combinations, islice
from multiprocessing import Pool
from tqdm import tqdm
from .common import list_asr_trees, weighted_schema

logger = logging.getLogger(__name__)


def run_asa(args):
    trees = list_asr_trees(args)
    n = len(trees)
    num_pairs = n - 1 if args.query else math.comb(n, 2)
    logger.info(f"Processing {num_pairs} OG pairs.")
    pairs = make_pairs(trees, args)
    if args.test:
        num_pairs = min(num_pairs, args.test)
        pairs = islice(pairs, args.test)

    with Pool(processes=args.cores) as pool:
        with tqdm(total=num_pairs, disable=args.quiet) as pbar:
            futures = [pool.apply_async(asa, pair, callback=lambda _: pbar.update())
                       for pair in pairs]
            result = pl.DataFrame([f.get() for f in futures],
                                  schema=weighted_schema, orient='row')

    return result


def make_pairs(trees, args):
    if args.query:
        query = trees.pop(0)
        pairs = ((query, tree, args.ignore_branch) for tree in trees)
    else:
        pairs = ((tree1, tree2, args.ignore_branch) for tree1, tree2
                in combinations(trees, 2))

    return pairs


def asa(tree_og1: tuple[str, str],
        tree_og2: tuple[str, str],
        ignore_branch: bool = False):
    tree1 = et.Tree(tree_og1[0], format=1)
    tree2 = et.Tree(tree_og2[0], format=1)
    og1 = tree_og1[1]
    og2 = tree_og2[1]
    merged_tree = merge_tree(tree1, og1, tree2, og2)
    if ignore_branch:
        result = count_by_ancestral_state(merged_tree)
    else:
        result = correct_by_ancestral_state(merged_tree)

    return og1, og2, result['1'], result['2'], result['3'], result['0']


def merge_tree(tree1: et.Tree, og1: str,
               tree2: et.Tree, og2: str):
    tree = tree1.copy()
    for node, node1, node2 in zip(tree.traverse(),
                                  tree1.traverse(),
                                  tree2.traverse()):
        node.trait = mix_trait(getattr(node1, og1), getattr(node2, og2))
    return tree


def mix_trait(og1: str, og2: str):
#    if og1 is None or og2 is None:
#        return '4'
    if og1 == '0' and og2 == '0':
        return '0'
    elif og1 == '1' and og2 == '1':
        return '1'
    elif og1 == '1' and og2 == '0':
        return '2'
    elif og1 == '0' and og2 == '1':
        return '3'
    else:
        return '4'


def correct_genomes(node: et.Tree) -> float:
    if node.num_child == 1:
        return 1
    else:
        return node.num_child * (node.pathlength / node.denominator)


def count_by_ancestral_state(tree: et.Tree):
    result = { str(i):0 for i in range(4) }
    for node in tree.traverse(strategy='postorder'):
        node.state = node.trait
        if node.is_leaf():
            node.num_child = 1
        else:
            node.num_child = 0
            if node.state in '0123':
                for child in node.get_children():
                    if node.state == child.state:
                        node.num_child = 1
                    else:
                        if child.num_child:
                            result[child.state] += 1
            else:
                for child in node.get_children():
                    if child.num_child:
                        result[child.state] += 1

    if tree.num_child:
        result[tree.state] += 1

    return result


def correct_by_ancestral_state(tree: et.Tree):
    result = { str(i):0 for i in range(4) }
    for node in tree.traverse(strategy='postorder'):
        node.state = node.trait
        if node.is_leaf():
            node.num_child = 1
            node.pathlength = node.dist
            node.denominator = node.dist
        else:
            node.num_child = 0
            node.pathlength = 0
            node.denominator = 0
            if node.state in '0123':
                for child in node.get_children():
                    if node.state == child.state:
                        node.num_child += child.num_child
                        node.pathlength += child.pathlength
                        node.denominator += child.denominator
                    else:
                        if child.num_child:
                            result[child.state] += correct_genomes(child)
                if node.num_child:
                    node.denominator += node.dist * node.num_child
                    node.pathlength += node.dist
            else:
                for child in node.get_children():
                    if child.num_child:
                        result[child.state] += correct_genomes(child)

    if tree.num_child:
        tree.denominator += tree.dist * tree.num_child
        tree.pathlength += tree.dist
        result[tree.state] += correct_genomes(tree)

    return result
