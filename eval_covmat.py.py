import os
import argparse
import pickle
import torch

from utils.datasets import PackedConformationDataset
from utils.evaluation.covmat import CovMatEvaluator, print_covmat_results
from utils.misc import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--ratio', type=int, default=2)
    parser.add_argument('--start_idx', type=int, default=0)
    args = parser.parse_args()
    assert os.path.isfile(args.path)

    # Logging
    tag = args.path.split('/')[-1].split('.')[0]
    logger = get_logger('eval', os.path.dirname(args.path), 'log_eval_'+tag+'.txt')
    
    # Load results
    logger.info('Loading results: %s' % args.path)
    with open(args.path, 'rb') as f:
        packed_dataset = pickle.load(f)
    logger.info('Total: %d' % len(packed_dataset))

    # Evaluator
    evaluator = CovMatEvaluator(
        num_workers = args.num_workers,
        ratio = args.ratio,
        print_fn=logger.info,
    )
    results = evaluator(
        packed_data_list = list(packed_dataset),
        start_idx = args.start_idx,
    )
    df = print_covmat_results(results, print_fn=logger.info)

    # Save results
    csv_fn = args.path[:-4] + '_covmat.csv'
    results_fn = args.path[:-4] + '_covmat.pkl'
    df.to_csv(csv_fn)
    with open(results_fn, 'wb') as f:
        pickle.dump(results, f)

