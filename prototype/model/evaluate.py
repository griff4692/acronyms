from collections import defaultdict
import os

import argparse
import numpy as np
from sklearn.metrics import classification_report

from main import sample_expansion_term
from params import SerializedParams
from utils import render_args
from vocab import Vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Automatic Acronym Expansion')
    # Data Path Arguments
    parser.add_argument('--acronyms_fn',
                        default='../context_extraction/data/merged_prototype_expansions_w_counts.csv')

    # Experimental Parameters
    parser.add_argument('--experiment', default='12-1-baseline', help='name of experiment')
    parser.add_argument('--iter', default='1', help='Iteration over which to compute MC approximation')

    args = parser.parse_args()
    # Fixed for now based on pre-processing
    render_args(args)

    params_file = os.path.join('experiments', args.experiment, 'params_{}.pkl'.format(args.iter))
    print('Retrieving latent parameters from {}'.format(params_file))
    assert os.path.exists(params_file)
    sp = SerializedParams.from_disc(params_file)

    prev_args = sp['args']
    betas = sp['betas']
    expansion_context_means = sp['expansion_context_means']
    sf_vocab = sp['sf_vocab']

    # Dummy data --> (sf, lf, ce)
    eval_data = []

    for _ in range(10):
        eval_data.append((
            'PVD',
            'peripheral vascular disease|peripheral vascular diseases',
            np.random.normal(size=(768, ))
        ))

    sf_results = defaultdict(lambda: ([], []))

    print('Starting evaluation...')
    for (sf, target_lf, ce) in eval_data:
        num_expansions = sf_vocab[sf].size()
        predicted_lf_id = sample_expansion_term(prev_args.context_likelihood_var, sf, ce, num_expansions, betas,
                                                expansion_context_means, take_argmax=True)
        predicted_lf = sf_vocab[sf].get_token(predicted_lf_id)
        target_lf_id = sf_vocab[sf].get_id(target_lf)
        sf_results[sf][0].append(target_lf_id)
        sf_results[sf][1].append(predicted_lf_id)

    for sf in sf_results.keys():
        print('Classification report for {}'.format(sf))
        y_true, y_pred = sf_results[sf][0], sf_results[sf][1]
        target_names = sf_vocab[sf].i2w
        print(y_true, y_pred, len(target_names))
        print(classification_report(y_true, y_pred, labels=list(range(len(target_names))), target_names=target_names))
