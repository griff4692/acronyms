from collections import defaultdict
import os
import pickle
from time import sleep

import argparse
import random
import pandas as pd
import numpy as np
from scipy.stats import norm
from tqdm import tqdm

from params import SerializedParams
from utils import render_args, safe_multiply
from vocab import Vocab


def compute_log_joint(args, sfs, data, betas, beta_priors, expansion_assignments, expansion_context_means,
                      expansion_context_mean_priors):
    # compute prior p(beta; beta_priors[sf]) for each SF
    prior_log_joint = 0.0
    for sf in sfs:
        for expansion_proportions in betas[sf]:
            prior_log_joint += 0.0  # compute dirichlet pdf based on expansion_proportions as sample
        prior_log_joint += np.log(
            norm(expansion_context_mean_priors[sf], args.context_prior_var).pdf(expansion_context_means[sf])
        ).sum()

    data_log_joint = 0.0
    for n, (metadata, ce) in enumerate(data):
        sf = metadata['sf']
        expansion_assignment = expansion_assignments[n]
        data_log_joint += np.log(betas[sf][expansion_assignment])
        assignment_means = expansion_context_means[sf][expansion_assignment]
        data_log_joint += log_context_likelihood(args.context_likelihood_var, assignment_means, ce)

    log_joint_normalized = prior_log_joint + data_log_joint / float(len(data))
    return log_joint_normalized


def log_context_likelihood(context_likelihood_var, expansion_means, context_vec):
    return sum(norm(expansion_means, context_likelihood_var).logpdf(context_vec))


def sample_expansion_term(likelihood_var, sf, ce, num_expansions, betas, expansion_context_means, take_argmax=False):
    # Re-sample expansion assignment probabilities
    expansion_assignment_log_probs = np.zeros([num_expansions,])
    for v in range(num_expansions):
        expansion_ll = np.log(betas[sf][v])
        context_expansion_ll = log_context_likelihood(likelihood_var, expansion_context_means[sf][v], ce)
        expansion_assignment_log_probs[v] = expansion_ll + context_expansion_ll

    # log normalization trick to avoid precision underflow/overflow
    # https://stats.stackexchange.com/questions/66616/converting-normalizing-very-small-likelihood-values-to-probability
    max_lprob = expansion_assignment_log_probs.max()
    expansion_assignment_probs = np.exp(expansion_assignment_log_probs - max_lprob)
    expansion_assignment_probs_norm = expansion_assignment_probs / expansion_assignment_probs.sum()
    new_expansion_assignment = (np.argmax(expansion_assignment_probs_norm) if take_argmax else
                                np.random.choice(np.arange(num_expansions), p=expansion_assignment_probs_norm))
    return new_expansion_assignment


def instantiate_context_priors(args, sf_vocab, lf_prior_embeddings):
    reduce_str = '_reduced' if args.reduced_dims else ''
    init_context_means = []
    priors = []
    for i in range(sf_vocab.size()):
        lf = sf_vocab.get_token(i)
        vec = np.random.normal(
            loc=args.context_prior_mean, scale=args.context_prior_var, size=(args.embed_dim, ))  # |V| x embed_dim
        prior = np.zeros([args.embed_dim, ])
        if not args.random_expansion_priors:
            if lf not in lf_prior_embeddings:
                assert '$' in lf
            else:
                vec_mean = np.array(lf_prior_embeddings[lf]).mean(0)
                vec = vec_mean
                # TODO [figure out why it's the case that setting the prior on this is so bad]
                # prior = vec_mean
        init_context_means.append(vec)
        priors.append(prior)
    return np.array(init_context_means), priors


def _dummy_data(args, sfs):
    sf_choices = []
    n = 100
    sf_dict = [{'sf': sf} for sf in sfs]
    for _ in range(n // len(sf_dict)):
        sf_choices += sf_dict
    sf_choices += [{'sf': sfs[0]}] * (n - len(sf_choices))
    embeddings = np.random.normal(0, args.context_prior_var, size=(n, args.embed_dim))
    data = []
    for i in range(n):
        data.append((sf_choices[i], embeddings[i, :]))
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Automatic Acronym Expansion')
    # Data Path Arguments
    parser.add_argument('--acronyms_fn',
                        default='../context_extraction/data/merged_prototype_expansions_w_counts.csv')
    parser.add_argument('--semgroups_fn', default='../../expansion_etl/data/original/umls_semantic_groups.txt')

    # Experimental Parameters
    parser.add_argument('-debug', action='store_true', default=False, help='Evaluate on small dummy dataset')
    parser.add_argument('--experiment', default='debug', help='name of experiment')

    # Model Parameters
    parser.add_argument('--max_n', default=10000000, type=int, help='Maximum examples on which to run model.')
    parser.add_argument('-random_expansion_priors', action='store_true', default=False)
    parser.add_argument('--sf_exclude', default=',', help='Comma-delimited list of prototype SFs to ignore.')
    parser.add_argument('-reduced_dims', action='store_true', default=False, help='Use PCA transformed BERT vectors')

    # Model Distribution Hyperparameters
    parser.add_argument('--expansion_prior', type=float, default=1.0)

    parser.add_argument('--context_prior_mean', type=float, default=0.0)
    parser.add_argument('--context_prior_var', type=float, default=1.0)
    parser.add_argument('--context_likelihood_var', type=float, default=1.0)

    args = parser.parse_args()
    # Fixed for now based on pre-processing
    args.embed_dim = 100 if args.reduced_dims else 10 if args.debug else 768
    render_args(args)

    semgroup_vocab = Vocab('semgroups')
    with open(args.semgroups_fn, 'r') as semgroup_fd:
        semgroup_vocab.add_tokens(list(map(lambda x: x.strip().split('|')[1], semgroup_fd)))

    acronyms = pd.read_csv(args.acronyms_fn)
    sf_vocab = {}
    sfs = acronyms['sf'].unique()
    lfs = acronyms['lf'].unique()
    print('Removing {} from prototype SFs'.format(args.sf_exclude))
    sfs = list(set(sfs) - set(args.sf_exclude.split(',')))
    print('Creating expansion vocabularies for {} short forms'.format(len(sfs)))
    for sf in sfs:
        sf_vocab[sf] = Vocab(sf)
        sf_df = acronyms[acronyms['sf'] == sf]
        nonzero_sf_df = sf_df[sf_df['lf_count'] > 0]
        zero_sf_df = sf_df[sf_df['lf_count'] == 0]
        zero_long_forms_agg = '$'.join(zero_sf_df['lf'].unique().tolist())
        if len(zero_long_forms_agg) == 0:
            zero_long_forms_agg = '$'
        full_lfs_to_add = nonzero_sf_df['lf'].tolist() + [zero_long_forms_agg]
        supports = nonzero_sf_df['lf_count'].tolist() + [0]
        sf_vocab[sf].add_tokens(full_lfs_to_add, supports=supports)
        print('\tVocabulary size of {} for {}'.format(sf_vocab[sf].size(), sf))

    # Load data
    lf_prior_embeddings = defaultdict(list)
    if args.debug:
        args.random_expansion_priors = True
        data = _dummy_data(args, sfs)
        lf_prior_embeddings = {}
    else:
        print('Loading data...')
        reduce_str = '_reduced' if args.reduced_dims else ''  # Use PCA vectors or not

        data_fn = '../context_embeddings/data/prototype_contexts_w_embeddings.pk'
        with open(data_fn, 'rb') as fd:
            pickle_data = pickle.load(fd)
        sf_data = []
        data = []
        N = len(pickle_data['row_idx'])
        cols = pickle_data.keys()
        for i in range(N):
            metadata = {}
            ce, sf, lf = None, None, None
            for col in cols:
                val = pickle_data[col][i]
                if col == 'embeddings':
                    ce = val
                else:
                    metadata[col] = pickle_data[col][i]
            sf, form = pickle_data['sf'][i], pickle_data['form'][i]
            if form == sf:
                data.append((metadata, ce))
            else:
                assert form in lfs
                lf_prior_embeddings[form].append(ce)
        random.shuffle(data)
        if args.max_n < len(data):
            print('Shrinking data from {} to {}'.format(len(data), args.max_n))
            data = data[:args.max_n]

    train_fract = 0.8
    train_split_idx = round(len(data) * 0.8)
    train_data, val_data = data[:train_split_idx], data[train_split_idx:]
    print('Splitting into {} train and {} validation examples'.format(len(train_data), len(val_data)))

    # Initialize latent variables
    # Expansion proportions
    # For each sf, store the number of times an expansion was chosen for each topic
    betas = {}
    beta_priors = {}
    expansion_context_embed_sums = {}
    expansion_assignment_counts = {}
    expansion_context_means = {}
    expansion_context_mean_priors = {}
    print('Setting priors...')
    for sf in sfs:
        V = sf_vocab[sf].size()
        beta_alpha_priors = np.array([args.expansion_prior] * V)
        beta_alpha_priors = np.add(np.array(sf_vocab[sf].supports), beta_alpha_priors)
        beta_priors[sf] = beta_alpha_priors
        betas[sf] = np.random.dirichlet(beta_priors[sf])
        expansion_assignment_counts[sf] = np.zeros([V, ])
        means, priors = instantiate_context_priors(args, sf_vocab[sf], lf_prior_embeddings)
        expansion_context_means[sf] = means
        expansion_context_mean_priors[sf] = priors
        expansion_context_embed_sums[sf] = np.zeros([V, args.embed_dim])

    # Register params and prepare directory
    serialized_params = SerializedParams(outpath=os.path.join('experiments', args.experiment))
    arg_dict = [{k: getattr(args, k)} for k in vars(args)]
    serialized_params.register_param('args', args)
    serialized_params.register_param('betas', betas)
    serialized_params.register_param('beta_priors', beta_priors)
    serialized_params.register_param('expansion_context_means', expansion_context_means)
    serialized_params.register_param('expansion_context_mean_priors', expansion_context_mean_priors)
    serialized_params.register_param('sf_vocab', sf_vocab)

    # Assignment variables: first draw a topic and then draw an expansion from that topic
    train_expansion_assignments = []  # N
    for n, (metadata, ce) in enumerate(train_data):
        sf = metadata['sf']
        expansion_assignment = np.random.randint(0, sf_vocab[sf].size(), size=1)[0]
        train_expansion_assignments.append(expansion_assignment)
    val_expansion_assignments = []  # N
    for n, (metadata, ce) in enumerate(val_data):
        sf = metadata['sf']
        expansion_assignment = np.random.randint(0, sf_vocab[sf].size(), size=1)[0]
        val_expansion_assignments.append(expansion_assignment)

    train_log_joint = compute_log_joint(args, sfs, train_data, betas, beta_priors, train_expansion_assignments,
                                        expansion_context_means, expansion_context_mean_priors)
    val_log_joint = compute_log_joint(args, sfs, val_data, betas, beta_priors, val_expansion_assignments,
                                      expansion_context_means, expansion_context_mean_priors)
    print('Train Log Joint={}, Val Log Joint={} at Iteration {}'.format(train_log_joint, val_log_joint, 0))

    MAX_ITER = 1000
    for iter_ct in range(1, MAX_ITER + 1):
        # Serialize latest sample of latent variables
        serialized_params.to_disc(epoch=iter_ct)

        for n in tqdm(range(len(train_data))):
            (metadata, ce) = train_data[n]
            sf = metadata['sf']
            new_expansion_assignment = sample_expansion_term(args.context_likelihood_var, sf, ce, sf_vocab[sf].size(),
                                                             betas, expansion_context_means)
            train_expansion_assignments[n] = new_expansion_assignment

            # Update the assignment mean sums
            curr_sum = expansion_context_embed_sums[sf][new_expansion_assignment]
            expansion_context_embed_sums[sf][new_expansion_assignment] = np.add(curr_sum, ce)
            expansion_assignment_counts[sf][new_expansion_assignment] += 1

        # Update SF-topic categorical distribution over expansion terms
        for sf in sfs:
            V = sf_vocab[sf].size()
            new_dist_over_expansions = np.random.dirichlet(expansion_assignment_counts[sf] + args.expansion_prior)
            betas[sf] = new_dist_over_expansions

            # Update expansion Gaussian means
            for expansion_idx in range(V):
                ct = float(expansion_assignment_counts[sf][expansion_idx])
                empirical_sums = expansion_context_embed_sums[sf][expansion_idx, :]
                empirical_mean = empirical_sums / max(ct, 1.0)
                prior_mean = expansion_context_mean_priors[sf][expansion_idx]
                posterior_mean_num = ((prior_mean / args.context_prior_var) +
                                      (ct * empirical_mean / args.context_likelihood_var))
                posterior_mean_denom = (1.0 / args.context_prior_var) + (ct / args.context_likelihood_var)
                posterior_mean = posterior_mean_num / posterior_mean_denom
                posterior_var = 1.0 / posterior_mean_denom
                if ct > 0:
                    expansion_context_means[sf][expansion_idx] = np.random.normal(loc=posterior_mean, scale=posterior_var)
        train_log_joint = compute_log_joint(
            args, sfs, train_data, betas, beta_priors, train_expansion_assignments, expansion_context_means,
            expansion_context_mean_priors)

        # Re-assign validation data
        for n in range(len(val_data)):
            (metadata, ce) = val_data[n]
            sf = metadata['sf']
            new_expansion_assignment = sample_expansion_term(args.context_likelihood_var, sf, ce, sf_vocab[sf].size(),
                                                             betas, expansion_context_means)
            val_expansion_assignments[n] = new_expansion_assignment
        val_log_joint = compute_log_joint(
            args, sfs, val_data, betas, beta_priors, val_expansion_assignments, expansion_context_means,
            expansion_context_mean_priors)

        sleep(0.1)
        print('Train Log Joint={}, Val Log Joint={} at Iteration {}'.format(train_log_joint, val_log_joint, iter_ct))
        sleep(0.1)
        # Reset count variables to 0
        for sf in sfs:
            expansion_context_embed_sums[sf].fill(0)
            expansion_assignment_counts[sf].fill(0)
