import json
import os

import argparse
import random
import pandas as pd
import numpy as np
from scipy import spatial
from scipy.stats import norm

from utils import render_args
from vocab import Vocab


def compute_log_joint(args, sfs, data, betas, beta_priors, expansion_assignments, expansion_context_means,
                      expansion_context_mean_priors):
    # compute prior p(beta; beta_priors[sf]) for each SF
    log_joint = 0.0
    for sf in sfs:
        for expansion_proportions in betas[sf]:
            log_joint += 0.0  # compute dirichlet pdf based on expansion_proportions as sample
        log_joint += np.log(
            norm(expansion_context_mean_priors[sf], args.context_prior_var).pdf(expansion_context_means[sf])
        ).sum()

    # compute prior p(theta) for each document
    for n, (sf, _, _, ce) in enumerate(data):
        expansion_assignment = expansion_assignments[n]

        # p(expansion_z | topic_z, betas, sf)
        log_joint += np.log(betas[sf][expansion_assignment])
        assignment_means = expansion_context_means[sf][expansion_assignment]
        log_joint += log_context_likelihood(args, assignment_means, ce)

    return log_joint


def safe_multiply(a, b):
    return np.exp(np.log(a + 1e-5) + np.log(b + 1e-5))


def log_context_likelihood(args, expansion_means, context_vec):
    if args.use_cosine_likelihood:
        distance = spatial.distance.cosine(expansion_means, context_vec)
        distance_normalized = 0.5 * distance + 0.5
        return np.log(distance_normalized)
    return sum(
        map(lambda x: np.log(norm(x[0], args.context_likelihood_var).pdf(x[1])), zip(expansion_means, context_vec))
    )


def instantiate_context_priors(args, sf_vocab):
    # Look up sf in ../context_embeddings/data/lf_embeddings
    init_context_means = []
    priors = []
    for i in range(sf_vocab.size()):
        lf = sf_vocab.get_token(i)
        fn = '../context_embeddings/data/lf_embeddings/{}.npy'.format(lf)
        vec = np.random.normal(
            loc=args.context_prior_mean, scale=args.context_prior_var, size=(args.embed_dim, ))  # |V| x embed_dim
        prior = np.zeros([args.embed_dim, ])
        if os.path.exists(fn) and not args.random_expansion_priors:
            with open(fn, 'rb') as fd:
                vectors = np.load(fd)
            if len(vectors) > 0 and False:
                vec = np.tanh(vectors.mean(0))
                prior = vec
        init_context_means.append(vec)
        priors.append(prior)
    return np.array(init_context_means), priors


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Automatic Acronym Expansion')
    # Data Path Arguments
    parser.add_argument('--acronyms_fn',
                        default='../expansion_etl/data/derived/prototype_acronym_expansions_w_counts.csv')
    parser.add_argument('--semgroups_fn', default='../expansion_etl/data/original/umls_semantic_groups.txt')

    # Model Parameters
    parser.add_argument('--max_n', default=1000, type=int, help='Maximum examples on which to run model.')
    parser.add_argument('--embed_dim', default=768, type=int, help='Fixed by BERT embedding dim')
    parser.add_argument('-random_expansion_priors', action='store_true', default=False)
    parser.add_argument('-use_cosine_likelihood', action='store_true', default=False)
    parser.add_argument('--sf_exclude', default='MD,SOB', help='Comma-delimited list of prototype SFs to ignore.')

    # Model Distribution Hyperparameters
    parser.add_argument('--expansion_prior', type=float, default=1.0)

    parser.add_argument('--context_prior_mean', type=float, default=0.0)
    parser.add_argument('--context_prior_var', type=float, default=1.0)
    parser.add_argument('--context_likelihood_var', type=float, default=1.0)

    args = parser.parse_args()
    render_args(args)

    semgroup_vocab = Vocab('semgroups')
    with open(args.semgroups_fn, 'r') as semgroup_fd:
        semgroup_vocab.add_tokens(list(map(lambda x: x.strip().split('|')[1], semgroup_fd)))

    acronyms = pd.read_csv(args.acronyms_fn)
    sf_vocab = {}
    sfs = acronyms['sf'].unique()
    print('Removing {} from prototype SFs'.format(args.sf_exclude))
    sfs = list(set(sfs) - set(args.sf_exclude.split(',')))
    print('Creating expansion vocabularies for {} short forms'.format(len(sfs)))
    for sf in sfs:
        sf_vocab[sf] = Vocab(sf)
        sf_df = acronyms[acronyms['sf'] == sf]
        nonzero_sf_df = sf_df[sf_df['lf_count'] > 0]
        zero_sf_df = sf_df[sf_df['lf_count'] == 0]
        zero_long_forms_agg = '$'.join(zero_sf_df['lf'].unique().tolist())
        full_lfs_to_add = nonzero_sf_df['lf'].tolist() + [zero_long_forms_agg]
        supports = nonzero_sf_df['lf_count'].tolist() + [0]
        sf_vocab[sf].add_tokens(full_lfs_to_add, supports=supports)
        print('\tVocabulary size of {} for {}'.format(sf_vocab[sf].size(), sf))

    # Initialize latent variables
    # Expansion proportions
    # For each sf, store the number of times an expansion was chosen for each topic
    betas = {}
    beta_priors = {}
    expansion_context_embed_sums = {}
    expansion_assignment_counts = {}
    expansion_context_means = {}
    expansion_context_mean_priors = {}
    for sf in sfs:
        V = sf_vocab[sf].size()
        beta_alpha_priors = np.add(np.array(sf_vocab[sf].supports), np.array([args.expansion_prior] * V))
        beta_priors[sf] = beta_alpha_priors
        betas[sf] = np.random.dirichlet(beta_priors[sf])
        expansion_assignment_counts[sf] = np.zeros([V, ])
        means, priors = instantiate_context_priors(args, sf_vocab[sf])
        expansion_context_means[sf] = means
        expansion_context_mean_priors[sf] = priors
        expansion_context_embed_sums[sf] = np.zeros([V, args.embed_dim])

    data = []
    for sf in sfs:
        embed_fn = '../context_embeddings/data/sf_embeddings/{}.npy'.format(sf)
        key_fn = '../context_embeddings/data/sf_embeddings/{}_keys.json'.format(sf)
        with open(embed_fn, 'rb') as fd:
            sf_vectors = np.load(fd)
        with open(key_fn, 'rb') as fd:
            sf_keys = json.load(fd)
        sf_N = sf_vectors.shape[0]
        for i in range(sf_N):
            # SF, line_idx, doc_id, sf_vector
            data.append((sf, sf_keys[i][0], sf_keys[i][1], sf_vectors[i, :]))
    random.shuffle(data)
    if args.max_n < len(data):
        print('Shrinking data from {} to {}'.format(len(data), args.max_n))
        data = data[:args.max_n]

    # Assignment variables: first draw a topic and then draw an expansion from that topic
    expansion_assignments = []  # N
    for n, (sf, _, _, ce) in enumerate(data):
        expansion_assignment = np.random.randint(0, sf_vocab[sf].size(), size=1)[0]
        expansion_assignments.append(expansion_assignment)

    log_joint = compute_log_joint(args, sfs, data, betas, beta_priors, expansion_assignments, expansion_context_means,
                                  expansion_context_mean_priors)
    print('Log Joint={} at Iteration {}'.format(log_joint, 0))

    MAX_ITER = 1000
    for iter_ct in range(1, MAX_ITER + 1):
        for n, (sf, _, _, ce) in enumerate(data):
            # Re-sample expansion assignment probabilities
            V = sf_vocab[sf].size()
            expansion_assignment_log_probs = np.zeros([V, ])
            for v in range(V):
                expansion_ll = np.log(betas[sf][v])
                context_expansion_ll = log_context_likelihood(args, expansion_context_means[sf][v], ce)
                expansion_assignment_log_probs[v] = expansion_ll + context_expansion_ll

            # log normalization trick to avoid precision underflow/overflow
            # https://stats.stackexchange.com/questions/66616/converting-normalizing-very-small-likelihood-values-to-probability
            max_lprob = expansion_assignment_log_probs.max()
            expansion_assignment_probs = np.exp(expansion_assignment_log_probs - max_lprob)
            expansion_assignment_probs_norm = expansion_assignment_probs / expansion_assignment_probs.sum()
            new_expansion_assignment = np.random.choice(np.arange(V), p=expansion_assignment_probs_norm)
            expansion_assignments[n] = new_expansion_assignment

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
                expansion_context_means[sf][expansion_idx] = np.random.normal(loc=posterior_mean, scale=posterior_var)
        log_joint = compute_log_joint(
            args, sfs, data, betas, beta_priors, expansion_assignments, expansion_context_means,
            expansion_context_mean_priors)
        print('Log Joint={} at Iteration {}'.format(log_joint, iter_ct))

        # Reset count variables to 0
        for sf in sfs:
            expansion_context_embed_sums[sf].fill(0)
            expansion_assignment_counts[sf].fill(0)
