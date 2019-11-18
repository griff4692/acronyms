import argparse
import pandas as pd
import numpy as np
from scipy.stats import norm

from model.vocab import Vocab


def safe_multiply(a, b):
    return np.exp(np.log(a + 1e-5) + np.log(b + 1e-5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Automatic Acronym Expansion')
    # Data Path Arguments
    parser.add_argument('--acronyms_fn', default='../expansion_etl/data/derived/prototype_acronym_expansions.csv')
    parser.add_argument('--semgroups_fn', default='../expansion_etl/data/original/umls_semantic_groups.txt')

    # Model Distribution Hyperparameters
    parser.add_argument('--document_topic_prior', type=float, default=1.0)
    parser.add_argument('--topic_expansion_prior', type=float, default=1.0)

    args = parser.parse_args()

    semgroup_vocab = Vocab('semgroups')
    with open(args.semgroups_fn, 'r') as semgroup_fd:
        semgroup_vocab.add_tokens(list(map(lambda x: x.strip().split('|')[1], semgroup_fd)))

    acronyms = pd.read_csv(args.acronyms_fn)
    sf_vocab = {}
    sfs = acronyms['sf'].unique()
    print('Creating expansion vocabularies for {} short forms'.format(len(sfs)))
    for sf in sfs:
        sf_vocab[sf] = Vocab(sf)
        sf_vocab[sf].add_tokens(acronyms[acronyms['sf'] == sf]['lf'].tolist())
        print('\tVocabulary size of {} for {}'.format(sf_vocab[sf].size(), sf))

    # Model Dimensions & Hyperparameters
    num_docs = 3
    num_acronyms = len(sfs)
    embed_dim = 100
    num_topics = semgroup_vocab.size()  # 15

    # Initialize latent variables

    # Per-document topic proportions
    theta_alpha_priors = np.array([args.document_topic_prior] * num_topics)
    thetas = np.random.dirichlet(theta_alpha_priors, size=num_docs)
    # Per-topic expansion proportions
    # For each sf, store the number of times an expansion was chosen for each topic
    topic_expansion_assignment_counts = {}
    betas = {}
    for sf in sfs:
        V = sf_vocab[sf].size()
        beta_alpha_priors = np.array([args.document_topic_prior] * V)
        betas[sf] = np.random.dirichlet(beta_alpha_priors, size=num_topics)
        topic_expansion_assignment_counts[sf] = np.zeros([num_topics, V])

    expansion_context_means = {}
    expansion_context_embed_sums = {}
    expansion_assignment_counts = {}
    for sf in sfs:
        V = sf_vocab[sf].size()
        expansion_context_means[sf] = np.random.normal(loc=0.0, scale=1.0, size=(V, embed_dim))  # |V| x embed_dim
        expansion_assignment_counts[sf] = np.zeros([V, ])
        expansion_context_embed_sums[sf] = np.zeros([V, embed_dim])

    # Dummy data
    # |num_docs| x examples in doc_i x (sf, context embedding)
    data = []
    for n in range(num_docs):
        doc_examples = np.zeros([10, embed_dim + 1])
        for m in range(10):
            # randomly choose a sf
            sf = np.random.choice(np.arange(num_acronyms), size=1)[0]
            ce = np.random.normal(loc=0.0, scale=1.0, size=(embed_dim, ))
            doc_examples[m, 0] = sf
            doc_examples[m, 1:] = ce
        data.append(doc_examples)

    doc_topic_assignment_counts = np.zeros([num_docs, num_topics])
    # Assignment variables: first draw a topic and then draw an expansion from that topic
    topic_assignments = []  # |num_docs | x examples in doc_i x 1
    expansion_assignments = []  # |num_docs | x examples in doc_i x 1
    for n in range(num_docs):
        m = 10
        doc_topic_assignments = np.random.randint(0, num_topics, size=m)
        for topic_z in doc_topic_assignments:
            doc_topic_assignment_counts[n, topic_z] += 1
        doc_expansion_assignments = np.zeros([m, ], dtype=int)
        for m, example in enumerate(data[n]):
            sf = sfs[int(example[0])]
            ce = example[1:]
            expansion_assignment = np.random.randint(0, sf_vocab[sf].size(), size=1)[0]
            doc_expansion_assignments[m] = expansion_assignment
            # Update topic_expansion_assignment_counts (sf, num_topics, |V|)
            topic_expansion_assignment_counts[sf][doc_topic_assignments[m]][expansion_assignment] += 1
        topic_assignments.append(doc_topic_assignments)
        expansion_assignments.append(doc_expansion_assignments)

    MAX_ITER = 1000
    for iter_ct in range(1, MAX_ITER + 1):
        for n, doc in enumerate(data):
            # Resample topic proportions for each document
            new_doc_topic_proportions = np.random.dirichlet(
                list(map(lambda x: x + args.document_topic_prior, doc_topic_assignment_counts[n])))
            thetas[n, :] = new_doc_topic_proportions

            # Resample expansion assignments (and then topic assignments)
            for m, example in enumerate(doc):
                sf = sfs[int(example[0])]
                ce = example[1:]

                #### START TOPIC ASSIGNMENT RESAMPLING FOR example (n,m)
                #### Depends on expansion assignment of (n,m) beta and theta
                # hold given topic assignment fixed and compute expansion assignment probabilities
                topic_assignment = topic_assignments[n][m]
                V = sf_vocab[sf].size()
                expansion_assignment_log_probs = np.zeros([V, ])
                for v in range(V):
                    expansion_lprob = np.log(betas[sf][topic_assignment][v])
                    assignment_means = expansion_context_means[sf][v]
                    for eidx in range(embed_dim):
                        expansion_lprob += np.log(norm(assignment_means[eidx], 1).pdf(ce[eidx]))
                    expansion_assignment_log_probs[v] = expansion_lprob
                expansion_assignment_probs = np.exp(expansion_assignment_log_probs)
                expansion_assignment_probs_norm = expansion_assignment_probs / expansion_assignment_probs.sum()
                new_expansion_assignment = np.random.choice(np.arange(V), p=expansion_assignment_probs_norm)
                expansion_assignments[n][m] = new_expansion_assignment

                # Update the assignment mean sums
                expansion_context_embed_sums[sf][new_expansion_assignment] = np.add(
                    expansion_context_embed_sums[sf][new_expansion_assignment], ce)
                expansion_assignment_counts[sf][new_expansion_assignment] += 1

                # Resample topic assignments
                # topic likelihood composed of p(z|theta) * p(e|z,beta)
                topic_assignment_lprobs = np.zeros([num_topics, ])
                for topic_idx in range(num_topics):
                    topic_assignment_lprobs[topic_idx] = (np.log(thetas[n][topic_idx]) +
                                              np.log(betas[sf][topic_idx][new_expansion_assignment]))
                topic_assignment_probs = np.exp(topic_assignment_lprobs)
                topic_assignment_probs_norm = topic_assignment_probs / topic_assignment_probs.sum()
                new_topic_assignment = int(np.random.choice(
                    np.arange(num_topics), p=topic_assignment_probs_norm, size=1)[0])
                topic_assignments[n][m] = new_topic_assignment

        # Update SF-topic categorical distribution over expansion terms
        for sf in sfs:
            for topic_idx in range(num_topics):
                expansion_counts = topic_expansion_assignment_counts[sf][topic_idx]
                new_dist_over_expansions = np.random.dirichlet(expansion_counts + args.topic_expansion_prior)
                betas[sf][topic_idx] = new_dist_over_expansions

        # Update expansion Gaussian means
        for sf in sfs:
            V = sf_vocab[sf].size()
            for expansion_idx in range(V):
                ct = float(expansion_assignment_counts[sf][expansion_idx])
                if ct > 0:
                    new_means = (expansion_context_embed_sums[sf][expansion_idx, :] / ct)
                    expansion_context_means[sf][expansion_idx] = new_means
            expansion_context_embed_sums[sf].fill(0)
            expansion_assignment_counts[sf].fill(0)
        print('Done Iteration {}'.format(iter_ct))
