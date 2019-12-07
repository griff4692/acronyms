import os
import sys
sys.path.insert(0, os.path.expanduser('~/acronyms/'))

import argparse
import pandas as pd

from prototype.context_embeddings.embed_contexts import embed_bert
from prototype.context_embeddings.reduce_dims import reduce
from prototype.context_embeddings.trim_contexts import trim
from utils import render_args


def embed_contexts(args):
    """
    :param args: Related to where to find context dataframe(s) from which this ETL pipeline commences
    :return: Output directories where final processed dataset with embeddings is located
    """
    prototye_fn = 'data/{}_contexts.csv'.format(args.data_purpose)
    print('Moving data from {} to {} and adding row_idx column'.format(args.context_input_fn, prototye_fn))
    if not os.path.exists(prototye_fn) or not args.use_cached:
        prototye_df = pd.read_csv(args.context_input_fn)
        N = prototye_df.shape[0]
        prototye_df['row_idx'] = list(range(N))
        prototye_df.to_csv(prototye_fn, index=False)

    print('Trimming contexts before passing to BERT...')
    trimmed_fp = trim(args.sf_identifier, args.window_size, prototye_fn, use_cached=args.use_cached)

    print('Getting BERT embeddings for contexts...')
    full_bert_fp = embed_bert(trimmed_fp, args.data_purpose, batch_size=args.batch_size, use_cached=args.use_cached)

    print('Reducing dimensions of BERT embeddings for both LF and SF...')
    full_reduced_fp = reduce(full_bert_fp, use_cached=args.use_cached)
    return full_reduced_fp


if __name__ == '__main__':
    # Pre-trained weight file paths
    parser = argparse.ArgumentParser(description='Embed Contexts from MIMIC & Columbia for Prototype Acronyms.')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--bert_layers', default=13, type=int)
    parser.add_argument('--context_input_fn',
                        default='~/acronyms/prototype/context_extraction/data/downsampled_prototype_contexts.csv')
    parser.add_argument('--embed_dims', default=768, type=int)
    parser.add_argument('--sf_identifier', default='TARGETWORD')
    parser.add_argument('-use_cached', default=False, action='store_true')
    parser.add_argument('--window_size', default=10, type=int, help='Number of tokens by which to truncate context')
    parser.add_argument('--data_purpose', default='prototype', help='prototype or eval')

    args = parser.parse_args()
    args.context_input_fn = os.path.expanduser(args.context_input_fn)
    render_args(args)

    final_fn = embed_contexts(args)
    print('Final dataset is located --> {}'.format(final_fn))
