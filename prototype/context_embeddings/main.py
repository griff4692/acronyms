import os
import sys
sys.path.insert(0, os.path.expanduser('~/acronyms/'))

import argparse
import pandas as pd

from prototype.context_embeddings.embed_contexts import embed_bert
from prototype.context_embeddings.trim_contexts import trim
from utils import render_args


def compute_lf_sf_map(input_fn):
    df = pd.read_csv(input_fn)
    lf_sf_map = {}
    sfs = df['sf'].unique().tolist()
    forms = df['form'].unique().tolist()
    for form in forms:
        sample_form_row = df[df['form'] == form].iloc[0].to_dict()
        if sample_form_row['form'] not in sfs:
            lf_sf_map[sample_form_row['form']] = sample_form_row['sf']

    return sfs, lf_sf_map


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
    return full_bert_fp

    # print('Reducing dimensions of BERT embeddings for both LF and SF...')
    # # Create dict of sf_fn --> [lf_fn, lf_fn, ...]
    # sf_lf_fn_map = defaultdict(list)
    # sfs, lf_sf_map = compute_lf_sf_map(prototye_fn)
    # lf_fns = os.listdir(lf_embed_dir)
    # for lf_fn in lf_fns:
    #     lf, ext = lf_fn.split('.')
    #     sf = lf_sf_map[lf]
    #     sf_fn = os.path.join(sf_embed_dir, '{}.{}'.format(sf, ext))
    #     sf_lf_fn_map[sf_fn].append(os.path.join(lf_embed_dir, lf_fn))
    # # Run PCA individually and visualize
    # run_group_by_sf(sf_lf_fn_map)
    # return sf_embed_dir, lf_embed_dir


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
