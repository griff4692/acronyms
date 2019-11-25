import os

import json
import pandas as pd

import argparse

from prototype.context_extraction.collect_contexts import collect
from prototype.context_extraction.extract_columbia import extract_columbia_contexts
from prototype.context_extraction.extract_mimic import extract_mimic_contexts
from prototype.context_extraction.merge_similar_lfs import merge_concurrently
from utils import render_args


def extract_contexts(args, fp):
    print('Loading prototype acronyms...')
    with open(fp, 'r') as fd:
        prototype_acronyms = list(sorted(map(lambda x: x.strip(), fd.readlines())))
    print('Prototype acronyms: {}'.format(', '.join(prototype_acronyms)))

    # Filtering expansion list for just prototype SFs
    expansions_out_fp = os.path.join('data', 'prototype_expansions.csv')
    if not os.path.exists(expansions_out_fp) or not args.use_cached:
        expansions_df = pd.read_csv(args.etl_output_fn)
        full_N = expansions_df.shape[0]
        expansions_prototype_df = expansions_df[expansions_df['sf'].isin(prototype_acronyms)]
        prototype_N = expansions_prototype_df.shape[0]
        print('Filtering full list of expansions from {} to {} prototype SF-LF pairs'.format(full_N, prototype_N))
        expansions_prototype_df.to_csv(expansions_out_fp, index=False)

    # Merging related pairs
    merge_tmp_dir = './data/merged_lf'
    items = os.listdir(merge_tmp_dir)
    if len(items) < len(prototype_acronyms) or not args.use_cached:
        merge_concurrently(expansions_out_fp, selected_sf=[], save_dir=merge_tmp_dir)

    merge_out_fp = os.path.join('data', 'merged_prototype_expansions.csv')
    if not os.path.exists(merge_out_fp) or not args.use_cached:
        merged_df = None
        for acronym in prototype_acronyms:
            acronym_df = pd.read_csv('{}/{}.csv'.format(merge_tmp_dir, acronym))
            merged_df = acronym_df if merged_df is None else pd.concat([merged_df, acronym_df])
        merged_df.to_csv(os.path.join('data', 'merged_prototype_expansions.csv'), index=False)

    # Extract contexts for MIMIC and Columbia
    mimic_out_fn = extract_mimic_contexts(merge_out_fp)
    columbia_out_fn = extract_columbia_contexts(merge_out_fp)

    combined_fn = collect(merge_out_fp, [('mimic', mimic_out_fn), ('columbia', columbia_out_fn)])
    return combined_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Contexts from MIMIC & Columbia for Prototype Acronyms.')
    parser.add_argument('-use_cached', default=False, action='store_true')
    parser.add_argument('--etl_output_fn',
                        default='../../expansion_etl/data/derived/standardized_acronym_expansions_with_cui.csv')

    args = parser.parse_args()
    render_args(args)

    initial_fp = './data/prototype_acronyms.txt'
    final_fn = extract_contexts(args, initial_fp)
    print('Final dataset is located --> {}'.format(final_fn))
