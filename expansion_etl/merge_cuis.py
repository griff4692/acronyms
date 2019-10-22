import os
import pandas as pd


def merge_sources(source_strings):
    merged_sources = set()
    for source_str in source_strings:
        sources = source_str.split(',')
        for source in sources:
            merged_sources.add(source)
    return ','.join(list(merged_sources))


def merge_lfs_by_cui(in_fn, use_cached=True):
    out_fn = './data/derived/merged_acronym_expansions_with_merged_cuis.csv'
    if use_cached and os.path.exists(out_fn):
        return out_fn

    print('Merging LFs that share the same SF and same set of CUIS (exact match)...')
    df = pd.read_csv(in_fn)
    old_N = df.shape[0]
    dfg = df.groupby(['sf', 'cui']).agg(
        {
            'lf': lambda x: '|'.join(list(set(list(x)))),
            'source': lambda x: merge_sources(list(x)),
            'semtypes': lambda x: list(x)[0],
            'semgroups': lambda x: list(x)[0],
         }).reset_index()

    N = dfg.shape[0]
    print('{} out of {} SF-LF tuples share the same set of CUIs'.format(old_N - N, old_N))
    dfg.to_csv(out_fn, index=False)
    return out_fn
