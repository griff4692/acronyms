import os

import pandas as pd


def get_form_counts(df, form):
    return df[df['form'] == form].shape[0]


def add_counts(acronyms_fn, contexts_fn, use_cached=False):
    out_fn = acronyms_fn.split('.')[0] + '_w_counts.csv'

    if os.path.exists(out_fn) and use_cached:
        return out_fn

    df = pd.read_csv(contexts_fn)
    acronyms = pd.read_csv(acronyms_fn)

    sf_counts = []
    lf_counts = []
    sf_counts_map = {}
    for idx, row in acronyms.iterrows():
        row = row.to_dict()
        sf = row['sf']
        lf = row['lf']
        if sf not in sf_counts_map:
            sf_counts_map[sf] = get_form_counts(df, sf)
        sf_counts.append(sf_counts_map[sf])
        lf_counts.append(get_form_counts(df, lf))
    acronyms['sf_count'] = sf_counts
    acronyms['lf_count'] = lf_counts
    acronyms.to_csv(out_fn, index=False)


if __name__ == '__main__':
    add_counts('data/merged_prototype_expansions.csv', 'data/all_prototype_contexts.csv')
