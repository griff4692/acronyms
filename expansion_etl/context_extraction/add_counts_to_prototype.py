import os

import pandas as pd


def get_form_counts(df, form):
    return df[df['form'] == form].shape[0]


if __name__ == '__main__':
    base_dir = '../data/derived/'
    df = pd.read_csv(os.path.join(base_dir, 'all_prototype_contexts.csv'))
    acronyms = pd.read_csv(os.path.join(base_dir, '/prototype_acronym_expansions.csv'))

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
    acronyms.to_csv(os.path.join(base_dir, '/prototype_acronym_expansions_w_counts.csv'), index=False)
