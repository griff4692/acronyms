import os
import pandas as pd


if __name__ == '__main__':
    base_path = 'data/derived'
    fns = [
        'merged_acronym_expansions.csv',
        'prototype_acronym_expansions.csv',
        'standardized_acronym_expansions.csv',
        'standardized_acronym_expansions_with_cui.csv',
        'standardized_acronym_expansions_inclusion_filtered.csv',
    ]

    for fn in fns:
        df = pd.read_csv(os.path.join(base_path, fn))
        df_md = df[df['sf'] == 'CRT']
        cols = list(df_md.columns)
        relevant_cols = [c for c in cols if 'lf' in c]

        found = False
        found_val = ''
        for col in relevant_cols:
            term = 'refill'
            vals = df_md[col].tolist()
            for val in vals:
                if term in val:
                    found = True
                    found_val = val
                    break
            if found:
                break
        if found:
            print('Found {} in {}'.format(found_val, fn))
        else:
            print('Found nothing in {}'.format(fn))
