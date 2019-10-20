import json
import os
import pandas as pd

from utils import standardize_lower, standardize_upper


NULL_LFS = ['no results']


def merge_sources(sources, acronym_list, use_cached):
    out_fn = './data/derived/merged_acronym_expansions.csv'
    if use_cached and os.path.exists(out_fn):
        return out_fn

    cols = ['sf', 'lf', 'source']
    df_arr = []
    for src in sources:
        in_fn = os.path.join('./data', 'derived', '{}_acronym_expansions.json'.format(src))
        data = json.load(open(in_fn, 'r'))
        print('Merging {} with {} acronyms + expansions ...'.format(src, len(data)))
        for sf, lfs in data.items():
            sf = standardize_upper(sf)
            if sf in acronym_list:
                lfs = list(set(list(map(standardize_lower, lfs))))
                for lf in lfs:
                    if lf not in NULL_LFS:
                        df_arr.append([
                            sf,
                            lf,
                            src
                        ])

    df = pd.DataFrame(df_arr, columns=cols)
    df.drop_duplicates(inplace=True)
    df.sort_values(['sf', 'lf', 'source'], inplace=True)
    df.to_csv(out_fn, index=False)
    print('Merged {} acronym-expansion pairs.'.format(df.shape[0]))
    return out_fn
