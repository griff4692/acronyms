import json
import os
from nltk.corpus import stopwords
import pandas as pd

from utils import standardize_lower, standardize_upper


NULL_LF = 'no results'
STOPWORDS = set(stopwords.words('english'))
STOPWORDS.add(NULL_LF)


def merge_sources(source_fns, acronym_list, use_cached=True):
    out_fn = './data/derived/merged_acronym_expansions.csv'
    if use_cached and os.path.exists(out_fn):
        return out_fn

    cols = ['sf', 'lf', 'source']
    df_arr = []
    for src, src_fn in source_fns:
        data = json.load(open(src_fn, 'r'))
        print('Merging {} with {} acronyms + expansions ...'.format(src, len(data)))
        for sf, lfs in data.items():
            sf = standardize_upper(sf)
            if sf in acronym_list:
                lfs = list(set(list(map(standardize_lower, lfs))))
                for lf in lfs:
                    if lf not in STOPWORDS and not lf.lower() == sf.lower():
                        df_arr.append([
                            sf,
                            lf,
                            src
                        ])

    df = pd.DataFrame(df_arr, columns=cols)
    df.drop_duplicates(inplace=True)
    df.sort_values(['sf', 'lf', 'source'], inplace=True)
    dfg = df.groupby(['sf', 'lf']).agg({'source': lambda x: '|'.join(list(x))}).reset_index()
    dfg.dropna(inplace=True)
    dfg.to_csv(out_fn, index=False)
    print('Merged {} acronym-expansion pairs.'.format(dfg.shape[0]))
    return out_fn
