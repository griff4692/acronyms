import os

from joblib import load

import editdistance as ed
import pandas as pd
import numpy as np
from nltk.corpus import stopwords


STOPWORDS = set(stopwords.words('english'))


def filter_inclusion(in_fp, use_cached=False):
    out_fn = './data/derived/standardized_acronym_expansions_inclusion_filtered.csv'
    if use_cached and os.path.exists(out_fn):
        return out_fn

    df = pd.read_csv(in_fp)

    transformer = load('./data/inclusion_transformer.joblib')
    estimator = load('./data/inclusion_estimator.joblib')

    df = df[~((df['semgroups'].isnull()) & (df['semtypes'].isnull()))]
    df.reset_index(inplace=True, drop=True)

    X_str = []
    for _, row in df.iterrows():
        try:
            row = row.to_dict()
            x = {}
            sem_groups = row['semgroups'].split('|')
            for sem_group in sem_groups:
                x[sem_group] = 1.0
            lfs = row['lf'].split('|')
            x['support'] = len(lfs)

            sem_types = row['semtypes'].split('|')
            for sem_type in sem_types:
                x[sem_type] = 1.0
            X_str.append(x)
        except:
            # Skip rows where semgroups, semtypes is nan
            print(row) # Already filtered, should print nothing
            pass
    X = transformer.transform(X_str)

    num_features = 5
    new_f = np.zeros((X.shape[0], num_features))
    X = np.hstack((X, new_f))
    for index, row in df.iterrows():
        sf = row['sf']
        lf = row['lf']
        lf_base = row['lf_base']
        sflf = ''.join([x[0] for x in lf.split(' ') if x])
        sflf_base = ''.join([x[0] for x in lf_base.split(' ') if x])
        X[index, 100] = ed.eval(sf.lower(), sflf.lower())
        X[index, 101] = ed.eval(sf.lower(), sflf_base.lower())
        X[index, 102] = len(lf_base)
        X[index, 103] = len([x for x in lf.split(' ') if x in STOPWORDS])
        X[index, 104] = 1 if row['source'] == "pubmed" else 0

    is_valid = estimator.predict(X)
    df['is_valid'] = is_valid.tolist()

    df = df[df['is_valid'] == 1]
    df.drop(['is_valid'], axis=1, inplace=True)
    df.to_csv(out_fn, index=False)
    return out_fn
