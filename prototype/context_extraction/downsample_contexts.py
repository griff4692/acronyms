import os

import numpy as np
import pandas as pd


MAX_LF_EXAMPLES = 1000
MAX_SF_EXAMPLES = 50000


def downsample(in_fn, use_cached=False):
    out_fn = 'data/downsampled_prototype_contexts.csv'
    if os.path.exists(out_fn) and use_cached:
        return out_fn

    df = pd.read_csv(in_fn)
    pre_N = df.shape[0]
    sfs = df['sf'].unique().tolist()
    forms = df['form'].unique().tolist()
    sampled_dfs = []
    for form in forms:
        form_df = df[df['form'] == form]
        N = form_df.shape[0]
        if N > MAX_LF_EXAMPLES and form not in sfs:
            print('Reducing'
                  ' LF examples for {} from {} to {}'.format(form, N, MAX_LF_EXAMPLES))
            form_df = form_df.sample(n=MAX_LF_EXAMPLES, random_state=1992)

        if N > MAX_SF_EXAMPLES and form in sfs:
            # Truncate by keeping fraction of document ids
            target_fract = MAX_SF_EXAMPLES / float(N)
            doc_ids = form_df['doc_id'].unique()
            keep_doc_ids = list(np.random.choice(doc_ids, size=int(target_fract * len(doc_ids)), replace=False))
            assert len(keep_doc_ids) == len(list(set(keep_doc_ids)))
            form_df = form_df[form_df['doc_id'].isin(keep_doc_ids)]
            print('Reducing SF examples for {} from {} to {}'.format(form, N, form_df.shape[0]))
        sampled_dfs.append(form_df)
    sampled_df = pd.concat(sampled_dfs, sort=False, axis=0)
    N = sampled_df.shape[0]
    print('Reduced overall number of contexts from {} to {}'.format(pre_N, N))
    assert sampled_df.drop_duplicates().shape[0] == N
    sampled_df.to_csv(out_fn, index=False)


if __name__ == '__main__':
    in_fn = 'data/all_prototype_contexts.csv'
    downsample(in_fn)
