from collections import defaultdict
import json
import os

import numpy as np
import pandas as pd


def separate(data_in_fn, embed_in_fn, data_purpose, use_cached=False):
    """
    :param data_in_fn: filepath to dataframe with contexts, sf, forms, and other metadata
    :param embed_in_fn: filepath to numpy array with embeddings for each corresponding form-context row in data_in_fn
    :param use_cached: if data exists already, whether or not to recalculate / refresh
    :return: list output directories where data has been separated by SF and LF
    """
    with open(embed_in_fn, 'rb') as fd:
        embeddings = np.load(fd)

    data_df = pd.read_csv(data_in_fn)
    sep_embeddings = defaultdict(lambda: {'keys': [], 'embeddings': []})

    sfs = data_df['sf'].unique().tolist()
    lfs = list(set(data_df['form'].unique().tolist()) - set(sfs))

    print('Separating {} SFs and {} LFs'.format(len(sfs), len(lfs)))
    out_dirs = ['data/{}_sf_embeddings'.format(data_purpose), 'data/{}_lf_embeddings'.format(data_purpose)]
    for out_dir in out_dirs:
        if not os.path.exists(out_dir):
            print('Creating directory={}'.format(out_dir))
            os.mkdir(out_dir)

    is_complete = len(os.listdir(out_dirs[0])) == len(sfs) and len(os.listdir(out_dirs[1])) == len(lfs)
    if is_complete and use_cached:
        return out_dirs

    ct_idx = 0
    for row_idx, row in data_df.iterrows():
        assert row_idx == ct_idx
        embedding = embeddings[row_idx, :]
        row = row.to_dict()
        sep_embeddings[row['form']]['keys'].append(row)
        sep_embeddings[row['form']]['embeddings'].append(embedding.tolist())
        ct_idx += 1

    type_strs = ['lf'] * len(lfs) + ['sf'] * len(sfs)
    for form, form_str in zip(lfs + sfs, type_strs):
        out_fn = 'data/{}_{}_embeddings/{}.json'.format(data_purpose, form_str, form)
        print('Saving {} embeddings and context information to {}'.format(form, out_fn))
        with open(out_fn, 'w') as fd:
            json.dump(sep_embeddings[form], fd)

    return out_dirs
