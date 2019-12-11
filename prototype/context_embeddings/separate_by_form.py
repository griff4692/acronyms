import os
import pickle

import numpy as np
from tqdm import tqdm


def separate(in_fp, data_purpose, use_cached=False):
    out_fp = in_fp.split('.')[0] + '_separated.pk'
    if os.path.exists(out_fp) and use_cached:
        return out_fp

    with open(in_fp, 'rb') as fd:
        data = pickle.load(fd)
    N = len(data['row_idx'])
    data_els = data.keys()
    sep_data = {}
    for n in tqdm(range(N)):
        form = data['form'][n] if data_purpose == 'prototype' else data['sf'][n]
        if form not in sep_data:
            new_obj = {}
            for col in data_els:
                new_obj[col] = []
            sep_data[form] = new_obj
        for col in data_els:
            sep_data[form][col].append(data[col][n])
    for k in sep_data:
        sep_data[k]['embeddings'] = np.array(sep_data[k]['embeddings'])
    with open(out_fp, 'wb') as fd:
        pickle.dump(sep_data, fd)
