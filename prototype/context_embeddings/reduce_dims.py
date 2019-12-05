from collections import defaultdict
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def compute_lf_sf_mappings(input_fn):
    df = pd.read_csv(input_fn)
    df = df[df['lf_count'] > 0]
    lf_sf_map = {}
    sf_lf_map = defaultdict(set)
    lfs = df['lf'].unique().tolist()
    for lf in lfs:
        sf = df[df['lf'] == lf].iloc[0].to_dict()['sf']
        lf_sf_map[lf] = sf
        sf_lf_map[sf].add(lf)
    return sf_lf_map, lf_sf_map


if __name__ == '__main__':
    sf_embed_dir, lf_embed_dir = 'data/prototype_sf_embeddings', 'data/prototype_lf_embeddings'

    # all_embed_fn = 'data/prototype_embeddings.npy'
    # with open(all_embed_fn, 'rb') as fd:
    #     all_embeddings = np.load(fd)
    #
    # all_embeddings = all_embeddings[:100]
    #
    # print('Training PCA')
    # pca = PCA(n_components=100)
    # pca.fit(all_embeddings)

    sf_lf_map, lf_sf_map = compute_lf_sf_mappings('../context_extraction/data/merged_prototype_expansions_w_counts.csv')

    # sf, form, embedding
    sfs = []
    forms = []
    embeddings = []

    for sf in sf_lf_map.keys():
        sf_embed_fn = os.path.join(sf_embed_dir, '{}.pk'.format(sf))
        with open(sf_embed_fn, 'rb') as fd:
            sf_embeds = pickle.load(fd)['embeddings']
            n = len(sf_embeds)
            forms += [sf] * n
            sfs += [sf] * n
            embeddings += sf_embeds

    for lf in lf_sf_map.keys():
        lf_embed_fn = os.path.join(lf_embed_dir, '{}.pk'.format(lf))
        with open(lf_embed_fn, 'rb') as fd:
            lf_embeds = pickle.load(fd)['embeddings']
            n = len(lf_embeds)
            forms += [lf] * n
            sfs += [lf_sf_map[lf]] * n
            embeddings += lf_embeds

    sep_embeds_df = pd.DataFrame({
        'sf': sfs,
        'forms': forms,
        'embeddings': embeddings
    })

    sep_embeds_df.to_csv('test.csv', index=False)