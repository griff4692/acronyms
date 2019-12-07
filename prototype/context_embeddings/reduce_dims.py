import os
import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def reduce(in_fp, target_dim=100, use_cached=False):
    out_fp = in_fp.split('.')[0] + '_reduced.pk'
    if os.path.exists(out_fp) and use_cached:
        return out_fp

    with open(in_fp, 'rb') as fd:
        data = pickle.load(fd)

    print('Fitting reducer...')
    reducer = PCA(n_components=target_dim)

    print('Reducing existing embeddings...')
    reduced_embeds = reducer.fit_transform(data['embeddings'])
    data['embeddings'] = reduced_embeds
    with open(out_fp, 'wb') as fd:
        pickle.dump(data, fd)
    return out_fp
