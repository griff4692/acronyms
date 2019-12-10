from collections import defaultdict
import numpy as np
import pickle

from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd

DEFAULT_EMBED_FP = '../context_embeddings/data/prototype_contexts_w_embeddings_reduced.pk'
DEFAULT_ACRONYMS_FP = '../context_extraction/data/merged_prototype_expansions_w_counts.csv'


class KNN:
    def __init__(self, embed_fp=DEFAULT_EMBED_FP, acronyms_fp=DEFAULT_ACRONYMS_FP):
        self.embed_fp = embed_fp
        self.acronyms_fp = acronyms_fp

        self.embed_dim = None
        self.embed_data = None
        self.setup()

    def setup(self):
        with open(self.embed_fp, 'rb') as fd:
            pickle_data = pickle.load(fd)

            self.embed_data = {}
            self.embed_dim = pickle_data['embeddings'][0].shape[0]
            acronyms = pd.read_csv(self.acronyms_fp)
            sfs = acronyms['sf'].unique()
            for sf in sfs:
                self.embed_data[sf] = {
                    'embeddings': np.empty((0, self.embed_dim)),
                    'lfs': []
                }

            N = len(pickle_data['row_idx'])
            for i in range(N):
                ce = np.expand_dims(pickle_data['embeddings'][i], 0)
                form = pickle_data['form'][i]
                sf = pickle_data['sf'][i]
                assert sf in sfs
                if form not in sfs:
                    self.embed_data[sf]['embeddings'] = np.concatenate([
                        self.embed_data[sf]['embeddings'],
                        ce
                    ])

                    self.embed_data[sf]['lfs'].append(form)

    def retrieve(self, sf, sf_embed, k=1):
        data = self.embed_data[sf]
        embeds, lfs = data['embeddings'], data['lfs']
        sims = cosine_similarity(embeds, np.expand_dims(sf_embed, axis=1))
        top_ids = np.argsort(sims)[-k:]

        lf_counts = defaultdict(int)
        max_count = 0
        for id in top_ids:
            lf = lfs[id]
            lf_counts[lf] += 1
            max_count = max(max_count, lf_counts[lf])
        for lf, count in lf_counts.items():
            if count == max_count:
                return lf


def evaluate_knn(eval_fp):
    eval_df = pd.read_csv(eval_fp)
    for row_idx, row in eval_df.iterrows():
        row = row.to_dict()
        target_lf = row['target_lf_expansion']


if __name__ == '__main__':
    knn = KNN()
    evaluate_knn('../context_extraction/annotated_prototype_sampled.csv')
