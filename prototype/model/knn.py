from collections import defaultdict
import itertools
import pickle

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

DEFAULT_EMBED_FP = '../context_embeddings/data/prototype_contexts_w_embeddings_separated.pk'
DEFAULT_EVAL_DATA = '../context_embeddings/data/eval_contexts_w_embeddings_separated.pk'
DEFAULT_ACRONYMS_FP = '../context_extraction/data/merged_prototype_expansions_w_counts.csv'


def approximate_match(a_pipe, b_pipe):
    a_toks = a_pipe.split('|')
    b_toks = b_pipe.split('|')
    pairs = [(x,y) for x in a_toks for y in b_toks]

    for x, y in pairs:
        if x in y or y in x:
            return True
    return False


class KNN:
    def __init__(self, embed_fp=DEFAULT_EMBED_FP, acronyms_fp=DEFAULT_ACRONYMS_FP):
        self.embed_fp = embed_fp
        self.acronyms_fp = acronyms_fp

        self.embed_dim = None
        self.embed_data = None
        self.setup()

    def setup(self):
        print('Loading data for KNN...')
        with open(self.embed_fp, 'rb') as fd:
            pickle_data = pickle.load(fd)

        self.embed_data = {}
        forms = list(pickle_data.keys())
        for form_idx in tqdm(range(len(forms))):
            form = forms[form_idx]
            form_data = pickle_data[form]
            cols = form_data.keys()
            sf = form_data['sf'][0]
            if sf not in self.embed_data:
                new_obj = {}
                for col in cols:
                    new_obj[col] = []
                self.embed_data[sf] = new_obj
            if not form == sf:
                for col in cols:
                    if type(form_data[col]) == np.ndarray:
                        for el in form_data[col]:
                            self.embed_data[sf][col].append(el)
                    else:
                        self.embed_data[sf][col] += form_data[col]
        for sf in self.embed_data:
            self.embed_data[sf]['embeddings'] = np.array(self.embed_data[sf]['embeddings'])

    def retrieve(self, sf, sf_embed, k=1):
        sf_data = self.embed_data[sf]
        embeds, lfs = sf_data['embeddings'], sf_data['form']
        sims = cosine_similarity(embeds, np.expand_dims(sf_embed, axis=0))
        top_ids = np.argsort(sims.squeeze())[-k:]
        lf_counts = defaultdict(int)
        max_count = 0
        for id in top_ids:
            lf = lfs[id]
            lf_counts[lf] += 1
            max_count = max(max_count, lf_counts[lf])
        for lf, count in lf_counts.items():
            if count == max_count:
                return lf

    def evaluate(self, eval_fp, k=1):
        with open(eval_fp, 'rb') as fd:
            eval_data = pickle.load(fd)
        sfs = eval_data.keys()
        em = 0
        app = 0
        ct = 0
        for sf in sfs:
            for target_lf, embedding in zip(eval_data[sf]['lf'], eval_data[sf]['embeddings']):
                predicted_lf = self.retrieve(sf, embedding, k=k)
                ct += 1
                if predicted_lf == target_lf:
                    em += 1
                if predicted_lf == target_lf or approximate_match(predicted_lf, target_lf):
                    app += 1
        return app / float(ct), em / float(ct)


if __name__ == '__main__':
    for k in range(5, 6):
        app, em = KNN().evaluate(DEFAULT_EVAL_DATA, k=k)
        print('K={}. Approximate Accuracy={}.  Exact Match Accuracy={}'.format(k, app, em))
