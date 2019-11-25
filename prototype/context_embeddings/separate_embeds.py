from collections import defaultdict
import json

import numpy as np
import pandas as pd


if __name__ == '__main__':
    with open('data/key_order.json', 'r') as fd:
        keys = json.load(fd)

    with open('data/embeddings.npy', 'rb') as fd:
        embeddings = np.load(fd)

    acronyms_df = pd.read_csv('../expansion_etl/data/derived/prototype_acronym_expansions_w_counts.csv')
    acronyms_df = acronyms_df[acronyms_df['lf_count'] > 0]
    sfs = acronyms_df['sf'].unique().tolist()
    lfs = acronyms_df['lf'].unique().tolist()

    for forms in [('sf', sfs), ('lf', lfs)]:
        embedding_dict = {}
        forms_str, forms_arr = forms
        for form in forms_arr:
            embedding_dict[form] = defaultdict(list)
        for key_idx, key in enumerate(keys):
            form = key[1]
            if form in forms_arr:
                key = (key[0], key[2])
                embedding_dict[form]['embeddings'].append(embeddings[key_idx, :])
                embedding_dict[form]['keys'].append(key)
        for form, form_dict in embedding_dict.items():
            np.save(open('data/{}_embeddings/{}.npy'.format(forms_str, form), 'wb'), np.array(form_dict['embeddings']))
            json.dump(form_dict['keys'], open('data/{}_embeddings/{}_keys.json'.format(forms_str, form), 'w'))
