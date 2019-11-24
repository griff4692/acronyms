import numpy as np
import pandas as pd
import json

import torch

if __name__ == '__main__':
    bert_data = torch.load('/home/ga2530/embed_bert/embed/columbia_embeddings_parsed.dict')
    df = pd.read_csv(
        '/home/ga2530/acronyms/expansion_etl/data/derived/sampled_columbia_prototype_contexts_trimmed.csv')
    N = df.shape[0]
    EMBED_DIM = 768
    embeds = np.zeros([N, EMBED_DIM])
    keys = []
    for ct, (idx, row) in enumerate(df.iterrows()):
        row = row.to_dict()
        key = (row['row_idx'], row['form'], row['doc_id'])
        embedding = bert_data[key]
        embeds[ct, :] = embedding
        keys.append(key)
        if (ct + 1) % 1000 == 0 or (ct + 1) == N:
            print('Processed {} out of {} rows'.format(ct + 1, N))
    np.save(open('/home/ga2530/acronyms/context_embeddings/data/columbia_embeddings.npy', 'wb'), embeds)
    json.dump(keys, open('/home/ga2530/acronyms/context_embeddings/data/key_order.json', 'w'))

