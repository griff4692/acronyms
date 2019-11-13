import os

import pandas as pd


if __name__ == '__main__':
    base_dir = '../data/derived/'
    columbia_df = pd.read_csv(os.path.join(base_dir, 'columbia_prototype_contexts.csv'))

    mimic_dfs = []
    mimic_dir = os.path.join(base_dir, 'mimic')
    mimic_chunks = os.listdir(mimic_dir)
    print('Collecting chunks of processed MIMIC contexts...')
    for chunk_idx, fn in enumerate(mimic_chunks):
        mimic_dfs.append(pd.read_csv(os.path.join(mimic_dir, fn)))
        if (chunk_idx + 1) % 25 == 0 or (chunk_idx + 1) == len(mimic_chunks):
            print('\tProcessed {} out of {} MIMIC batches'.format(chunk_idx + 1, len(mimic_chunks)))
    mimic_df = pd.concat(mimic_dfs, sort=False)
    full_df = pd.concat([mimic_df, columbia_df], sort=False)
    full_df.dropna(inplace=True)
    out_fn = os.path.join(base_dir, 'all_prototype_contexts.csv')
    print('Saving a whopping {} contexts to {}'.format(full_df.shape[0], out_fn))
    full_df.to_csv(out_fn, index=False)
