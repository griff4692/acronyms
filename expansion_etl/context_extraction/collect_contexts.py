import os

import pandas as pd


if __name__ == '__main__':
    base_dir = '../data/derived/'
    columbia_df = pd.read_csv(os.path.join(base_dir, 'columbia_prototype_contexts.csv'))

    acronyms = pd.read_csv(os.path.join(base_dir, 'prototype_acronym_expansions.csv'))

    lfs = acronyms['lf'].unique().tolist()
    sfs = acronyms['sf'].unique().tolist()

    mimic_dfs = []
    mimic_dir = os.path.join(base_dir, 'mimic')
    mimic_chunks = os.listdir(mimic_dir)
    mimic_chunks = [m for m in mimic_chunks if '.csv' in m]
    print('Collecting chunks of processed MIMIC contexts...')
    for chunk_idx, fn in enumerate(mimic_chunks):
        chunk_df = pd.read_csv(os.path.join(mimic_dir, fn))
        if chunk_df.shape[0] > 0:
            mimic_dfs.append(chunk_df)
        if (chunk_idx + 1) % 25 == 0 or (chunk_idx + 1) == len(mimic_chunks):
            print('\tProcessed {} out of {} MIMIC batches'.format(chunk_idx + 1, len(mimic_chunks)))
    full_df = pd.concat(mimic_dfs + [columbia_df], sort=False, axis=0)
    full_df.dropna(inplace=True)
    out_fn = os.path.join(base_dir, 'all_prototype_contexts.csv')
    print('Saving a whopping {} contexts to {}'.format(full_df.shape[0], out_fn))

    # Append requisite short forms to the dataframe
    context_sfs = []
    for idx, row in full_df.iterrows():
        form = row.to_dict()['form']
        sf = form if form in sfs else acronyms[acronyms['lf'] == form]['sf'].tolist()[0]
        context_sfs.append(sf)
    full_df.to_csv(out_fn, index=False)