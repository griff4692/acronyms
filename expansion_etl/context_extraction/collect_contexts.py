import os

import pandas as pd


if __name__ == '__main__':
    base_dir = '../data/derived/'
    columbia_df = pd.read_csv(os.path.join(base_dir, 'columbia_prototype_contexts.csv'))
    acronyms = pd.read_csv(os.path.join(base_dir, 'prototype_acronym_expansions.csv'))

    lfs = acronyms['lf'].unique().tolist()
    sfs = acronyms['sf'].unique().tolist()

    form_to_sf = {}
    for lf in lfs:
        form_to_sf[lf] = acronyms[acronyms['lf'] == lf]['sf'].tolist()[0]
    for sf in sfs:
        form_to_sf[sf] = sf

    mimic_chunk_dfs = []
    mimic_dir = os.path.join(base_dir, 'mimic')
    mimic_chunks = os.listdir(mimic_dir)
    mimic_chunks = [m for m in mimic_chunks if '.csv' in m]
    print('Collecting chunks of processed MIMIC contexts...')
    n_mimic = 0
    for chunk_idx, fn in enumerate(mimic_chunks):
        chunk_df = pd.read_csv(os.path.join(mimic_dir, fn))
        n_mimic += chunk_df.shape[0]
        if chunk_df.shape[0] > 0:
            mimic_chunk_dfs.append(chunk_df)
        if (chunk_idx + 1) % 25 == 0 or (chunk_idx + 1) == len(mimic_chunks):
            print('\tProcessed {} out of {} MIMIC batches'.format(chunk_idx + 1, len(mimic_chunks)))

    mimic_df = pd.concat(mimic_chunk_dfs, sort=False, axis=0)
    mimic_df['source'] = ['mimic'] * mimic_df.shape[0]
    columbia_df['source'] = ['columbia'] * columbia_df.shape[0]
    full_df = pd.concat([mimic_df, columbia_df], sort=False, axis=0)
    full_df.dropna(inplace=True)
    out_fn = os.path.join(base_dir, 'all_prototype_contexts.csv')
    print('Saving a whopping {} contexts to {}'.format(full_df.shape[0], out_fn))

    # Append requisite short forms to the dataframe
    full_df['sf'] = full_df['form'].apply(lambda k: form_to_sf[k])
    full_df.to_csv(out_fn, index=False)
