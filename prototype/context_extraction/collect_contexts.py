import pandas as pd


def collect(acronyms_fp, source_paths):
    source_dfs = []
    for source_name, source_fn in source_paths:
        source_df = pd.read_csv(source_fn)
        source_df['source'] = [source_name] * source_df.shape[0]
        source_dfs.append(source_name)
    full_df = pd.concat(source_dfs, sort=False, axis=0)

    acronyms_df = pd.read_csv(acronyms_fp)
    lfs = acronyms_df['lf'].unique().tolist()
    sfs = acronyms_df['sf'].unique().tolist()
    form_to_sf = {}
    for lf in lfs:
        form_to_sf[lf] = acronyms_df[acronyms_df['lf'] == lf]['sf'].tolist()[0]
    for sf in sfs:
        form_to_sf[sf] = sf

    full_df.dropna(inplace=True)
    out_fn = 'data/all_prototype_contexts.csv'
    print('Saving a whopping {} contexts to {}'.format(full_df.shape[0], out_fn))

    # Append requisite short forms to the dataframe
    full_df['sf'] = full_df['form'].apply(lambda k: form_to_sf[k])
    full_df.to_csv(out_fn, index=False)

    return out_fn
