import numpy as np
import pandas as pd

N = 300
np.random.seed(1992)


if __name__ == '__main__':
    df = pd.read_csv('../data/derived/merged_acronym_expansions_with_merged_cuis.csv')
    acronym_counts = df['sf'].value_counts().to_dict()
    acronym_multiexpansion = []
    for acronym, v in acronym_counts.items():
        if v > 1:
            acronym_multiexpansion.append(acronym)
    sampled_acronyms = np.random.choice(acronym_multiexpansion, N, replace=False)

    cols = ['sf',
            'lf_1', 'source_1', 'cui_1', 'semtypes_1', 'semgroups_1',
            'lf_2', 'source_2', 'cui_2', 'semtypes_2', 'semgroups_2']

    dataset = []
    for acronym in sampled_acronyms:
        acronym_df = df[df['sf'] == acronym]
        num_expansions = acronym_df.shape[0]
        expansion_pair_idxs = np.random.choice(np.array(np.arange(num_expansions)), 2, replace=False)

        dataset_row = [acronym]
        for eidx in expansion_pair_idxs:
            row = acronym_df.iloc[eidx].to_dict()
            dataset_row += [
                row['lf'],
                row['source'],
                row['cui'],
                row['semtypes'],
                row['semgroups']
            ]

        dataset.append(dataset_row)

    df = pd.DataFrame(dataset, columns=cols)
    # 2/3s train / 1/3 test
    df['is_train'] = [0] * (N // 3) + [1] * (2 * N // 3)
    df.to_csv('../data/ml/merge_dataset.csv', index=False)