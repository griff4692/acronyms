import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('../data/derived/sampled_prototype_contexts.csv')
    mimic = df[df['source'] == 'mimic']
    print('Saving MIMIC only dataframe with {} rows.'.format(mimic.shape[0]))
    mimic.to_csv('../data/derived/sampled_mimic_prototype_contexts.csv')
