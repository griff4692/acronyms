import pandas as pd

N = 300


if __name__ == '__main__':
    df = pd.read_csv('../data/derived/merged_acronym_expansions_with_merged_cuis.csv')
    df = df.sample(n=N, random_state=1992).reset_index(drop=True)

    # 2/3s train / 1/3 test
    df['is_train'] = [0] * (N // 3) + [1] * (2 * N // 3)
    df.to_csv('../data/ml/inclusion_dataset.csv', index=False)
