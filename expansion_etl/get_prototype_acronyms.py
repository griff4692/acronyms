import argparse
import pandas as pd


if __name__ == '__main__':
    """
    Get final ETL output for subset of acronyms (~10) on which we are developing our model.
    """
    parser = argparse.ArgumentParser(description='Get ETL Output for prototype acronyms only.')
    args = parser.parse_args()

    INPATH = './data/derived/merged_lf/'
<<<<<<< HEAD
    df = pd.read_csv('./data/derived/standardized_acronym_expansions_inclusion_filtered.csv')

    acronyms = set(map(lambda x: x.strip(),
        open('./data/original/prototype_acronyms.txt', 'r'))
    )

=======
    acronyms = set(map(lambda x: x.strip(),
        open('./data/original/prototype_acronyms.txt', 'r'))
    )
>>>>>>> 0c3cc09... Extraction plan for Columbia Discharge Summaries
    df = None
    for acronym in acronyms:
        acronym_df = pd.read_csv('{}/{}.csv'.format(INPATH, acronym))
        df = acronym_df if df is None else pd.concat([df, acronym_df])
    df.to_csv('./data/derived/prototype_acronym_expansions.csv', index=False)
