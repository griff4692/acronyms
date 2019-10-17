import os

import pandas as pd

from searchTermsUMLS import search


if __name__ == '__main__':
    df = pd.read_csv('raw_source_data.csv')

    cuis = []
    for term in df['lf'].tolist():
        print('here')
        cui_results = list(map(lambda x: x['ui'], search(term, os.environ['UMLS_API'])))
        if len(cui_results) == 0:
            print('Couldn\'t find CUI for term={}'.format(term))
        else:
            cui_str = ','.join(cui_results)
            cuis.append(cui_str)
            if len(cui_results) > 1:
                print('Multiple CUIS={} found for term={}'.format(cui_str, term))
    print(len(cuis), len(df))
