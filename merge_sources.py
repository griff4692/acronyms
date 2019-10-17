import json
import pandas as pd


if __name__ == '__main__':
    umls_fn = 'acronym_expansions_UMLS.json'
    src_info = [
        (umls_fn, 'umls')
    ]

    cols = ['sf', 'lf', 'source']
    df_arr = []
    for src in src_info:
        data = json.load(open(src[0], 'r'))
        for sf, lfs in data.items():
            for lf in lfs:
                if not lf == 'NO RESULTS':
                    df_arr.append([
                        sf,
                        lf,
                        src[1]
                    ])

    df = pd.DataFrame(df_arr, columns=cols)
    df.to_csv('raw_source_data.csv', index=False)
