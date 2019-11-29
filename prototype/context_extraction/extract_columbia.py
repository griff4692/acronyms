import os

import pandas as pd

from extract_context_utils import ContextExtractor, ContextType


def get_all_contexts(context_extractor, doc_id, doc_string, sfs, lfs):
    contexts, doc_ids, forms = [], [], []
    config = {
        'type': ContextType.WORD,
        'size': 25
    }

    for sf in sfs:
        c = context_extractor.get_contexts_for_short_form(sf, doc_string, config, allow_inflections=False,
                                                          ignore_case=False)
        forms += [sf] * len(c)
        contexts += c

    for lf in lfs:
        c = context_extractor.get_contexts_for_long_form(lf, doc_string, config, allow_inflections=True,
                                                         ignore_case=True)
        forms += [lf] * len(c)
        contexts += c
    doc_ids += [doc_id] * len(contexts)
    return list(zip(forms, doc_ids, contexts))


def extract_columbia_contexts(in_fp, use_cached=False):
    out_fn = 'data/columbia_prototype_contexts.csv'
    if use_cached and os.path.exists(out_fn):
        return out_fn

    years = list(range(1988, 2015))
    context_extractor = ContextExtractor()
    df = pd.read_csv(in_fp)

    sfs = df['sf'].unique().tolist()
    lfs = df['lf'].unique().tolist()

    tmp_batch_dir = 'data/columbia_context_batches'

    for year in years:
        contexts = []
        print('Processing Columbia Discharge summaries in {}'.format(year))
        year_out_fn = os.path.join(tmp_batch_dir, 'columbia_prototype_contexts_{}.csv'.format(year))
        doc_string = ''

        if os.path.exists(year_out_fn):
            print('Skipping year={} because its already stored in {}'.format(year, year_out_fn))
            continue
        # Must be run on Columbia's DBMI rashi server
        file = open('/nlp/projects/medical_wsd/data/dsum_corpus_02/{}'.format(year), 'r')
        for line_idx, line in enumerate(file):
            if line == '\n':
                doc_id = '{}_{}'.format(line_idx, year)
                doc_contexts = get_all_contexts(context_extractor, doc_id, doc_string, sfs, lfs)
                contexts += doc_contexts
                doc_string = ''
            else:
                doc_string += line
        df = pd.DataFrame(contexts, columns=['form', 'doc_id', 'context'])
        print('Saving {} contexts from Columbia discharge summaries for year={}.'.format(df.shape[0], year))
        df.to_csv(year_out_fn, index=False)
    print('Done processing Columbia discharge summaries!')

    columbia_chunk_dfs = []
    columbia_chunks = os.listdir(tmp_batch_dir)
    print('Collecting chunks of processed Columbia contexts...')
    for chunk_idx, fn in enumerate(columbia_chunks):
        chunk_df = pd.read_csv(os.path.join(tmp_batch_dir, fn))
        if chunk_df.shape[0] > 0:
            columbia_chunk_dfs.append(chunk_df)
    columbia_df = pd.concat(columbia_chunk_dfs, sort=False, axis=0)
    columbia_df.drop_duplicates(inplace=True)
    columbia_df.to_csv(out_fn, index=False)
    print('Done processing MIMIC notes!')

    return out_fn


if __name__ == '__main__':
    extract_columbia_contexts('data/merged_prototype_expansions.csv')
