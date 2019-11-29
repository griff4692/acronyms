import os

import pandas as pd

from prototype.context_extraction.extract_context_utils import ContextExtractor, ContextType


BOUNDARY = '<DOCSTART>'
PATH_TO_MIMIC = os.path.expanduser('~/Desktop/clean_text.txt')


def get_all_contexts(context_extractor, doc_id, doc_string, sfs, lfs):
    contexts, doc_ids, forms = [], [], []
    config = {'type': ContextType.WORD, 'size': 25}

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


def dump_batch_contexts(contexts, line_idx, tmp_batch_dir):
    if len(contexts) > 0:
        df = pd.DataFrame(contexts, columns=['form', 'doc_id', 'context'])
        print('Saving {} contexts from MIMIC with end index={}.'.format(df.shape[0], line_idx))
        df.to_csv(os.path.join(tmp_batch_dir, 'prototype_contexts_{}.csv'.format(line_idx)), index=False)


def extract_mimic_contexts(in_fp, mimic_fp=PATH_TO_MIMIC, use_cached=False):
    out_fn = 'data/mimic_prototype_contexts.csv'

    if use_cached and os.path.exists(out_fn):
        return out_fn

    context_extractor = ContextExtractor()
    batch_contexts = []
    df = pd.read_csv(in_fp)

    tmp_batch_dir = './data/mimic_context_batches/'
    if not os.path.exists(tmp_batch_dir):
        print('Creating dir={}'.format(tmp_batch_dir))
        os.mkdir(tmp_batch_dir)

    sfs = df['sf'].unique().tolist()
    lfs = df['lf'].unique().tolist()

    doc_string = ''
    with open(mimic_fp, 'r') as file:
        num_docs = 0
        final_line_idx = 0
        for line_idx, line in enumerate(file):
            if line.startswith(BOUNDARY):
                doc_id = '{}_{}'.format('mimic', line_idx)
                if len(doc_string) > 1:
                    doc_contexts = get_all_contexts(context_extractor, doc_id, doc_string, sfs, lfs)
                    batch_contexts += doc_contexts
                    num_docs += 1
                    if num_docs % 1000 == 0:
                        print('Num Docs Processed={}'.format(num_docs))
                doc_string = ''
            else:
                doc_string += line
            if (line_idx + 1) % 10000 == 0:
                dump_batch_contexts(batch_contexts, line_idx + 1, tmp_batch_dir)
                batch_contexts = []
            final_line_idx = line_idx

        print('Extracting final contexts.')
        doc_id = '{}_{}'.format('mimic', final_line_idx)
        if len(doc_string) > 0:
            doc_contexts = get_all_contexts(context_extractor, doc_id, doc_string, sfs, lfs)
            batch_contexts += doc_contexts
        if len(batch_contexts) > 0:
            dump_batch_contexts(batch_contexts, final_line_idx, tmp_batch_dir)

    mimic_chunk_dfs = []
    mimic_chunks = os.listdir(tmp_batch_dir)
    print('Collecting chunks of processed MIMIC contexts...')
    for chunk_idx, fn in enumerate(mimic_chunks):
        chunk_df = pd.read_csv(os.path.join(tmp_batch_dir, fn))
        if chunk_df.shape[0] > 0:
            mimic_chunk_dfs.append(chunk_df)
    mimic_df = pd.concat(mimic_chunk_dfs, sort=False, axis=0)
    mimic_df.drop_duplicates(inplace=True)
    mimic_df.to_csv(out_fn, index=False)
    print('Done processing MIMIC notes!')
    return out_fn


if __name__ == '__main__':
    extract_mimic_contexts('./expansion_etl/data/derived/prototype_acronym_expansions.csv')

