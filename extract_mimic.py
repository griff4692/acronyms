import json

import pandas as pd

from expansion_etl.extract_context import ContextExtractor, ContextType


BOUNDARY = '<DOCSTART>'


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


def dump_batch_contexts(contexts, line_idx):
    df = pd.DataFrame(contexts, columns=['form', 'doc_id', 'context'])
    print('Saving {} contexts from MIMIC with end index={}.'.format(df.shape[0], line_idx))
    df.to_csv('./expansion_etl/data/derived/mimic/prototype_contexts_{}.csv'.format(line_idx), index=False)
    json.dump(contexts, open('./expansion_etl/data/derived/mimic/prototype_contexts_{}.json'.format(line_idx), 'w'))


if __name__ == '__main__':
    context_extractor = ContextExtractor()
    batch_contexts = []
    df = pd.read_csv('./expansion_etl/data/derived/prototype_acronym_expansions.csv')

    sfs = df['sf'].unique().tolist()
    lfs = df['lf'].unique().tolist()

    doc_string = ''
    file = open('./expansion_etl/data/derived/clean_text.txt', 'r')
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
            dump_batch_contexts(batch_contexts, line_idx + 1)
            batch_contexts = []
        final_line_idx = line_idx

    doc_id = '{}_{}'.format('mimic', final_line_idx)
    if len(doc_string) > 0:
        doc_contexts = get_all_contexts(context_extractor, doc_id, doc_string, sfs, lfs)
        batch_contexts += doc_contexts
    if len(batch_contexts) > 0:
        dump_batch_contexts(batch_contexts, final_line_idx)

    print('Done processing MIMIC notes!')
