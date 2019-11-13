import json

import pandas as pd

from expansion_etl.extract_context import ContextExtractor, ContextType


def get_all_contexts(context_extractor, doc_id, doc_string, sfs, lfs):
    contexts, doc_ids, forms = [], [], []
    config = {
        'type': ContextType.WORD,
        'size': 100
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


if __name__ == '__main__':
    years = list(range(1988, 2015))
    context_extractor = ContextExtractor()
    contexts = []
    df = pd.read_csv('../data/derived/prototype_acronym_expansions.csv')

    sfs = df['sf'].unique().tolist()
    lfs = df['lf'].unique().tolist()

    context_ct = 0
    for year in years:
        print('Processing Columbia Discharge summaries in {}'.format(year))
        doc_string = ''
        start_line_idx = 0
        # Must be run on Columbia's DBMI rashi server
        file = open('/nlp/projects/medical_wsd/data/dsum_corpus_02/{}'.format(year), 'r')
        for line_idx, line in enumerate(file):
            if line == '\n':
                doc_id = '{}_{}'.format(line_idx, year)
                doc_contexts = get_all_contexts(context_extractor, doc_id, doc_string, sfs, lfs)
                contexts += doc_contexts
                context_ct += len(doc_contexts)
                if len(doc_contexts) > 0:
                    print('New Context Count={}'.format(context_ct))
                doc_string = ''
                start_line_idx = line_idx
            else:
                doc_string += line
    df = pd.DataFrame(contexts, columns=['form', 'doc_id', 'context'])
    print('Saving {} contexts from Columbia discharge summaries.'.format(df.shape[0]))
    df.to_csv('./expansion_etl/data/derived/columbia_prototype_contexts.csv', index=False)
    json.dump(contexts, open('./expansion_etl/data/derived/columbia_prototype_contexts.json', 'w'))
    print('Done processing Columbia discharge summaries!')