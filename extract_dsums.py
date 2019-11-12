import pandas as pd

from extract_context import ContextExtractor, ContextType


def get_all_contexts(context_extractor, doc_id, doc_string, sf, lfs):
    contexts = []
    forms = []
    doc_ids = []

    config = {
        'type': ContextType.WORD,
        'size': 100
    }

    for lf in lfs:
        contexts += context_extractor.get_contexts_for_long_form(lf, doc_string, config, allow_inflections=True,
                                                                 ignore_case=True)
    contexts += context_extractor.get_contexts_for_short_form(sf, doc_string, config, allow_inflections=False,
                                                              ignore_case=False)

    doc_ids += [doc_id] * len(contexts)
    return zip(contexts, forms, doc_ids)


if __name__ == '__main__':
    years = list(range(1988, 2015))
    context_extractor = ContextExtractor()
    contexts = []
    doc_ids = []

    df = pd.read_csv('./data/derived/prototype_acronym_expansions_final.csv')

    print(df.shape)

    for year in years:
        doc_string = ''
        start_line_idx = 0
        file = open('/nlp/projects/medical_wsd/data/dsum_corpus_02/2014', 'r')
        for line_idx, line in enumerate(file):
            if line == '\n':
                doc_id = '{}_{}'.format(line_idx, year)
                contexts += list(get_all_contexts(context_extractor, doc_id, doc_string, sf, lfs))
                doc_string = line
                start_line_idx = line_idx
            else:
                doc_string += line

        doc_string = ''