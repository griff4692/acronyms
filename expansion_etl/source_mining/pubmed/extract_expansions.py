import collections
import json
import os

from abbreviations import schwartz_hearst


def extract_expansions(acronyms, use_cached=True):
    print('Extracting expansions from Pubmed...')
    out_fn = './data/derived/pubmed_acronym_expansions.json'
    # TODO use_cached = True because script is not runnable right now
    use_cached = True
    if use_cached and os.path.exists(out_fn):
        return out_fn
    lt_files = os.listdir('./abstracts')
    acronyms = collections.defaultdict(list)
    for fname in lt_files:
        if fname.endswith('.txt'):
            pairs = schwartz_hearst.extract_abbreviation_definition_pairs(file_path='./abstracts/' + fname)
            for acronym, expansion in pairs.items():
                acronyms[acronym].append(expansion)
    with open(out_fn, 'w') as f:
        json.dump(acronyms, f)
    return out_fn


if __name__ == '__main__':
    extract_expansions([])
