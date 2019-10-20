import collections
import json
import os

from abbreviations import schwartz_hearst


if __name__ == '__main__':
    lt_files = os.listdir('./abstracts')
    acronyms = collections.defaultdict(list)
    for fname in lt_files:
        if fname.endswith('.txt'):
            pairs = schwartz_hearst.extract_abbreviation_definition_pairs(file_path='./abstracts/' + fname)
            for acronym, expansion in pairs.items():
                acronyms[acronym].append(expansion)

    with open('pubmed_acronym_expansions.json', 'w') as f:
        json.dump(acronyms, f)
