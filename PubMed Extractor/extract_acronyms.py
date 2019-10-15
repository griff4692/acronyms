import os
import collections
import json

from abbreviations import schwartz_hearst

lt_files = os.listdir('./abstracts')

acronyms = collections.defaultdict(list)

for fname in lt_files:
    if fname.endswith('.txt'):
        pairs = schwartz_hearst.extract_abbreviation_definition_pairs(file_path='./abstracts/' + fname)
        for acronym, expansion in pairs.items():
            acronyms[acronym].append(expansion)

with open('pubmed_acronyms.json', 'w') as f:
    json.dump(acronyms, f)

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

with open('pubmed_acronyms.pkl', 'wb') as f:
    pickle.dump(acronyms, f, protocol=pickle.HIGHEST_PROTOCOL)