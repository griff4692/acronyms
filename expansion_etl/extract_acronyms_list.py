import os
import json

from utils import standardize_upper


def extract_acronyms_list(fn, use_cached=False):
    out_fn = './data/derived/acronyms.json'
    if use_cached and os.path.exists(out_fn):
        return out_fn
    acronym_file = open(fn, 'r')
    acronym_list = []
    for ac in acronym_file.readlines():
        acronym = standardize_upper(ac.split('\t')[0])
        if len(acronym) > 1:
            acronym_list.append(acronym)

    print('Saving {} acronyms to {}.'.format(len(acronym_list), out_fn))
    json.dump(acronym_list, open(out_fn, 'w'))

    return out_fn
