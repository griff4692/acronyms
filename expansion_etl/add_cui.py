from collections import defaultdict
import json
import os
import traceback

import numpy as np
import pandas as pd
from pymetamap import MetaMap

NULL_CUI = 'N/A'

SEMANTIC_GROUP_MAP = defaultdict(int)
SEMTYPE_TO_GROUP = {}
GROUP_TO_SEMTYPES = defaultdict(list)
SEMSF_TO_SEMTYPE = {}


class ErrorConcept:
    """
    Sometimes pymetamap throws unknown error.  This will affect all in a batch.
    Label them as errors with proper index before
    """
    def __init__(self, index):
        self.index = index


def parse_concepts(concepts):
    if len(concepts) == 1 and type(concepts[0]) == ErrorConcept:
        return 'error', 'error', 'error'

    cuis = set()
    semtypes = set()
    semgroups = set()
    for concept in concepts:
        cuis.add(concept.cui)
        semcodes = concept.semtypes[1:-1].split(',')
        for sc in semcodes:
            semtype = SEMSF_TO_SEMTYPE[sc]
            semtypes.add(semtype)
            semgroup = SEMTYPE_TO_GROUP[semtype]
            semgroups.add(semgroup)
            SEMANTIC_GROUP_MAP[semgroup] += 1

    cui_str = '|'.join(list(sorted(cuis)))
    type_str = '|'.join(list(sorted(semtypes)))
    group_str = '|'.join(list(sorted(semgroups)))
    return cui_str, type_str, group_str


def add_cui(in_fp, use_cached=False):
    out_fp = './data/derived/merged_acronym_expansions_with_cui.csv'
    if use_cached and os.path.exists(out_fp):
        return out_fp

    semtype_data = open('./data/original/umls_semantic_types.txt', 'r')
    for line in semtype_data:
        split = line.strip().split('|')
        SEMSF_TO_SEMTYPE[split[0]] = split[-1]

    semgroup_data = open('./data/original/umls_semantic_groups.txt', 'r')
    for line in semgroup_data:
        split = line.strip().split('|')
        group = split[1]
        semtype = split[-1]
        SEMTYPE_TO_GROUP[semtype] = group
        GROUP_TO_SEMTYPES[group].append(semtype)

    mm_path = os.path.expanduser('~/Desktop/public_mm/bin/metamap18')
    mm = MetaMap.get_instance(mm_path, version='metamap18')

    df = pd.read_csv(in_fp)
    cuis, semtypes, semgroups = [], [], []

    long_forms = df['lf'].tolist()
    concepts = []

    bsize = 100
    batches = range(0, len(long_forms), bsize)
    for bidx, start_idx in enumerate(batches):
        end_idx = min(len(long_forms), start_idx + bsize)
        lf_chunk = long_forms[start_idx:end_idx]
        try:
            c = mm.extract_concepts(lf_chunk, ids=list(range(start_idx, end_idx)), derivational_variants=True,
                                    ignore_word_order=True, ignore_stop_phrases=True)[0]
        except Exception as e:
            print("Type error = {}".format(str(e)))
            print(traceback.format_exc())
            c = [ErrorConcept(i) for i in range(start_idx, end_idx)]
        concepts += c
        print('Processed batch {} out of {}'.format(bidx + 1, len(batches)))

    np.save(open('tmp.npy', 'w'), concepts)
    prev_id = -1
    curr_concepts = []
    for concept in concepts:
        curr_id = int(concept.index)
        if curr_id == prev_id:
            curr_concepts.append(concept)
        else:
            if len(curr_concepts) > 0:
                cui_str, type_str, group_str = parse_concepts(curr_concepts)
                assert int(curr_concepts[-1].index) == len(cuis)
                cuis.append(cui_str)
                semtypes.append(type_str)
                semgroups.append(group_str)
            for _ in range(prev_id + 1, curr_id):
                cuis.append(NULL_CUI)
                semtypes.append(NULL_CUI)
                semgroups.append(NULL_CUI)
            prev_id = curr_id
            curr_concepts = [concept]
    # Process Last batch
    cui_str, type_str, group_str = parse_concepts(curr_concepts)
    cuis.append(cui_str)
    semtypes.append(type_str)
    semgroups.append(group_str)

    df['cui'] = cuis
    df['semtypes'] = semtypes
    df['semgroups'] = semgroups
    json.dump(SEMANTIC_GROUP_MAP, open('./data/derived/semantic_group_counts.json', 'w'))
    df.to_csv(out_fp, index=False)
    return out_fp
