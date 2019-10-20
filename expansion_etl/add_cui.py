from collections import defaultdict
import json
import os

import pandas as pd
from pymetamap import MetaMap

NULL_CUI = 'N/A'

SEMANTIC_GROUP_MAP = defaultdict(int)
SEMTYPE_TO_GROUP = {}
GROUP_TO_SEMTYPES = defaultdict(list)
SEMSF_TO_SEMTYPE = {}


def parse_concepts(concepts):
    if concepts is None:
        return 'N/A'
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
    for ridx, lf in enumerate(df['lf'].tolist()):
        concepts = mm.extract_concepts(
            [lf], derivational_variants=True, ignore_word_order=True, ignore_stop_phrases=True)
        cui_str, type_str, group_str = parse_concepts(concepts[0])
        cuis.append(cui_str)
        semtypes.append(type_str)
        semgroups.append(group_str)
        if ridx + 1 % 100 == 0:
            print('{} out of {}'.format(ridx + 1, df.shape[0]))

    df['cui'] = cuis
    df['semtypes'] = semtypes
    df['semgroups'] = semgroups

    json.dump(SEMANTIC_GROUP_MAP, open('./data/derived/semantic_group_counts.json', 'w'))

    df.to_csv(out_fp, index=False)
    return out_fp
