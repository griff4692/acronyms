from collections import defaultdict
import json
import os
import traceback

import pandas as pd
from pymetamap import MetaMap

NULL_CUI = 'N/A'

SEMANTIC_GROUP_MAP = defaultdict(int)
SEMTYPE_TO_GROUP = {}
GROUP_TO_SEMTYPES = defaultdict(list)
SEMSF_TO_SEMTYPE = {}


def merge_bars(string_arr):
    """
    Merge data of the sort ['a|b|c', 'b|c|d'] --> 'a|b|c|d'
    """
    merged = set()
    for str in string_arr:
        for x in str.split('|'):
            merged.add(x)
    return '|'.join(list(merged))


class ErrorConcept:
    """
    Sometimes pymetamap throws unknown error.  This will affect all in a batch.
    Label them as errors with proper index before
    """
    def __init__(self, index):
        self.index = index


def patch_batch_errors(df, mm):
    """
    pymetamap sometimes makes errors. Since we process in batch,
    this means we don't get the results for an entire batch.
    The solution is to look for CUIS we've inserted with dummy ErrorConcept
    and try to re-run pymetamap on these examples individually.
    """
    error_df = df[df['cui'] == 'error']
    error_row_ids = error_df.index.tolist()

    print('Re-processing {} rows with potential errors.'.format(len(error_row_ids)))
    cols = df.columns.tolist()
    cui_idx = cols.index('cui')
    types_idx = cols.index('semtypes')
    groups_idx = cols.index('semgroups')

    lfs = error_df['lf'].tolist()
    for lf, row_idx in zip(lfs, error_row_ids):
        try:
            concepts = mm.extract_concepts([lf])[0]
            cui_str, type_str, group_str = parse_concepts(concepts)
        except Exception as e:
            print("Type error = {}".format(str(e)))
            print(traceback.format_exc())
            cui_str, type_str, group_str = NULL_CUI, NULL_CUI, NULL_CUI
        df.iat[row_idx, cui_idx] = cui_str
        df.iat[row_idx, types_idx] = type_str
        df.iat[row_idx, groups_idx] = group_str
    assert len(df[df['cui'] == 'error']) == 0


def parse_concepts(concepts):
    assert len(concepts) > 0
    if type(concepts[0]) == ErrorConcept:
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
    out_fn = './data/derived/standardized_acronym_expansions_with_cui.csv'
    if use_cached and os.path.exists(out_fn):
        return out_fn

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

    df = pd.read_csv(in_fp)

    cuis, semtypes, semgroups = [], [], []
    mm_path = os.path.expanduser('~/Desktop/public_mm/bin/metamap18')
    mm = MetaMap.get_instance(mm_path, version='metamap18')

    concepts = []
    lfs = df['lf_base'].tolist()
    target_bsize = 1000
    start_idx = 0
    while start_idx < len(lfs):
        end_idx = min(len(lfs), start_idx + target_bsize)
        lf_chunk = lfs[start_idx:end_idx]
        try:
            c = mm.extract_concepts(lf_chunk, ids=list(range(start_idx, end_idx)))[0]
            if len(c) == 0:
                c = [ErrorConcept(i) for i in range(start_idx, end_idx)]
                start_idx = end_idx
            else:
                start_idx = int(c[-1].index) + 1
        except Exception as e:
            print("Type error = {}".format(str(e)))
            print(traceback.format_exc())
            c = [ErrorConcept(i) for i in range(start_idx, end_idx)]
            start_idx = end_idx
        concepts += c
        print('{} out of {} examples processed ({})'.format(start_idx, len(lfs), round(start_idx / float(len(lfs)), 4)))

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

    # May end in NULL CUI
    for _ in range(df.shape[0] - len(cuis)):
        cuis.append(NULL_CUI)
        semtypes.append(NULL_CUI)
        semgroups.append(NULL_CUI)

    df['cui'] = cuis
    df['semtypes'] = semtypes
    df['semgroups'] = semgroups
    patch_batch_errors(df, mm)

    json.dump(SEMANTIC_GROUP_MAP, open('./data/derived/semantic_group_counts.json', 'w'))

    df.dropna(inplace=True)

    old_N = df.shape[0]
    dfg = df.groupby(['sf', 'cui']).agg(
        {
            'lf': merge_bars,
            'lf_base': merge_bars,
            'source': merge_bars,
            'semtypes': lambda x: list(x)[0],
            'semgroups': lambda x: list(x)[0],
        }).reset_index()

    N = dfg.shape[0]
    print('{} out of {} SF-LF tuples share the same set of CUIs'.format(old_N - N, old_N))
    dfg.to_csv(out_fn, index=False)
    return out_fn
