import json
import os
from nltk.corpus import stopwords
from distance import aligned_edit_distance, jaccard_overlap
import pandas as pd
from collections import defaultdict
from source_mining.umls.extract_expansions import search
import threading
import concurrent.futures

apikey = 'b3a1775f-e041-43c8-b529-03c843ff8934'
PATH_LF = 'data/derived/standardized_acronym_expansions_with_cui.csv'
KEY_SF = 'sf'
KEY_LF = 'lf_base'
KEY_GROUP = 'semgroups'
THRESHOLD = 0.666
NONCLINICAL_GROUPS = set(['Genes & Molecular Sequences'])
OTHER_KEYS = ['cui', 'source', 'semtypes', 'semgroups']




def merge_lf(path = PATH_LF):
    df = pd.read_csv(path)
    sf = ''
    s = set()
    pre_defined = {}
    other_informations = defaultdict(dict)
    start = False

    for idx, row in df.iterrows():
        if row[KEY_SF] == 'CAGG':
            start = True
        if not start:
            continue
        if idx < 60048:
            continue
        print(idx)
        if not sf:
            sf = row[KEY_SF]
        if sf != row[KEY_SF]:
            print('=====================')
            merge_single_sf(sf, s, pre_defined, other_informations).to_csv('data/derived/merged_lf/' + sf + '.csv', index=False)
            sf = row[KEY_SF]
            s = set()
            pre_defined = {}
            other_informations = defaultdict(dict)

        if '|' in row[KEY_LF]:
            # create dict to store lf with same CUI
            lfs = row[KEY_LF].split('|')
            for lf in lfs[1:]:
                pre_defined[lfs[0]] = lf
        else:
            lfs = [row[KEY_LF]]
        for lf in lfs:
            if not pd.isna(row[KEY_GROUP]) and is_clinical(row[KEY_GROUP]) and search(lf, apikey)[0]['ui'] != 'NONE':
                for key in OTHER_KEYS:
                    other_informations[lf][key] = row[key]
                s.add(lf)

def is_clinical(semgroups):
    semgroups = semgroups.split('|')
    return any(group not in NONCLINICAL_GROUPS for group in semgroups)

def find(unions, i):
    if unions[i] != i:
        return find(unions, unions[i])
    return unions[i]

def union(unions, i1, i2):
    i1 = find(unions, i1)
    i2 = find(unions, i2)
    if i1 != i2:
        unions[i1] = i2

def merge_single_sf(sf, s, pre_defined, other_informations):
    lfs = [key for key in s]
    unions = list(range(len(lfs)))
    # apply union find to group similar lf
    for i in range(len(lfs)-1):
        for j in range(i+1, len(lfs)):
            if find(unions, i) != find(unions, j) and (pre_defined.get(lfs[i]) == lfs[j] or jaccard_overlap(lfs[i].split(), lfs[j].split()) > THRESHOLD):
                union(unions, i, j)
    finals = []
    for i in range(len(unions)):
        root = find(unions, i)
        row = [sf, lfs[root], lfs[i]] + [other_informations[lfs[i]][key] for key in OTHER_KEYS]
        finals.append(row)

    df = pd.DataFrame(finals, columns=['sf', 'lf', 'original_lf'] + OTHER_KEYS)
    return df

def test_union_find():
    a = list(range(10))
    for i in range(10):
        if i in (9, 5):
            union(a, 2, i)
        elif i in (4, 3):
            union(a, i, 8)
    b = []
    for i, u in enumerate(a):
        b.append(find(a, i))
    print(a)
    print(b)


if __name__ == '__main__':
    # print(search('pp', apikey))
    merge_lf()
