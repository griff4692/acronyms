from collections import defaultdict
import concurrent.futures
import os

import pandas as pd

from distance import jaccard_overlap
from source_mining.umls.extract_expansions import search


apikey = 'b3a1775f-e041-43c8-b529-03c843ff8934'
PATH_LF = 'data/derived/prototype_acronym_expansions_final.csv'
SAVE_DIR = 'data/derived/merged_lf/'
KEY_SF = 'sf'
KEY_LF = 'lf'
KEY_LF_BASE = 'lf_base'
KEY_GROUP = 'semgroups'
KEY_CUI = 'cui'
THRESHOLD = 0.66
NONCLINICAL_GROUPS = set(['Genes & Molecular Sequences'])
OTHER_KEYS = ['cui', 'source', 'semtypes', 'semgroups']
FINAL_KEYS = ['sf', 'lf'] + OTHER_KEYS


def merge_concurrently(read_path, selected_sf=[], save_dir=SAVE_DIR, verbose=True):
    """
    Input:
        read_path: the path to standardized_acronym_expansions_with_cui.csv
        selected_sf: the short forms that you want to process, if empty, will iterate through all short forms, dtype = list
        save_dir: the directory which you want to save files
        verbose: print the completed percentage for each short form

    Output:
        return the save_dir where all files were saved
    """
    # read PATH_LF
    df = pd.read_csv(read_path)
    # groupby each short form
    gk = df.groupby(KEY_SF)

    df_w_same_sf = []
    sfs = []

    for name, group in gk:
        # only process those short forms that we targeted
        if selected_sf and name in selected_sf:
            df_w_same_sf.append(group)
            sfs.append(name)
        # run all short forms
        elif not selected_sf:
            df_w_same_sf.append(group)
            sfs.append(name)

    print('start merging...')
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(merge_lf, df_w_same_sf, sfs, [save_dir for _ in range(len(sfs))],
                     [verbose for _ in range(len(sfs))])
    return save_dir


def merge_lf(df_w_same_sf, sf, save_dir, verbose):
    """
    Input:
        df_w_same_sf: the dataframe with the same short forms, dtype = DataFrame
        sf: the short form itself, dtype = string
    Output:
        save each dataframe as {sf}.csv in save_path
        return nothing
    """
    # storing each long form in a set
    set_lfs = set()

    # pre_defined store the rows with multiple long forms seperated by '|',
    # e.g. lf1|lf2|lf3
    # where key: value = {lf1: lf2, lf1: lf3, lf1: lf1}
    pre_defined = {}

    # store all the other information
    other_informations = defaultdict(dict)
    percentage = 0
    for idx, row in df_w_same_sf.iterrows():
        if verbose and ((idx - df_w_same_sf.index[0]) / len(df_w_same_sf.index) * 100) > percentage:
            percentage += 10
            print('{0}: {1: .2f}%'.format(sf, (idx - df_w_same_sf.index[0]) / len(df_w_same_sf.index) * 100))
        # store all the long forms in current row
        lfs = set()
        lfs = lfs.union(set(row[KEY_LF_BASE].split('|')))
        lfs = lfs.union(set(row[KEY_LF].split('|')))
        # run through all the long forms in 'lf' and store in pre_defined
        lfs = list(lfs)
        for lf in lfs:
            pre_defined[lfs[0]] = lf

        # iterate through all the long forms in current row
        for lf in lfs:
            # only deal with the long forms that has a semgroup, is clinical
            # relevant, and has an exact match in UMLS
            if not pd.isna(row[KEY_GROUP]) and is_clinical(row[KEY_GROUP]) and search(lf, apikey)[0]['ui'] != 'NONE':
                for key in OTHER_KEYS:
                    other_informations[lf][key] = row[key]
                set_lfs.add(lf)
    print('finish finding all the long forms of', sf)
    merge_single_sf(sf, set_lfs, pre_defined, other_informations).to_csv(os.path.join(save_dir, sf + '.csv'), index=False)


def merge_single_sf(sf, set_lfs, pre_defined, other_informations):
    """
    Input:
        sf: the short form itself, dtype = string
        set_lfs: set of relevant long forms, dtype = set
        pre_defined: store the rows with multiple long forms seperated by '|', e.g. {lf1: lf1, lf1: lf2, lf1: lf3}, dtype = dict
        other_informations: store other information, dtype = dict
    Output:
        return dataframe with columns = ['sf', 'lf_key', 'lf_origin',  OTHER_KEYS],
        where lf_key is the key of a group of long forms after merging and
        lf_origin is the original long form.
        For example, lf1, lf2, and lf3 are the same group after merging.
        The dataframe will look like:
            'sf' ,       'lf'
             sf  ,   lf1|lf2|lf3
    """
    lfs = [key for key in set_lfs]
    unions = list(range(len(lfs)))
    # apply union find to group similar lf
    for i in range(len(lfs)-1):
        for j in range(i+1, len(lfs)):
            # union those long forms that are
            # 1. already in pre_defined, or
            # 2. with high jaccard index, or
            # 3. have the same cui
            if find(unions, i) != find(unions, j) and (pre_defined.get(lfs[i]) == lfs[j] or
                                                       jaccard_overlap(lfs[i].split(), lfs[j].split()) > THRESHOLD or
                                                       have_same_cui(lfs[i], lfs[j], other_informations)):
                union(unions, i, j)

    finals = defaultdict(lambda: defaultdict(set))
    for i in range(len(unions)):
        root = find(unions, i)
        finals[lfs[root]][KEY_LF].add(lfs[i])
        for key in OTHER_KEYS:
            for element in other_informations[lfs[i]][key].split('|'):
                finals[lfs[root]][key].add(element)

    rows = []
    for lf_root in finals:
        row = [sf]
        for key in FINAL_KEYS[1:]:
            row.append('|'.join(finals[lf_root][key]))
        rows.append(row)
    df = pd.DataFrame(rows, columns=FINAL_KEYS)
    return df


def have_same_cui(lf1, lf2, other_informations):
    """check if lf1 and lf2 have the same cui"""
    return any(cui1 in other_informations[lf2][KEY_CUI].split('|') for cui1 in other_informations[lf1][KEY_CUI].split('|'))


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


if __name__ == '__main__':
    merge_concurrently(PATH_LF)
