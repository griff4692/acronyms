import json
import os

import argparse
import pandas as pd
import numpy as np

from add_cui import add_cui
from extract_acronyms_list import extract_acronyms_list
from merge_sources import merge_sources
from utils import render_args


def etl(args, sources, fp):
    acronym_fp = extract_acronyms_list(fp, use_cached=args.use_cached)
    acronym_list = json.load(open(acronym_fp, 'r'))

    merged_fp = merge_sources(sources, acronym_list, use_cached=args.use_cached)

    cui_fp = add_cui(merged_fp, use_cached=args.use_cached)

    print(cui_fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mine Set of Expansion Terms for Given List of Acronym.')
    parser.add_argument('-use_cached', default=False, action='store_true')
    parser.add_argument('--sources', default='umls,pubmed')

    args = parser.parse_args()
    render_args(args)

    initial_fp = 'data/original/initial_acronyms.txt'
    sources = args.sources.split(',')
    etl(args, sources, initial_fp)
