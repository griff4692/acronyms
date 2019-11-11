import json

import argparse

from add_cui import add_cui
from filter_inclusion import filter_inclusion
from standardize_expansions import standardize_expansions
from extract_acronyms_list import extract_acronyms_list
from merge_sources import merge_sources
from source_mining.pubmed.extract_expansions import extract_expansions as extract_pubmed_expansions
from source_mining.umls.extract_expansions import extract_expansions as extract_umls_expansions
from source_mining.wikipedia.extract_expansions import extract_expansions as extract_wikipedia_expansions
from utils import render_args

SOURCE_EXTRACTION_MAP = {
    'pubmed': extract_pubmed_expansions,
    'umls': extract_umls_expansions,
    'wikipedia': extract_wikipedia_expansions,
}


def etl(args, sources, fp):
    print('Loading acronyms...')
    acronym_fp = extract_acronyms_list(fp, use_cached=args.use_cached)
    acronym_list = json.load(open(acronym_fp, 'r'))
    print('Extracting acronym expansions from sources...')
    source_fns = map(lambda source: SOURCE_EXTRACTION_MAP[source](acronym_list, use_cached=args.use_cached), sources)
    merged_fp = merge_sources(zip(sources, list(source_fns)), acronym_list, use_cached=args.use_cached)
    print('Standardizing expansions (LFs) by lemmatizing and removing stopwords...')
    standard_fp = standardize_expansions(merged_fp, use_cached=args.use_cached)
    print('Adding UMLS Concepts to standardized LFs with MetaMap...')
    cui_fp = add_cui(standard_fp, use_cached=args.use_cached)
    print("Filtering for is_accurate and is_clinical_relevant")
    inclusion_fp = filter_inclusion(cui_fp, use_cached=args.use_cached)
    return inclusion_fp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mine Set of Expansion Terms for Given List of Acronym.')
    parser.add_argument('-use_cached', default=False, action='store_true')
    parser.add_argument('--sources', default='umls|pubmed|wikipedia')

    args = parser.parse_args()
    render_args(args)

    initial_fp = 'data/original/initial_acronyms.txt'
    sources = args.sources.split('|')
    final_fn = etl(args, sources, initial_fp)
    print('Final dataset is located --> {}'.format(final_fn))
