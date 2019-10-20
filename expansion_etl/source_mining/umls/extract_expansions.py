#################################################################################
# usage of the script
# usage: python search-terms.py -k APIKEY -v VERSION -s STRING
# see https://documentation.uts.nlm.nih.gov/rest/search/index.html for full docs
# on the /search endpoint
#################################################################################

from __future__ import print_function

from collections import defaultdict
import glob
import json
import os

from expansion_etl.source_mining.umls.Authentication import *


def search(string, apikey, searchType='exact'):
    """
    search string at UMLS, default is Exact search
    """
    version = 'current'
    uri = "https://uts-ws.nlm.nih.gov"
    content_endpoint = "/rest/search/" + version
    # get a ticket granting ticket for the session
    AuthClient = Authentication(apikey)
    tgt = AuthClient.gettgt()
    ticket = AuthClient.getst(tgt)
    query = {'string':string,'ticket':ticket, 'searchType':searchType}
    r = requests.get(uri+content_endpoint,params=query)
    r.encoding = 'utf-8'
    items  = json.loads(r.text)
    return items["result"]["results"]


def download_acronym_pairs(acronyms, apikey, path):
    """
    Download acronym pairs with the form of {'acronym1': 'meaning1',
    'meaning2'...} every 10 acronyms.
    """
    d = defaultdict(list)
    count = 1
    for acronym in acronyms:
        print(count, ': ', acronym)
        for result in search(acronym, apikey):
            d[acronym].append(result['name'])
        if count % 10 == 0:
            with open('json/' + str(count) + path, 'w') as f:
                json.dump(d, f)
            d = defaultdict(list)
        count += 1
    with open('json/' + str(count) + path, 'w') as f:
        json.dump(d, f)


def merge_jsons(json_pattern, out_fn):
    d = {}
    for file_path in glob.glob(json_pattern):
        with open(file_path, 'r') as f:
            d.update(json.load(f))
    with open(out_fn, 'w') as f:
        json.dump(d, f)


def extract_expansions(acronyms, use_cached=True):
    print('Extracting expansions from UMLS...')
    # TODO use_cached = True to avoid issues with this script
    use_cached = True
    out_fn = './data/derived/umls_acronym_expansions.json'
    if use_cached and os.path.exists(out_fn):
        return out_fn
    return download_acronym_pairs(acronyms, os.environ['UMLS_API'], None)
