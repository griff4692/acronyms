#################################################################################
# usage of the script
# usage: python search-terms.py -k APIKEY -v VERSION -s STRING
# see https://documentation.uts.nlm.nih.gov/rest/search/index.html for full docs
# on the /search endpoint
#################################################################################

from __future__ import print_function
from Authentication import *
import glob
import requests
import json
import argparse

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
    results = []
    ticket = AuthClient.getst(tgt)
    query = {'string':string,'ticket':ticket, 'searchType':searchType}
    r = requests.get(uri+content_endpoint,params=query)
    r.encoding = 'utf-8'
    items  = json.loads(r.text)
    return items["result"]["results"]

def _read_acronyms(acronyms_path):
    """
    read acronyms from txt file
    return a gernerator
    """
    with open(acronyms_path, 'r') as f:
        for line in f:
            for i, char in enumerate(line):
                if char.isdigit():
                    yield line[:i-1]
                    break

def download_acronym_pairs(acronyms_path, apikey, path):
    """
    Download acronym pairs with the form of {'acronym1': 'meaning1',
    'meaning2'...} every 10 acronyms.
    """
    d = defaultdict(list)
    count = 1
    for acronym in _read_acronyms(acronyms_path):
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

def merge_jsons(json_pattern):
    d = {}
    for file_path in glob.glob(json_pattern):
        with open(file_path, 'r') as f:
            d.update(json.load(f))
    with open('acronym_expansions_UMLS.json', 'w') as f:
        json.dump(d, f)

if __name__ == "__main__":
    download_acronym_pairs(acronyms_path, apikey, path)
