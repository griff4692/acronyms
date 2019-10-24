import os
import re

from nltk.corpus import stopwords
import pandas as pd
import spacy

STOPWORDS = set(stopwords.words('english'))


def choose_longest(string_arr):
    longest_len = -1
    longest_str = ''
    for str in string_arr:
        if len(str) >= longest_len:
            longest_len = len(str)
            longest_str = str
    # Strip extra whitespace
    return re.sub(r'\s+', ' ', longest_str).strip()


def convert_to_base(str, nlp):
    return ' '.join([token.lemma_ for token in nlp(str) if token.lemma_ not in STOPWORDS])


def merge_bars(string_arr):
    """
    Merge data of the sort ['a|b|c', 'b|c|d'] --> 'a|b|c|d'
    """
    merged = set()
    for str in string_arr:
        for x in str.split('|'):
            merged.add(x)
    return '|'.join(list(merged))


def standardize_expansions(in_fp, use_cached=False):
    """
    Lemmatize and remove stopwords from expansions (LFs)
    """
    out_fn = './data/derived/standardized_acronym_expansions.csv'
    if use_cached and os.path.exists(out_fn):
        return out_fn

    df = pd.read_csv(in_fp)
    nlp = spacy.load("en_core_sci_sm", disable=['parser', 'ner'])

    df.dropna(inplace=True)  # This should be done in previous step but not might not be working properly
    df['lf_base'] = df['lf'].apply(lambda lf: convert_to_base(lf, nlp))
    df['lf_base_stripped'] = df['lf_base'].apply(lambda lf_base: re.sub(r'\W+', '', lf_base))
    df.drop(df[df['sf'].str.lower() == df['lf_base']].index, inplace=True)
    pre_empty = df.shape[0]
    df.drop(df[df['lf_base'].str.len() == 0].index, inplace=True)
    print('{} out of {} LFS contained only stopwords'.format(pre_empty - df.shape[0], pre_empty))

    dfg = df.groupby(['sf', 'lf_base_stripped']).agg(
        {
            'lf': lambda x: merge_bars(list(x)),
            # if we have an elided phrase ('lungcancer') and 'lung cancer' assume the longer is the correct one
            'lf_base': lambda x: choose_longest(list(x)),
            'source': lambda x: merge_bars(list(x)),
        }).reset_index()[['sf', 'lf', 'lf_base', 'source']]

    print('{} out of {} SF-LF pairs share the same lemmatized LF'.format(df.shape[0] - dfg.shape[0], dfg.shape[0]))
    dfg.to_csv(out_fn, index=False)
    return out_fn
