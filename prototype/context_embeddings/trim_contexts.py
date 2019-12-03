import os

import pandas as pd


def trim_and_replace(sf_identifier, ctx, window_size):
    split_words = ctx.split(' ')
    tgt_idx = split_words.index(sf_identifier)
    truncated = split_words[max(0, tgt_idx - window_size): min(len(split_words), tgt_idx + window_size + 1)]
    return ' '.join(truncated)


def trim(sf_identifier, window_size, in_fp, use_cached=False):
    out_fp = in_fp.replace('.csv','') + '_trimmed.csv'

    if use_cached and os.path.exists(out_fp):
        return out_fp

    contexts = pd.read_csv(in_fp)
    # Trim window size and replace TARGETWORDS
    contexts['context_trimmed'] = contexts.apply(
        lambda x: trim_and_replace(sf_identifier, x['context'], window_size), axis=1)
    contexts['tgt_idx'] = contexts['context_trimmed'].map(lambda x: x.split(' ').index(sf_identifier))
    contexts['context_trimmed'] = contexts.apply(lambda x: x['context_trimmed'].replace(sf_identifier, x['sf']), axis=1)
    contexts.to_csv(out_fp, index=False)

    return out_fp
