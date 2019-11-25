import pandas as pd
from config import *

contexts = pd.read_csv(ORIGINAL_DATA_FILE)

def trim_and_replace(ctx):
    split_words = ctx.split(' ')
    tgt_idx = split_words.index(SF_IDENTIFIER)
    return ' '.join(split_words[max(0, tgt_idx - MAX_WINDOW_SIZE): min(len(split_words), tgt_idx + MAX_WINDOW_SIZE + 1)])

# Trim window size and replace TARGETWORDS
contexts['context_trimmed'] = contexts.apply(lambda x: trim_and_replace(x['context']), axis=1)
contexts['tgt_idx'] = contexts['context_trimmed'].map(lambda x: x.split(' ').index(SF_IDENTIFIER))
contexts['context_trimmed'] = contexts.apply(lambda x: x['context_trimmed'].replace(SF_IDENTIFIER, x['sf']), axis=1)
contexts['row_idx'] = contexts.index
contexts.to_csv(ORIGINAL_DATA_FILE.replace('.csv','') + '_trimmed.csv', index=False)
