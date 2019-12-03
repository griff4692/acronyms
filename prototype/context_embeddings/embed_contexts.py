import os

import numpy as np
import pandas as pd
import torch
from transformers import BertForPreTraining, BertTokenizer, BertConfig

from prototype.context_embeddings.bert_config import (BERT_CONFIG_FILE, CHKPT_INDEX_FILE, PYTORCH_MODEL_FILE,
                                                      VOCAB_FILE, WEIGHTS_BASE_PATH)


def summarize_bert_layers(batch_embeds):
    """
    :param batch_embeds: tensor of batch_size x num_layers x embedding_dim
    :return: summarized batch embeddings of batch_size x embedding_dim by taking mean of last 4 layers
    """
    return batch_embeds[:, -4:, :].mean(1).data.numpy()


def extract_word_piece_index(tokenized_input, abbr):
    tokenized_bin = np.array([tkn.startswith('##') for tkn in tokenized_input])
    idx_of_tokens = np.where(tokenized_bin == False)[0]
    idx_and_offset = []
    running_offset = 0
    for idx in range(len(idx_of_tokens) - 1):
        idx_and_offset.append((idx + running_offset, idx_of_tokens[idx + 1]  - idx_of_tokens[idx] - 1))
        running_offset += idx_of_tokens[idx+1] - idx_of_tokens[idx] - 1
    idx_and_offset.append((len(idx_of_tokens) - 1 + running_offset, len(tokenized_bin) - idx_of_tokens[-1] - 1))
    reconstruct_input = [''.join(tokenized_input[i:i+j+1]).replace('##', '') for i,j in idx_and_offset]
    abbr_idx = reconstruct_input.index(abbr.lower())
    abbr_indexes = (idx_and_offset[abbr_idx][0], idx_and_offset[abbr_idx][0] + idx_and_offset[abbr_idx][1] + 1)
    assert ''.join(tokenized_input[abbr_indexes[0]:abbr_indexes[1]]).replace('##','') == abbr.lower()
    return abbr_indexes


def compute_bert_embeds(in_fp, data_purpose, keep_case=False, batch_size=100, use_cached=False):
    out_fn = 'data/{}_embeddings.npy'.format(data_purpose)

    if os.path.exists(out_fn) and use_cached:
        return out_fn

    # HuggingFace model, tokenizer, config
    model_class, tokenizer_class, config_class = BertForPreTraining, BertTokenizer, BertConfig

    # Create config object
    config_obj = config_class.from_json_file(WEIGHTS_BASE_PATH + BERT_CONFIG_FILE)
    config_obj.output_hidden_states = True
    config_obj.output_attentions = True

    # Create model object
    model = model_class(config_obj)
    state_dict = torch.load(WEIGHTS_BASE_PATH + PYTORCH_MODEL_FILE)
    model.load_state_dict(state_dict)

    # Create tokenizer object
    tokenizer = tokenizer_class(WEIGHTS_BASE_PATH + VOCAB_FILE, do_lower_case=not keep_case)

    contexts = pd.read_csv(in_fp, chunksize=batch_size)
    for batch_idx, context_batch in enumerate(contexts):
        print('Computing examples {}-{}'.format(batch_idx * batch_size, (batch_idx + 1) * batch_size))
        context_batch['word_piece_tuple'] = context_batch.apply(
            lambda x: (tokenizer.tokenize(x['context_trimmed']), x['tgt_idx'], x['sf']), axis=1)
        abbr_indexes = [
            extract_word_piece_index(tkn, abb) for
            tkn, abbr_idx, abb in context_batch['word_piece_tuple']
        ]
        input_ids = [
            torch.tensor(tokenizer.encode(ctx, add_special_tokens=True))
            for ctx in context_batch['context_trimmed']
        ]
        max_len = max([x.shape[0] for x in input_ids])
        input_ids = [
            torch.nn.functional.pad(x, pad=(0, max_len-x.shape[0]), value=0)
            for x in input_ids
        ]
        input_ids_batch = torch.stack(input_ids)
        hidden_states, _ = model(input_ids_batch)[-2:]
        hidden_states = torch.stack(hidden_states)
        hidden_states = hidden_states.permute(1,2,0,3)
        abbr_indexes = [np.arange(x[0]+1,x[1]+1) for x in abbr_indexes]
        abbr_hidden_states = [
            h_state[abbr_idx, :, :] for h_state, abbr_idx in
            zip(hidden_states, abbr_indexes)
        ]
        abbr_hidden_states = torch.stack([
            torch.mean(abbr_hstate, dim=0) for abbr_hstate in
            abbr_hidden_states
        ])

        embeddings = summarize_bert_layers(abbr_hidden_states)
        with open(out_fn, 'a+b') as fp:
            np.save(fp, embeddings)

    print('Done generating embeddings.  Now flattening chunked embedding file...')
    with open(out_fn, 'rb') as fd:
        fsz = os.fstat(fd.fileno()).st_size
        final_embeddings = np.load(fd)
        while fd.tell() < fsz:
            final_embeddings = np.vstack((final_embeddings, np.load(fd)))

    with open(out_fn, 'wb') as fp:
        np.save(fp, final_embeddings)
    return out_fn
