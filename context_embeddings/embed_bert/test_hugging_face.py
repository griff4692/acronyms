import torch
from transformers import *

model_class, tokenizer_class, pretrained_weights = (BertModel, BertTokenizer, 'bert-base-uncased')

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

input_ids = torch.tensor([
    tokenizer.encode("Here is some text to encode Here here here", add_special_tokens=True)
    ]
)

print(input_ids)


