import pickle
import torch

PROCESS_EMBED_LAYERS = {
    'mean': torch.mean,
    'first_layer': lambda x: x[0],
    'last_layer': lambda x: x[-1],
    'last_layers': lambda x: x[-4:].mean(0).data.numpy()
}
SELECTION_METHOD = 'last_layers'
EMBEDDING_PATH = '/home/ga2530/acronyms/prototype/context_embeddings/data/embeddings.dict'
def get_idx_to_embedding():
    data = []
    total_len = 0
    with open(EMBEDDING_PATH, 'rb') as f:
        while True:
            try:
                data.append(
                    {x:PROCESS_EMBED_LAYERS[SELECTION_METHOD](y) for x, y in torch.load(f).items()}
                )
                total_len += 100
            except EOFError:
                break
    idx_to_embedding = {}
    for d in data:
        idx_to_embedding.update(d)
    return idx_to_embedding

d = get_idx_to_embedding()
torch.save(d, '/home/ga2530/acronyms/prototype/context_embeddings/data/embeddings_parsed.dict')
