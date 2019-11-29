# Pre-trained weight file paths
WEIGHTS_BASE_PATH = '/home/ga2530/embed_bert/pre_trained_weights/ncbi_bert_pubmed_uncased_l12_h768_a12/'
vocab_file = 'vocab.txt'
bert_config_file = 'bert_config.json'
# Need to specify either a chkpt_index_file or a pytorch_model_file
chkpt_index_file ='bert_model.ckpt.index'
pytorch_model_file = 'pytorch_model.bin'
do_lower_case_input = True
NUM_LAYERS = 13
EMBED_SIZE = 768

# Original data file
ORIGINAL_DATA_FILE = '/home/ga2530/acronyms/prototype/context_extraction/data/downsampled_prototype_contexts.csv'
MAX_WINDOW_SIZE = 10
SF_IDENTIFIER = 'TARGETWORD'

# Data files (expects pandas dataframe)
DATA_FILE = '/home/ga2530/acronyms/prototype/context_extraction/data/downsampled_prototype_contexts_trimmed.csv'
BATCH_SIZE = 100
