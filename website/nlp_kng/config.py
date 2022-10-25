# from transformers import BertTokenizer, BertModel, BertForQuestionAnswering, BertConfig
# import torch
# import tqdm as notebook_tqdm

def timer(fn):
    from time import perf_counter
    
    def inner(*args, **kwargs):
        start_time = perf_counter()
        to_execute = fn(*args, **kwargs)
        end_time = perf_counter()
        execution_time = end_time - start_time
        print('{0} took {1:.8f}s to execute'.format(fn.__name__, execution_time))
        return to_execute
    return inner


## Variables used for WebScraping and data cleaning
WEB_URL = 'https://www.investopedia.com/ask/answers/111314/what-causes-inflation-and-does-anyone-gain-it.asp'
#WEB_URL = 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3271273/'
DATA_LOC = './'
RAW_TEXT_FILE_NAME = 'raw_data.csv'
DATA_TEXT_FILE_NAME = 'cleaned_data.csv'
COQA_DATA_FILE_NAME = 'coqa_data.csv'

## Variables used for creating Entity Extraction using BERT model
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10

TMP_PATH = '../../tmp/'
TMP_DAYS = 1

# BASE_MODEL = 'bert-large-uncased-whole-word-masking-finetuned-squad'
BASE_MODEL = 'en_core_web_lg'
MODEL_PATH = 'model'
# TOKENIZER = BertTokenizer.from_pretrained(BASE_MODEL, 
#                                           do_lower_case = True)


## 

ROBERTA_LARGE_MODEL = 'stsb-roberta-large'
TENSORFLOW_HUB_MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
GLOVE_MODEL = "glove-wiki-gigaword-100"
KERAS_SEMANTIC_SIMILARITY = 'bert-semantic-similarity'
ALL_MPNET_BASE_MODEL = 'all-mpnet-base-v2'
QA_BASE_MPNET_MODEL = "multi-qa-mpnet-base-cos-v1"
# CORENLP_DIR = '../../stanza_nlp'
CORENLP_DIR = r"../../stanford-corenlp-4.5.1"

## 

STANZA_PATH = r"../../stanford-corenlp-4.5.1/*"
STANZA_MODEL_JAR = r"stanford-corenlp-4.5.1-models.jar"
STANZA_JAR = r"stanford-corenlp-4.5.1.jar"

##
# SUB_VERB_OBJ_DF_COLS = ['subject','verb','object','sentence','reason_flag','qa_base_mpnet_embeddings']
SUB_VERB_OBJ_DF_COLS = ['subject','verb','object','sentence','reason_flag']

## 
QA_BASE_MPNET_MODEL_CONF = 0.35
QA_BASE_MPNET_MODEL_TOPN = 40
