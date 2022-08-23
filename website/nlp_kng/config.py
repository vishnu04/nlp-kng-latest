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
