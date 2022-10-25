
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoModel, BertTokenizer
import torch

from re import sub
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords

import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity
import numpy as np
from . import query
from . import config

import nltk
from textblob import TextBlob


from huggingface_hub import from_pretrained_keras

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

stop_words = stopwords.words('english')

def create_heatmap(similarity, labels, cmap = "YlGnBu"):
  df = pd.DataFrame(similarity)
  df.columns = labels
  df.index = labels
  fig, ax = plt.subplots(figsize=(5,5))
  sns.heatmap(df, cmap=cmap)

def load_sentence_transformer(pretrained):
    return SentenceTransformer(pretrained)

def sim_sentence_transformer(model, l_sentences):
    similarity = []
    sentences = l_sentences
    embeddings = model.encode(sentences)
    labels = [headline[:30] for headline in sentences]
    for i in range(len(sentences)):
        row = []
        for j in range(len(sentences)):
            row.append(util.pytorch_cos_sim(embeddings[i], embeddings[j]).item())
        similarity.append(row)
    return similarity

def load_tensorflow_hub_model(hub_module_url):
    return hub.load(hub_module_url)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def transformers_auto_model(sentences, pretrained_model= 'stsb-roberta-large'):

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(f'sentence-transformers/{pretrained_model}')
    model = AutoModel.from_pretrained(f'sentence-transformers/{pretrained_model}')

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    return sentence_embeddings

def preprocess(doc):
    # Tokenize, clean up input document string
    doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
    doc = sub(r'<[^<>]+(>|$)', " ", doc)
    doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
    doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
    return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stop_words]

def sim_gensim(question_list, response_list, glove_model = "glove-wiki-gigaword-100", stopwords = ['the', 'and', 'are', 'a']):
    response_corpus = [preprocess(response) for response in response_list]
    question_corpus = [preprocess(question) for question in question_list]

    # Load the model: this is a big file, can take a while to download and open
    glove = api.load(f"{glove_model}")

    ## similarity
    similarity_matrix = WordEmbeddingSimilarityIndex(glove)

    # Build the term dictionary, TF-idf model
    dictionary = Dictionary(response_corpus+question_corpus)
    tfidf = TfidfModel(dictionary=dictionary)

    # Create the term similarity matrix.  
    similarity_matrix = SparseTermSimilarityMatrix(similarity_matrix, dictionary, tfidf)
    
    print(similarity_matrix)
    print(question_corpus)
    
    # Compute Soft Cosine Measure between the query and the documents.
    # From: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb
    for qidx, ques_corpus in enumerate(question_corpus):
        query_tf = tfidf[dictionary.doc2bow(ques_corpus)]

        index = SoftCosineSimilarity(
                    tfidf[[dictionary.doc2bow(document) for document in response_corpus]],
                    similarity_matrix)

        doc_similarity_scores = index[query_tf]

        # Output the sorted similarity scores and documents
        sorted_indexes = np.argsort(doc_similarity_scores)[::-1]
        print(question_list[qidx],ques_corpus)
        for idx in sorted_indexes:
            print(f'{idx} \t {doc_similarity_scores[idx]:0.3f} \t {response_list[idx]}')


class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.

    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(
        self,
        sentence_pairs,
        labels,
        batch_size=200,
        shuffle=True,
        include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=200,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)

def load_keras_pretrained(pretrained_model='bert-semantic-similarity'):
    return from_pretrained_keras(f"keras-io/{pretrained_model}")

def check_similarity(model, sentence1, sentence2):
    sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
    # Labels in our dataset.
    labels = ["contradiction", "entailment", "neutral"]
    test_data = BertSemanticDataGenerator(
        sentence_pairs, 
        labels=None, 
        batch_size=1, 
        shuffle=False, 
        include_targets=False,
    )
    proba = model.predict(test_data[0])[0]
    idx = np.argmax(proba)
    proba = f"{proba[idx]: .2f}%"
    pred = labels[idx]
    return pred, proba

def semantic_search(model, question_list, response_list, response_df):
    # s_quest_embeddings = model.encode(question_list, convert_to_tensor = True)
    # s_response_embeddings = model.encode(response_list, convert_to_tensor = True)
    # s_response_embeddings = torch.TensorType
    s_quest_embeddings = question_list
    s_response_embeddings = response_list
    # print(f'semantic_search s_quest_embeddings : {s_quest_embeddings.size()}')
    # print(f'semantic_search s_response_embeddings : {s_response_embeddings.size()}')
    # print(s_quest_embeddings)
    # print('---------------')
    # print(s_response_embeddings)
    # print(f'response list element type : {type(response_list[0])}')
    # for response in response_list:
    #    s_response_embeddings = torch.stack(s_response_embeddings,torch.as_tensor(response))
    # s_response_embeddings = torch.stack(response_list)
    # print(f'type of s_quest_embeddings : {type(s_quest_embeddings)}')
    # for response in response_list:  
    #     s_response_embeddings.stack(torch.as_tensor(response_list))
    hits = util.semantic_search(s_quest_embeddings, s_response_embeddings, top_k= config.QA_BASE_MPNET_MODEL_TOPN)
    semantic_search_output = set()
    for qidx, hit in enumerate(hits):
        for k in hit:
            # search_result = "\t{:.3f}\t{}".format(k['score'], response_list[k['corpus_id']])
            search_result = "\t{:.3f}\t{}".format(float(k['score'])*10, response_df['sentence'][k['corpus_id']]) 
            semantic_search_output.add(search_result)
    return semantic_search_output 

def dot_score(model, question_list, response_list):
    quest_embeddings = model.encode(question_list, convert_to_tensor = True)
    response_embeddings = model.encode(response_list, show_progress_bar = False, convert_to_tensor = True)
    dot_sims = util.dot_score(quest_embeddings, response_embeddings)
    dot_score_output = set()
    for qidx, sims in enumerate(dot_sims):
        print(question_list[qidx])
        
        for sidx,k in enumerate(sims):
            dot_score_result = "\t{:.3f}\t{}".format(k.tolist(), response_list[sidx])
            print(dot_score_result) 
            dot_score_output.add(dot_score_result)
    return dot_score_output

def load_qa_mpnet_base_model():
    return load_sentence_transformer(config.QA_BASE_MPNET_MODEL)

# def gen_qa_mpnet_embeddings(model, sentence):
#     if len(sentence) > 0:
#         return model.encode(sentence, show_progress_bar = False, convert_to_tensor = True)
#     return 'Unable to convert to embedding'

def gen_qa_mpnet_embeddings(model, sentence_list):
    if len(sentence_list) > 0:
        # return model.encode(sentence_list, show_progress_bar = False, convert_to_tensor = True)
        if len(sentence_list) == 1:
            return model.encode(sentence_list, show_progress_bar = False, convert_to_tensor = True)
        else:
            output = []
            for sent in sentence_list:
                print(f'gen_qa_mpnet_embeddings : {sent}')
                output_tensor = model.encode(sent, show_progress_bar = False, convert_to_tensor = True)
                output.append(output_tensor)
            result = torch.Tensor(torch.stack(output, dim=1))
            return result
    return 'Unable to convert to embedding'


def call_qa_mpnet(model, quest, svo_df, question_embedding, sentence_embeddings, question_svo_df ):
    print(f'Calling qa base mpnet model ---->')
    answer_facts = []
    answer_index = []
    question_lemma = []
    question_pos = []
    ## implement models
    # qa_base_mpnet_model = load_sentence_transformer('multi-qa-mpnet-base-cos-v1')
    qa_base_mpnet_model = model
    # semantic_search_output = list(semantic_search(qa_base_mpnet_model, [quest], list(svo_df['sentence'])))
    # semantic_search_output = list(semantic_search(qa_base_mpnet_model, [quest], list(svo_df.qa_base_mpnet_embeddings)))
    semantic_search_output = list(semantic_search(qa_base_mpnet_model, question_embedding, sentence_embeddings, svo_df))
    # qa_base_mpnet_model = load_sentence_transformer('multi-qa-mpnet-base-cos-v1')
    # dot_score_output = list(dot_score(qa_base_mpnet_model, [quest], list(svo_df['sentence'])))
    
    answers_df = pd.DataFrame(columns=['confidence','output_str', 'output_sent'])
    answers_df = answers_df.astype({'confidence':'float','output_str': 'str','output_sent':'str'})
    filter_answers_df = pd.DataFrame(columns=['confidence','output_str', 'output_sent'])
    filter_answers_df = filter_answers_df.astype({'confidence':'float','output_str': 'str','output_sent':'str'})
    n = 1
    # for output in dot_score_output:
    # print(f'semantic_search_output: {len(semantic_search_output)}')
    for output in semantic_search_output:
        confidence, output_str = float(output.split('\t')[1].strip()),output.split('\t')[2]
        answers_df.loc[len(answers_df)] = [confidence, output_str, output]
    # print(f'len(answers_df) --> {len(answers_df)}')
    new_filtered_answers = pd.DataFrame(columns=['confidence','output_str', 'output_sent']) 
    print(f'question_svo_df --> {question_svo_df}')
    if len(question_svo_df) == 0:
        question_nouns = [w for (w, pos) in TextBlob(str(quest)).pos_tags if pos[0] == 'N']
    else:
        question_nouns = [w for (w, pos) in TextBlob(str(question_svo_df.head(2)['sentence'].values[0])).pos_tags if pos[0] == 'N']
    print(question_svo_df.head(2)['sentence'].values)
    print(f'question_nouns --> {question_nouns}')
    for index, row in answers_df.iterrows():
        if any(word in row['output_str'] for word in question_nouns):
            new_filtered_answers.loc[len(new_filtered_answers)] = row
    print(f'len(new_filtered_answers) --> {len(new_filtered_answers)}')
    if len(new_filtered_answers) > 0:
        answers_df = new_filtered_answers.copy()
        del new_filtered_answers
    if len(answers_df) > 0 :
        answers_df = answers_df.sort_values(by=['confidence'], ascending=False).reset_index()
        filter_answers_df = answers_df[answers_df.confidence  >= config.QA_BASE_MPNET_MODEL_CONF]
    print(f'len(filter_answers_df) --> {len(filter_answers_df)}')
    if len(filter_answers_df) == 0:
        print(answers_df)
        print(f'if = 0 len(filter_answers_df) --> {len(filter_answers_df)}')
        print(f'len(answers_df) --> {len(answers_df)}')
        for index, row in answers_df.iterrows():
            if row.confidence >= 0.4:
                answer_facts.append(f'[score:{row.confidence}] - {row.output_str}')
                answer_index.append(query.get_detailed_answer_index(row.output_str, svo_df))
                n += 1
        if len(answer_facts) != 0:
            return answer_facts, question_lemma, question_pos, answer_index 
        else:
            for index, row in answers_df.iterrows():
                answer_facts.append(f'[score:{row.confidence}] - {row.output_str}')
                answer_index.append(query.get_detailed_answer_index(row.output_str, svo_df))
                n += 1
            return answer_facts, question_lemma, question_pos, answer_index 
    elif len(filter_answers_df) > 3:
        print(f'if > 3len(filter_answers_df) --> {len(filter_answers_df)}')
        for index, row in filter_answers_df.iterrows():
            answer_facts.append(f'[score:{row.confidence}] - {row.output_str}')
            answer_index.append(query.get_detailed_answer_index(row.output_str, svo_df))
            n += 1
        return answer_facts, question_lemma, question_pos, answer_index 
    else:
        print(f'else len(filter_answers_df) --> {len(filter_answers_df)}')
        for index, row in filter_answers_df.iterrows():
            answer_facts.append(f'[score:{row.confidence}] - {row.output_str}')
            answer_index.append(query.get_detailed_answer_index(row.output_str, svo_df))
            n += 1
    return answer_facts, question_lemma, question_pos, answer_index 