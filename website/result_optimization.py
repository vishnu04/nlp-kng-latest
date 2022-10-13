import stanza
from stanza.server import CoreNLPClient
import os
from website.actions import config
os.environ["CORENLP_HOME"] = config.CORENLP_DIR
# print(CORENLP_DIR)
# stanza.download('en') 
from . import actions


def start_stanza_client():
    stanza_nlp = stanza.Pipeline('en')
    client = CoreNLPClient(annotators=['openie','tokenize','ssplit','pos','lemma','ner','parse','coref'], be_quiet = True, classpath = config.CORENLP_DIR)
    client.start()
    return client

def stop_stanza_client(client):
    client.stop()
    return 

def generate_stanza_doc(client, text):
    stanza_text = text
    stanza_document = client.annotate(stanza_text)
    return stanza_document

def generate_stanza_svo(stanza_document):
    ## SVO triplets using Stanza
    for sentence in stanza_document.sentence:
        for triple in sentence.openieTriple:
            svo_df.loc[len(svo_df)] = [triple.subject,triple.relation, triple.object]
    svo_df = svo_df.drop_duplicates()
    return svo_df

# def print_temp_dir():
#     print(f'In result optimization {actions.create_temp_dir.get_temp_dir()}')

def generate_stanza_svo_triplets(text):
    client = CoreNLPClient(annotators=['openie','tokenize','ssplit','pos','lemma','ner','parse','coref'], be_quiet = True, classpath = config.CORENLP_DIR)
    client.start()
    stanza_text = text
    stanza_document = client.annotate(stanza_text)
    for sentence in stanza_document.sentence:
        for triple in sentence.openieTriple:
            svo_df.loc[len(svo_df)] = [triple.subject,triple.relation, triple.object]
    svo_df = svo_df.drop_duplicates()
    client.stop()
    return svo_df

