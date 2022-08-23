## Installation of neuralcoref: https://github.com/huggingface/neuralcoref

from spacy import displacy
# import spacy
import textacy
import pandas as pd
# import neuralcoref
from textacy import keyterms
from . import config
from spacy.tokens import DocBin

''' Extracting Subject Verb Object triplets from text'''
def extract_SVO(text):
    tuples_to_list = []
    tuples = textacy.extract.subject_verb_object_triples(text)
    if tuples:
        tuples_list = list(tuples)
        tuples_to_list.append(tuples_list)
    return tuples_to_list



@config.timer
def svo(text_doc, nlp):
# def svo(docs_byte_data, nlp):

    # '''Converting text into Document'''
    # text = text.lower()
    # print('Entered svo')
    # text_doc = nlp(text)
    # print('Entered svo')
    # text_doc = text_doc._.coref_resolved
    # print('Entered svo')
    # text_doc = nlp(text_doc)
    # doc_bin = DocBin().from_bytes(docs_byte_data)
    # text_doc = list(doc_bin.get_docs(nlp.vocab))
    # text_doc = text 
    # print(text_doc)

    print('Entered svo')
    '''For displaying in webpage'''
    # displacy.render(text_doc,style="ent")

    '''Extracting SVO triplets into dataframe'''
    svo_df = pd.DataFrame(columns = ['subject','verb','object'])
    sentence_tuples = extract_SVO(text_doc)
    for svoTriple in sentence_tuples:
        for s,v,o in svoTriple:
            subj = ' '.join([str(n) for n in s])
            verb = ' '.join([str(n) for n in v])
            obj = ' '.join([str(n) for n in o])
            svo_df.loc[len(svo_df)] = [subj,verb,obj]
    new_svo_df = pd.DataFrame(columns = ['subject','verb','object'])
    print(svo_df.head(5))

    for index, svo_tuple in svo_df.iterrows():
        if set(["PROPN",'NOUN']) & set([tok.pos_ for tok in nlp(svo_tuple.subject)]):
            if(set(["PROPN",'NOUN']) & set([tok.pos_ for tok in nlp(svo_tuple.object)])):
                new_svo_df.loc[len(new_svo_df)] = [svo_tuple.subject,svo_tuple.verb, svo_tuple.object]
    new_svo_df = new_svo_df.drop_duplicates()
    svo_df = new_svo_df.copy()
    return svo_df,text_doc




# if __name__ == "__main__":
    
#     '''defining the pipeline'''
#     # nlp = spacy.load('en_core_web_lg',n_threads=LEMMATIZER_N_THREADS,  batch_size=LEMMATIZER_BATCH_SIZE)
#     nlp = spacy.load('en_core_web_lg')
#     neuralcoref.add_to_pipe(nlp)
    
#     '''Reading the text'''
#     text = ''
#     with open(f'{DATA_TEXT_FILE_NAME}','r') as file:
#         lines = file.readlines()
#         for line in lines:
#             text = text + ' ' + line
#         file.close()
#     text = text.replace('\n', ' ')
        
#     '''Converting text into Document'''
#     text = text.lower()
#     text_doc = nlp(text)
#     text_doc = text_doc._.coref_resolved
#     text_doc = nlp(text_doc)
    
#     '''For displaying in webpage'''
#     # displacy.render(text_doc,style="ent")

#     '''Extracting SVO triplets into dataframe'''
#     svo_df = pd.DataFrame(columns = ['subject','verb','object'])
#     sentence_tuples = extract_SVO(text_doc)
#     for svoTriple in sentence_tuples:
#         for s,v,o in svoTriple:
#             subj = ' '.join([str(n) for n in s])
#             verb = ' '.join([str(n) for n in v])
#             obj = ' '.join([str(n) for n in o])
#             svo_df.loc[len(svo_df)] = [subj,verb,obj]
#     new_svo_df = pd.DataFrame(columns = ['subject','verb','object'])
#     for index, svo_tuple in svo_df.iterrows():
#         if set(["PROPN",'NOUN']) & set([tok.pos_ for tok in nlp(svo_tuple.subject)]):
#             if(set(["PROPN",'NOUN']) & set([tok.pos_ for tok in nlp(svo_tuple.object)])):
#                 new_svo_df.loc[len(new_svo_df)] = [svo_tuple.subject,svo_tuple.verb, svo_tuple.object]
#     new_svo_df = new_svo_df.drop_duplicates()
#     svo_df = new_svo_df.copy()
#     del new_svo_df