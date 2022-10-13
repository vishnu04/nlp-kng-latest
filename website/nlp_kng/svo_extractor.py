## Installation of neuralcoref: https://github.com/huggingface/neuralcoref

from spacy import displacy
import textacy
import pandas as pd
from . import config, models
from spacy.tokens import DocBin
from . import stanza_svo_extractor

''' Extracting Subject Verb Object triplets from text'''
def extract_SVO(text):
    tuples_to_list = []
    tuples = textacy.extract.subject_verb_object_triples(text)
    if tuples:
        tuples_list = list(tuples)
        tuples_to_list.append(tuples_list)
    return tuples_to_list



@config.timer
def svo(text_doc, text, nlp, model):
    print('Entered svo')
    '''For displaying in webpage'''
    # displacy.render(text_doc,style="ent")

    '''Extracting SVO triplets into dataframe'''
    svo_df = pd.DataFrame(columns = config.SUB_VERB_OBJ_DF_COLS)
    sentences = [sent for sent in text_doc.sents]
    qa_mpnet_base_model = model
    # for sentence in sentences:
    #     sentence_tuples = []
    #     # sentence_tuples = extract_SVO(sentence)
    #     for svoTriple in sentence_tuples:
    #         for s,v,o in svoTriple:
    #             subj = ' '.join([str(n) for n in s])
    #             verb = ' '.join([str(n) for n in v])
    #             obj = ' '.join([str(n) for n in o])
    #             check_flag, reason_flag = stanza_svo_extractor.change_subject_object(str(sentence), subj, verb, obj)
    #             input_sent_embedding = models.gen_qa_mpnet_embeddings(qa_mpnet_base_model, str(sentence))
    #             svo_df.loc[len(svo_df)] = [subj,verb,obj, sentence, reason_flag,input_sent_embedding]
    # new_svo_df = pd.DataFrame(columns = config.SUB_VERB_OBJ_DF_COLS)

    # for index, svo_tuple in svo_df.iterrows():
    #     if set(["PROPN",'NOUN']) & set([tok.pos_ for tok in nlp(svo_tuple.subject)]):
    #         if(set(["PROPN",'NOUN']) & set([tok.pos_ for tok in nlp(svo_tuple.object)])):
    #             new_svo_df.loc[len(new_svo_df)] = [svo_tuple.subject,svo_tuple.verb, svo_tuple.object, svo_tuple.sentence, svo_tuple.reason_flag, svo_tuple.qa_base_mpnet_embeddings]
    # new_svo_df = new_svo_df.drop_duplicates()
    # svo_df = new_svo_df.copy()
    # del new_svo_df

    if len(svo_df) > 1:
        svo_df,reason_present, sentence_embeddings = stanza_svo_extractor.triplet_extraction(text_doc,qa_mpnet_base_model,text, output = ['result'])
        return svo_df,text_doc, sentence_embeddings
    else:
        print(f'Extracting using stanza ---->')
        svo_df,reason_present, sentence_embeddings = stanza_svo_extractor.triplet_extraction(text_doc,qa_mpnet_base_model,text, output = ['result'])
        new_svo_df = pd.DataFrame(columns = config.SUB_VERB_OBJ_DF_COLS)
        new_svo_df = svo_df.drop_duplicates()
        svo_df = new_svo_df.copy()
        del new_svo_df
        return svo_df,text_doc, sentence_embeddings
