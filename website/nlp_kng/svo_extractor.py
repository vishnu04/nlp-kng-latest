## Installation of neuralcoref: https://github.com/huggingface/neuralcoref

from xml.etree.ElementTree import TreeBuilder
from spacy import displacy
import textacy
import pandas as pd
from . import config, models
from spacy.tokens import DocBin
from . import stanza_svo_extractor
import collections
from typing import Iterable, List, Optional, Pattern, Tuple
from spacy.tokens import Doc, Span, Token

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
    SVOTriple: Tuple[List[Token], List[Token], List[Token]] = collections.namedtuple("SVOTriple", ["subject", "verb", "object"])
    '''For displaying in webpage'''
    # displacy.render(text_doc,style="ent")

    '''Extracting SVO triplets into dataframe'''
    svo_df = pd.DataFrame(columns = config.SUB_VERB_OBJ_DF_COLS)
    # sentences = [sent for sent in text_doc.sents]
    sentences = []
    for lines in text.splitlines():
        sentences.append(nlp(lines))
    # print(f'Svo extract - len of sentences --> : {len(sentences)}')
    qa_mpnet_base_model = model
    for sentence in sentences:
        # print(f'SVO extract --> {sentence}')
        sentence_tuples = []
        sentence_tuples = extract_SVO(sentence)
        # print(f'len(sentence_tuples) --> {sentence_tuples[0]}')
        stanza_svo_df = pd.DataFrame(columns = config.SUB_VERB_OBJ_DF_COLS)
        if len(sentence_tuples[0]) == 0:
            stanza_svo_df,reason_present, sentence_embeddings = stanza_svo_extractor.triplet_extraction(text_doc,nlp,model,str(sentence), output = ['result'])
            if len(stanza_svo_df) > 0:
                for index, stanza_row in stanza_svo_df.iterrows():
                    # print(stanza_row)
                    # sentence_tuples.append([stanza_row.subject, stanza_row.verb, stanza_row.object])
                    sentence_tuples.append([SVOTriple(subject=[stanza_row.subject], verb=[stanza_row.verb], object=[stanza_row.object])])
        # print(f'sentence_tuples --> {sentence_tuples}')
        for svoTriple in sentence_tuples:
            # print(svoTriple, len(svoTriple))
            if len(svoTriple) > 0 :
                for s,v,o in svoTriple:
                    subj = ' '.join([str(n) for n in s])
                    verb = ' '.join([str(n) for n in v])
                    obj = ' '.join([str(n) for n in o])
                    check_flag, reason_flag = stanza_svo_extractor.change_subject_object(str(sentence), subj, verb, obj)
                    # input_sent_embedding = models.gen_qa_mpnet_embeddings(qa_mpnet_base_model, str(sentence))
                    # svo_df.loc[len(svo_df)] = [subj,verb,obj, sentence, reason_flag,input_sent_embedding]
                    if check_flag == True:
                        svo_df.loc[len(svo_df)] = [obj,verb,subj, str(sentence), reason_flag]
                    else:
                        svo_df.loc[len(svo_df)] = [subj,verb,obj, str(sentence), reason_flag]
    new_svo_df = pd.DataFrame(columns = config.SUB_VERB_OBJ_DF_COLS)
    for index, svo_tuple in svo_df.iterrows():
        # print(f'{svo_tuple} -- {[tok.pos_ for tok in nlp(svo_tuple.subject)]} {[tok.pos_ for tok in nlp(svo_tuple.object)]}')
        if set(["PROPN",'NOUN']) & set([tok.pos_ for tok in nlp(svo_tuple.subject)]):
            if(set(["PROPN",'NOUN']) & set([tok.pos_ for tok in nlp(svo_tuple.object)])):
                # new_svo_df.loc[len(new_svo_df)] = [svo_tuple.subject,svo_tuple.verb, svo_tuple.object, svo_tuple.sentence, svo_tuple.reason_flag, svo_tuple.qa_base_mpnet_embeddings]
                new_svo_df.loc[len(new_svo_df)] = [svo_tuple.subject,svo_tuple.verb, svo_tuple.object, svo_tuple.sentence, svo_tuple.reason_flag]
            elif set(['PROPN','NOUN','VERB']) & set([tok.pos_ for tok in nlp(svo_tuple.sentence)]):
                new_svo_df.loc[len(new_svo_df)] = [svo_tuple.subject,svo_tuple.verb, svo_tuple.object, svo_tuple.sentence, svo_tuple.reason_flag]
        elif set(['PROPN','NOUN','VERB']) & set([tok.pos_ for tok in nlp(svo_tuple.sentence)]):
                new_svo_df.loc[len(new_svo_df)] = [svo_tuple.subject,svo_tuple.verb, svo_tuple.object, svo_tuple.sentence, svo_tuple.reason_flag]
    new_svo_df = new_svo_df.drop_duplicates()
    svo_df = new_svo_df.copy()
    del new_svo_df
    
    # svo_df, text_doc, embeddings = extract_embeddings(qa_mpnet_base_model, text_doc, nlp, svo_df, text)
    return svo_df, text_doc, None
  

@config.timer
def extract_embeddings(model, text_doc, nlp, svo_df, text):
    print(f'Inside extract_embeddings')
    if len(svo_df) > 1:
        # svo_df,reason_present, sentence_embeddings = stanza_svo_extractor.triplet_extraction(text_doc,nlp,qa_mpnet_base_model,text, output = ['result'])
        input_sent_embeddings = models.gen_qa_mpnet_embeddings(model, list(svo_df['sentence']))
        return svo_df,text_doc, input_sent_embeddings
    else:
        print(f'Extracting using stanza ---->')
        svo_df,reason_present, sentence_embeddings = stanza_svo_extractor.triplet_extraction(text_doc,nlp,model,text, output = ['result'])
        new_svo_df = pd.DataFrame(columns = config.SUB_VERB_OBJ_DF_COLS)
        new_svo_df = svo_df.drop_duplicates()
        svo_df = new_svo_df.copy()
        del new_svo_df
        return svo_df,text_doc, sentence_embeddings