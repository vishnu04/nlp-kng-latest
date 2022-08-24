import textacy
import pandas as pd
from . import config
import itertools 
from . import synonyms_extractor
from . import svo_extractor 

def peek(iterable):
    try:
        first = next(iterable)
    except AttributeError:
        print('AttributeError')
        return None
    except Exception as e:
        print(f'Exception Peek: {e}')
        return None
    return first, itertools.chain([first], iterable)

@config.timer
def short_answer_question(quest,spacy_doc,nlp, svo_df):    
    question_doc = nlp(quest.lower())
    question_lemma=[]
    question_pos = []
    answer_facts = []
    for i, q_ngrams in enumerate(list(textacy.extract.ngrams(question_doc,1))):
        question_lemma.append(' '.join([listitem.lemma_ for listitem in q_ngrams]))
        question_pos.append(' '.join([listitem.pos_ for listitem in q_ngrams]))
        question_lemma.append(' '.join([listitem.text for listitem in q_ngrams]))
        question_pos.append(' '.join([listitem.pos_ for listitem in q_ngrams]))
        question_lemma.append(' '.join([listitem.text for listitem in q_ngrams]))
        question_pos.append(' '.join([tok.pos_ for tok in nlp(str(q_ngrams))]))
        
    '''Code to extract semistructed statements from SVO triplets'''
    unique_statements =set()
    question_nouns = []
    question_verbs = []
    answers_df = pd.DataFrame(columns = ['subject','verb','object'])
    for i, pos in enumerate(question_pos):
        print(i,pos)
        if pos in ['NOUN','PROPN']:
            question_nouns.append(question_lemma[i])
        if pos in ['VERB','ROOT']:
            question_verbs.append(question_lemma[i])
    print(f'short_answer question verbs:{question_verbs}')
    print(f'short_answer question nouns:{question_nouns}')
    question_verbs = synonyms_extractor.get_synonyms(question_verbs,svo_df)
    qsvo_df, qtext_doc = svo_extractor.svo(question_doc,nlp)
    print(qsvo_df)
    for noun in question_nouns:
        for verb in question_verbs:
            if noun != verb:
                #print(f'noun:{noun}\tverb:{verb}')
                sv = svo_df.loc[(svo_df['object'].str.contains(noun) & svo_df['verb'].str.contains(verb)) | \
                               (svo_df['subject'].str.contains(noun) & svo_df['verb'].str.contains(verb))]
                for index, svo in sv.iterrows():
                    answers_df.loc[len(answers_df)] = svo
    answers_df = answers_df.drop_duplicates()
    new_answers_df = pd.DataFrame(columns = ['subject','verb','object'])
    for index, svo_tuple in answers_df.iterrows():
        new_subject = ' '.join([tok.lemma_ for tok in nlp(svo_tuple.subject)])
        new_verb = ' '.join([tok.lemma_ for tok in nlp(svo_tuple.verb)])
        new_object = ' '.join([tok.lemma_ for tok in nlp(svo_tuple.object)])
        #print(new_subject, new_verb, new_object)
        new_answers_df.loc[len(new_answers_df)] = [new_subject, new_verb, new_object]
    new_answers_df = new_answers_df.drop_duplicates()
    answers_df = new_answers_df.copy()
    if len(answers_df) > 0:
        for index, svo in answers_df.iterrows():
            if svo.subject in question_nouns or svo.object in question_nouns:
                unique_statements.add(svo.subject+' '+svo.verb+' '+svo.object)
    # return unique_statements, question_lemma, question_pos
    if len(unique_statements) > 0:
        no_of_facts = len(unique_statements)
        n = 1
        for statement in unique_statements:
            answer_facts.append(f'\t{statement}')
            n += 1
        if len(answer_facts) >0 :
            return answer_facts, question_lemma, question_pos
    return unique_statements, question_lemma, question_pos

@config.timer
def detailed_answer_question(quest,spacy_doc,nlp, svo_df):
    question_doc = nlp(quest)
    question_lemma=[]
    question_pos = []
    answer_facts = []
    for i, q_ngrams in enumerate(list(textacy.extract.ngrams(question_doc,1))):
        question_lemma.append(' '.join([listitem.lemma_ for listitem in q_ngrams]))
        question_pos.append(' '.join([listitem.pos_ for listitem in q_ngrams]))
        question_lemma.append(' '.join([listitem.text for listitem in q_ngrams]))
        question_pos.append(' '.join([listitem.pos_ for listitem in q_ngrams]))
        question_lemma.append(' '.join([listitem.text for listitem in q_ngrams]))
        question_pos.append(' '.join([tok.pos_ for tok in nlp(str(q_ngrams))]))
    
    '''Code to extract semistructed statements from text_doc'''
    unique_statements =set()
    question_nouns = []
    question_verbs = []
    for i, pos in enumerate(question_pos):
        if pos in ['NOUN','PROPN']:
            question_nouns.append(question_lemma[i])
        if pos in ['ADV','VERB','ROOT']:
            question_verbs.append(question_lemma[i])
    # print(f'spacy_doc :{spacy_doc}')
    question_verbs = synonyms_extractor.get_synonyms(question_verbs,svo_df)
    for noun in question_nouns:
        for verb in question_verbs:
            if noun != verb:
                statements = textacy.extract.semistructured_statements(spacy_doc,entity= noun, cue= verb,max_n_words = 200,)
                # res = peek(statements)
                res = 1
                print(f' Res is none -----> {res} {noun} {verb}')
                if res is None:
                    res = 1
                try:
                    if res:
                        for statement in statements:
                            unique_statements.add(statement)
                except Exception as e:
                    print(f'Error in detailed answers extract. textacy : {e}')
    if len(unique_statements) > 0 :
        no_of_facts = len(unique_statements)
        n = 1
        for statement in unique_statements:
                entity, cue, fact = statement
                if len(fact) > 0:
                    if str(entity).lower() != str(cue).lower():
                        answer_facts.append(f'\t{n}. {entity} - {cue} - {fact}')
                        n += 1
        return answer_facts, question_lemma, question_pos
    return unique_statements, question_lemma, question_pos
