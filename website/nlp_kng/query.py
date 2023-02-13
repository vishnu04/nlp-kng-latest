from regex import F
import textacy
import pandas as pd
from . import config
import itertools 
from . import synonyms_extractor
from . import svo_extractor 
import re
from . import stanza_svo_extractor
from . import models
from . import graph_traverse

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

def get_detailed_answer_index(detailed_answer, svo_df):
    for index, row in svo_df.iterrows():
        sent_true = True
        for word in detailed_answer.split(' '):
            if word not in row['sentence']:
                sent_true = False
            if sent_true == False:
                break
        if sent_true == True:
            return index
    return -1

@config.timer
def short_answer_question(quest,spacy_doc,nlp, svo_df,detailed_answer_index,danswers_df, G, pos, edge_labels, check_cause = True):
    question_doc = nlp(quest.lower())
    question_lemma=[]
    question_pos = []
    answer_facts = []
    if len(detailed_answer_index) == 0:
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
            # print(i,pos)
            if pos in ['NOUN','PROPN']:
                question_nouns.append(question_lemma[i])
            if pos in ['VERB','ROOT']:
                question_verbs.append(question_lemma[i])
        # print(f'short_answer question verbs:{question_verbs}')
        # print(f'short_answer question nouns:{question_nouns}')
        if len(question_verbs) == 0:
            question_verbs = question_nouns.copy()
        # print(f'short_answer question verbs:{question_verbs}')
        # print(f'short_answer question nouns:{question_nouns}')
        question_verbs = synonyms_extractor.get_synonyms(question_verbs,svo_df)
        question_verbs, question_verbs_pos = generate_lemma(question_verbs,nlp)
        for noun in question_nouns:
            for verb in question_verbs:
                if noun != verb:
                    if check_cause:
                        # print(f'short answer - checking object --> noun:{noun}\tverb:{verb}')
                        sv = svo_df.loc[(svo_df['object'].str.contains(noun) & svo_df['verb'].str.contains(verb)) & svo_df['reason_flag'] == True]
                    else:
                        # print(f'short answer - checking subject --> noun:{noun}\tverb:{verb}')
                        sv = svo_df.loc[(svo_df['subject'].str.contains(noun) & svo_df['verb'].str.contains(verb)) & svo_df['reason_flag'] == True] 
                        sv = svo_df.loc[(svo_df['object'].str.contains(noun) & svo_df['verb'].str.contains(verb)) & svo_df['reason_flag'] == True] 
                    for index, svo in sv.iterrows():
                        answers_df.loc[len(answers_df)] = svo
        answers_df = answers_df.drop_duplicates()
        new_answers_df = pd.DataFrame(columns = ['subject','verb','object'])
        for index, svo_tuple in answers_df.iterrows():
            new_subject = ' '.join([tok.lemma_ for tok in nlp(svo_tuple.subject)])
            new_verb = ' '.join([tok.lemma_ for tok in nlp(svo_tuple.verb)])
            new_object = ' '.join([tok.lemma_ for tok in nlp(svo_tuple.object)])
            new_answers_df.loc[len(new_answers_df)] = [new_subject, new_verb, new_object]
        new_answers_df = new_answers_df.drop_duplicates()
        answers_df = new_answers_df.copy()
        if len(answers_df) > 0:
            for index, svo in answers_df.iterrows():
                if svo.subject in question_nouns or svo.object in question_nouns:
                    unique_statements.add(svo.subject+' '+svo.verb+' '+svo.object)
        if len(unique_statements) > 0:
            no_of_facts = len(unique_statements)
            n = 1
            for statement in unique_statements:
                answer_facts.append(f'\t{statement}')
                n += 1
            if len(answer_facts) >0 :
                return answer_facts, question_lemma, question_pos
        return unique_statements, question_lemma, question_pos
    else: ## if detailed answer is not None
        
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
            # print(i,pos)
            if pos in ['NOUN','PROPN']:
                question_nouns.append(question_lemma[i])
            if pos in ['VERB','ROOT']:
                question_verbs.append(question_lemma[i])
        # print(f'short_answer question verbs:{question_verbs}')
        # print(f'short_answer question nouns:{question_nouns}')
        if len(question_verbs) == 0:
            question_verbs = question_nouns.copy()
        # print(f'short_answer question verbs:{question_verbs}')
        # print(f'short_answer question nouns:{question_nouns}')
        question_verbs = synonyms_extractor.get_synonyms(question_verbs,svo_df)
        question_verbs, question_verbs_pos = generate_lemma(question_verbs,nlp)




        n = 1
        # print(f'svo_df --> {svo_df}')
        # print(f' danswers_df --> {danswers_df}')
        # print(f'detailed_answer_index --> {detailed_answer_index}')
        networkx_answers = []
        danswers_df_index_count = 0
        for answer_index in detailed_answer_index:
            # print(f'answer_index ---> {answer_index}')
            # danswers_row = danswers_df[danswers_df['index'] == answer_index]
            danswers_row = danswers_df.iloc[danswers_df_index_count]
            # row = svo_df[svo_df['sentence'] == danswers_row['output_str'].to_string(index=False)]
            row = svo_df.iloc[answer_index]
            # print(row)
            # print(danswers_row)
            # score = danswers_row['confidence'].to_string(index=False)
            score = danswers_row['confidence']
            # print(score)
            row_output = row.subject+' '+row.verb+' '+row.object
            answer_facts.append(f'{n}. [score:{score}] - {row_output}')
            # networkx_answers.append(graph_traverse.graph_traverse(G,edge_labels,row.subject,row.object, row.verb, svo_df))
            n += 1
            danswers_df_index_count += 1
        if len(networkx_answers) == 0:
            # print('quest', quest)
            qsubjects = []
            qobjects = []
            qverbs = []
            for tok in nlp(quest):
                if tok.pos_ in ['PRON','NOUN','JJ']:
                    qsubjects.append(tok)
                if tok.pos_ in ['VERB']:
                    qverbs.append(tok)
            
            qsubjects = list(set(qsubjects))
            qverbs = list(set(qverbs))
            if len(qsubjects) == 1:
                for qsubject in qsubjects:
                    if len(qverbs) == 0:
                        qverbs.append(None)
                    for qverb in qverbs:
                        networkx_answers.append(graph_traverse.graph_traverse(G,edge_labels,qsubject,None,qverb,svo_df))
                        networkx_answers.append(graph_traverse.graph_traverse(G,edge_labels,None,qsubject,qverb,svo_df))
            else:
                for qsubject in qsubjects:
                    for qobject in qsubjects:
                        if len(qverbs) == 0:
                            qverbs.append(None)
                        if str(qsubject) in str(qobject) and str(qobject) in str(qsubject):
                            pass
                        else:
                            for qverb in qverbs:
                                networkx_answers.append(graph_traverse.graph_traverse(G,edge_labels,qsubject,qobject,qverb,svo_df))
                                networkx_answers.append(graph_traverse.graph_traverse(G,edge_labels,qsubject,qobject,qverb,svo_df))
        if len(networkx_answers) > 0:
            answer_facts.append(f' ')
            answer_facts.append(f'GraphTraverse Answers ====> ')
            n = 1
            new_networkx_answers = []
            for nx_answer in networkx_answers:
                if len(nx_answer) > 0 and nx_answer not in new_networkx_answers:
                    answer_facts.append(f'{n}. {nx_answer}')
                    n += 1
                    new_networkx_answers.append(nx_answer)
        return answer_facts, question_lemma, question_pos
    
    

def generate_lemma(words_list, nlp):
    lemma_list = []
    pos_list = []
    for word in words_list:
        for i, q_ngrams in enumerate(list(textacy.extract.ngrams(nlp(word),1))):
            lemma_list.append(' '.join([listitem.lemma_ for listitem in q_ngrams]))
            pos_list.append(' '.join([listitem.pos_ for listitem in q_ngrams]))
            lemma_list.append(' '.join([listitem.text for listitem in q_ngrams]))
            pos_list.append(' '.join([listitem.pos_ for listitem in q_ngrams]))
            lemma_list.append(' '.join([listitem.text for listitem in q_ngrams]))
            pos_list.append(' '.join([tok.pos_ for tok in nlp(str(q_ngrams))]))
    return lemma_list, pos_list

@config.timer
def detailed_answer_question(model, quest,spacy_doc,nlp, svo_df, question_embedding, sentence_embeddings,question_svo_df,check_cause = True):
    question_doc = nlp(quest)
    question_lemma=[]
    question_pos = []
    qa_mpnet_base_model = model
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
    if len(question_verbs) == 0:
        question_verbs = question_nouns.copy()
    question_verbs = synonyms_extractor.get_synonyms(question_verbs,svo_df)
    question_verbs, question_verbs_pos = generate_lemma(question_verbs,nlp)

    # for noun in question_nouns:
    #     for verb in question_verbs:
    #         print(f'Detailed answer --> {noun} {verb}')
    #         if noun != verb:
    #             statements = textacy.extract.semistructured_statements(spacy_doc,entity= noun, cue= verb,max_n_words = 200,)
    #             res = 1
    #             if res is None:
    #                 res = 1
    #             try:
    #                 if res:
    #                     for statement in statements:
    #                         unique_statements.add(statement)
    #             except Exception as e:
    #                 print(f'Error in detailed answers extract. textacy : {e}')
    answer_facts = []
    answer_index = []
    if len(unique_statements) < 0 :
        no_of_facts = len(unique_statements)
        n = 1
        for statement in unique_statements:
                # print(f'Detailed Answer --> statement --> {statement}')
                entity, cue, fact = statement
                if len(fact) > 0:
                    if str(entity).lower() != str(cue).lower():
                        sentence = str(entity)+' '+ str(cue)+ ' '+ str(fact)
                        if check_cause:
                            check_flag, reason_flag = stanza_svo_extractor.change_subject_object (sentence, str(entity), str(cue), str(fact))
                            # print(f'check_cause:{check_cause}, reason_flag:{reason_flag}, sentence: {sentence}')
                            if reason_flag is False:
                                answer_index.append(get_detailed_answer_index(str(entity)+' '+str(cue)+' '+str(fact), svo_df))
                                if answer_index[-1] == -1:
                                    answer_index.pop()
                                    pass
                                else:
                                    danswer_facts, question_lemma, question_pos, danswer_index , danswers_df = models.call_qa_mpnet(qa_mpnet_base_model,quest, svo_df.filter(items = [answer_index[-1]], axis = 0), question_embedding, sentence_embeddings[answer_index[-1]], question_svo_df)
                                    answer_facts.append(f'{n}. {danswer_facts[0]}')
                                    n += 1
                            else:
                                print('Reason not present in object')
                                # n += 1
                        elif check_cause == False:
                            check_flag, reason_flag = stanza_svo_extractor.change_subject_object (sentence, str(entity), str(cue), str(fact))
                            # print(f'check_cause:{check_cause}, reason_flag:{reason_flag}, sentence: {sentence}')
                            if reason_flag:
                                answer_index.append(get_detailed_answer_index(str(entity)+' '+str(cue)+' '+str(fact), svo_df))
                                if answer_index[-1] == -1:
                                    answer_index.pop()
                                    pass
                                else:
                                    danswer_facts, question_lemma, question_pos, danswer_index, danswers_df = models.call_qa_mpnet(qa_mpnet_base_model,quest, svo_df.filter(items = [answer_index[-1]], axis = 0), question_embedding, sentence_embeddings[answer_index[-1]], question_svo_df)
                                    answer_facts.append(f'{n}. {danswer_facts[0]}')
                                    n += 1
                            else:
                                print('Noun not present in subject')
                        else:
                            pass
        if len(answer_facts) > 0:
            return answer_facts, question_lemma, question_pos, answer_index, danswers_df
    else:
        print(f'Calling else in detailed answer')
        danswer_facts, question_lemma, question_pos, answer_index, danswers_df = models.call_qa_mpnet(qa_mpnet_base_model, quest, svo_df,question_embedding, sentence_embeddings, question_svo_df)
        n = 1
        for answer in danswer_facts:
            if answer.strip()[-1] == '?':
                pass
            else:
                answer_facts.append(f'{n}. {answer}')
                n +=1 
        return answer_facts, question_lemma, question_pos, answer_index,danswers_df
    # print('Query detailed answer-- Returning unique_statements')
    answer_facts = []
    answer_index = []
    # print(f'query - detailed answer - len of svo_df: {len(svo_df)}')
    for statement in unique_statements:
        entity, cue, fact = statement
        answer_index.append(get_detailed_answer_index(str(entity)+' '+str(cue)+' '+str(fact), svo_df))
        if answer_index[-1] == -1:
            answer_index.pop()
            pass
        else:
            danswer_facts, question_lemma, question_pos, danswer_index, danswers_df = models.call_qa_mpnet(qa_mpnet_base_model, quest, svo_df.filter(items = [answer_index[-1]], axis = 0), question_embedding, sentence_embeddings[answer_index[-1]],question_svo_df)
            answer_facts.append(f'{n}. {danswer_facts[0]}')
    return answer_facts, question_lemma, question_pos, answer_index, danswers_df
