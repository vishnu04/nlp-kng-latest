import nltk, pandas as pd, numpy as np
from nltk.parse.corenlp import CoreNLPParser, CoreNLPDependencyParser, CoreNLPServer
from nltk.tree import ParentedTree
from . import config
import os
import re
from . import synonyms_extractor, query, models
# dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
# pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
os.environ['CLASSPATH'] = config.STANZA_PATH
def triplet_extraction (text_doc, nlp, model, text, output=['parse_tree','spo','result']):
    # STANFORD = os.path.join(os.getcwd(), "stanford-corenlp-4.5.1")
    # server = CoreNLPServer(os.path.join(STANFORD, "stanford-corenlp-4.5.1.jar"),\
    #                       os.path.join(STANFORD, "stanford-corenlp-4.5.1-models.jar"),
    #                     )
    input_sent_embeddings = None
    # try:
    #     print('Starting stanza server -->')
    #     server.start()
    # except Exception as e:
    #     print(f'Exception 1 --> {e}')
    #     try:
    #         print('Stoping stanza server -->')
    #         server.stop()
    #         print('Starting stanza server -->')
    #         server.start()
    #     except Exception as e:
    #         print(f'Exception 2 --> {e}')
    #         print('Failed to start Stanza server -->')
    #         svo_df = pd.DataFrame(columns = config.SUB_VERB_OBJ_DF_COLS)
    #         return svo_df, False, input_sent_embeddings
    # dep_parser = CoreNLPDependencyParser()
    # pos_tagger = CoreNLPParser(tagtype='pos')
    pos_tagger = None
    svo_df = pd.DataFrame(columns = config.SUB_VERB_OBJ_DF_COLS)
    try:
        pos_tagger = CoreNLPParser(url='http://localhost:9001', tagtype='pos')
        # qa_mpnet_base_model = models.load_qa_mpnet_base_model()
        qa_mpnet_base_model = model
        sentences = []
        for lines in text.splitlines():
            sentences.append(nlp(lines))
        print(f'Stanza Svo extract - len of sentences --> : {len(sentences)}')
        # if text_doc != '':
        #     sentences = [sent for sent in text_doc.sents]
        # else:
        #     sentences = [text]
        for input_sentence in sentences:
            input_sent = str(input_sentence)
            pos_type = pos_tagger.tag(input_sent.split())
            parse_tree, = ParentedTree.convert(list(pos_tagger.parse(input_sent.split()))[0])
            # print('Parse_tree -----------------> ')
            # print(list(pos_tagger.parse(input_sent.split()))[0])
            # dep_type, = ParentedTree.convert(dep_parser.parse(input_sent.split()))
            # Extract subject, predicate and object
            subject = extract_subject(parse_tree)
            predicate = extract_predicate(parse_tree)
            objects = extract_object(parse_tree)
            if 'parse_tree' in output:
                # print('---Parse Tree---')
                # parse_tree.pretty_print()
                pass
            if 'spo' in output:
                # print('---Subject---')
                # print(subject)
                # print('---Predicate---')
                # print(predicate)
                # print('---Object---')
                # print(objects)
                pass
            if 'result' in output:
                # print('---Result---')
                change_obj_sub, reason_flag = change_subject_object(input_sent, subject[0], predicate[0],objects[0]) 
                # print(subject[0],predicate[0],objects[0],input_sent)  
                if change_obj_sub:
                    if len(objects[0].strip()) !=0 and len(predicate[0].strip()) != 0 and len(subject[0].strip()) != 0:
                        svo_df.loc[len(svo_df)] = [objects[0],predicate[0],subject[0], input_sent, reason_flag]
                    else:
                        pass
                else:
                    if len(objects[0].strip()) !=0 and len(predicate[0].strip()) != 0 and len(subject[0].strip()) != 0:
                        svo_df.loc[len(svo_df)] = [subject[0],predicate[0],objects[0], input_sent, reason_flag]
                    else:
                        pass

        # server.stop()
        # print(f'Stanza_extract SVO')
        # print(svo_df.head(5))
        
        print(f'triplet_extraction ---> {len(svo_df)}')
        if len(svo_df) > 0:
            print(f'triplet_extraction ---> {len(svo_df)}')
            # print(f'triplet_extraction --->')
            # print(list(svo_df['sentence']))
            input_sent_embeddings = models.gen_qa_mpnet_embeddings(qa_mpnet_base_model, list(svo_df['sentence']))
        else:
            input_sent_embeddings = models.gen_qa_mpnet_embeddings(qa_mpnet_base_model, list(sentences))
        return svo_df, reason_flag,input_sent_embeddings
    except Exception as e:
        print(f'Exception CoreNLPParser -->: {e}')
        return svo_df, False,None

def change_subject_object (sentence, subj, verb, obj):
    change_sub_obj = False
    reason_loc = float('inf')
    reason_loc_list = []
    main_verb_loc = -1
    word_list = re.findall(r'\w+', sentence)
    reason_list = ['by','due','results', 'effects of', 'effect of', 
                'affects of', 'affect of', 'due to', 'impacts of',
                'impact of', 'is caused due','caused by'
                # 'reason for', 'reasons for',
                ]
    # syn_reason_list = synonyms_extractor.get_verb_synonyms(reason_list)
    # reason_list += syn_reason_list
    reason_present = False
    try:
        for reason in reason_list:
            try:
                reason = re.findall(r'\w+', reason)
                if all( r in word_list for r in reason):
                    reason_present = True
                    for r in reason:
                        reason_loc_list.append(word_list.index(r))
                    reason_loc = min(reason_loc_list)
                    break
                else:
                    pass
            except:
                print(f'Reason not present.')
                pass
    except:
        print(f'Reason not present.')
        pass

    try:
        main_verb_loc = word_list.index(verb)
        subject_loc = word_list.index(subj)
        object_loc = float('inf')
        try:
            for ob in re.findall(r'\w+', obj):
                object_loc = min(word_list.index(ob),object_loc)
        except:
            pass
        if reason_loc > main_verb_loc and subject_loc < main_verb_loc and reason_loc < object_loc:
            change_sub_obj = True
    except:
        pass
    return change_sub_obj, reason_present
    

def extract_subject (parse_tree):
    # Extract the first noun found in NP_subtree
    subject = []
    for s in parse_tree.subtrees(lambda x: x.label() == 'NP'):
        for t in s.subtrees(lambda y: y.label().startswith('NN')):
            output = [t[0], extract_attr(t)]
            # Avoid empty or repeated values
            if output != [] and output not in subject:
                subject.append(output) 
    if len(subject) != 0: return subject[0] 
    else: return ['']

def extract_predicate (parse_tree):
    # Extract the deepest(last) verb foybd ub VP_subtree
    output, predicate = [],[]
    for s in parse_tree.subtrees(lambda x: x.label() == 'VP'):
        for t in s.subtrees(lambda y: y.label().startswith('VB')):
            output = [t[0], extract_attr(t)]
            if output != [] and output not in predicate:    
                predicate.append(output)
    if len(predicate) != 0: return predicate[-1]
    else: return ['']

def extract_object (parse_tree):
    # Extract the first noun or first adjective in NP, PP, ADP siblings of VP_subtree
    objects, output, word = [],[],[]
    for s in parse_tree.subtrees(lambda x: x.label() == 'VP'):
        for t in s.subtrees(lambda y: y.label() in ['NP','PP','ADP']):
            if t.label() in ['NP','PP']:
                for u in t.subtrees(lambda z: z.label().startswith('NN')):
                    word = u          
            else:
                for u in t.subtrees(lambda z: z.label().startswith('JJ')):
                    word = u
            if len(word) != 0:
                output = [word[0], extract_attr(word)]
            if output != [] and output not in objects:
                objects.append(output)
    
    sentence_object = ''
    for objs in objects:
        sentence_object = sentence_object + objs[0] + ' '
    if len(objects) != 0: 
        return [sentence_object]
    else: return ['']

def extract_attr (word):
    attrs = []     
    # Search among the word's siblings
    if word.label().startswith('JJ'):
        for p in word.parent(): 
            if p.label() == 'RB':
                attrs.append(p[0])
    elif word.label().startswith('NN'):
        for p in word.parent():
            if p.label() in ['DT','PRP$','POS','JJ','CD','ADJP','QP','NP']:
                attrs.append(p[0])
    elif word.label().startswith('VB'):
        for p in word.parent():
            if p.label() in ['ADVP']:
                attrs.append(p[0])
    # Search among the word's uncles
    if word.label().startswith('NN') or word.label().startswith('JJ'):
        for p in word.parent().parent():
            if p.label() == 'PP' and p != word.parent():
                attrs.append(' '.join(p.flatten()))
    elif word.label().startswith('VB'):
        for p in word.parent().parent():
            if p.label().startswith('VB') and p != word.parent():
                attrs.append(' '.join(p.flatten()))
    return attrs


