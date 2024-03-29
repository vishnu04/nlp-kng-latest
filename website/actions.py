from venv import create
from flask import Blueprint, render_template, request, flash, redirect, url_for, send_file, session
from .nlp_kng import scrapper, cleaner, svo_extractor, kg_generator, query, config, create_temp_dir, cleantmp, stanza_svo_extractor, models
import spacy
from io import BytesIO
import networkx as nx
import matplotlib.pyplot as plt
import base64
import pandas as pd 
from PIL import Image
import os 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from spacy.tokens import Doc
import torch
import numpy as np
from flask import jsonify
from io import StringIO

'''
Code to save nlp vocab to tmp folder.
'''
plt.switch_backend('Agg') 

def save_nlp_to_disk(text_doc):
    try:
        tmpdir = create_temp_dir.createTempDir()
        text_nlp = tmpdir+'/spacy_nlp.nlp'
        text_doc.to_disk(text_nlp)
        text_vocab = tmpdir+'/spacy_vocab.voc'
        text_doc.vocab.to_disk(text_vocab)
        # print(f'tmpdir : {tmpdir}')
        return tmpdir
    except Exception as e:
        print(f'Unable to save NLP to disk {e}')
        return None

'''
Main code - called once at the start of the session.
'''
actions = Blueprint('actions',__name__)
## Cleaning tmp folder
cleantmp.main()
nlp = spacy.load(config.BASE_MODEL)
nlp.add_pipe('sentencizer')

'''Loading qa_mpnet_base_model'''
qa_mpnet_base_model = models.load_qa_mpnet_base_model()

@actions.route('/',methods=['GET'])
def home():
    # print(f'calling home - {request.method}')
    return render_template('home.html')

@actions.route('/scrape', methods=['POST'])
def scrape():
    # print(f'calling scrape {request.method}')
    ## Cleaning tmp folder
    cleantmp.main()
    if request.method == 'POST':
        web_url = request.form.get("weburl")
        if len(web_url) <1 :
            flash('URL is too short! Please enter correct URL. e.g: https://textacy.readthedocs.io/', category='error')
            return render_template('home.html')
        else:
            try:
                text = scrapper.scrape_text(web_url)
                text_df,text = cleaner.clean_data(text)
                # print(f'length of text scraped:{len(text)}')
                if len(text) > 1:
                    return render_template('home.html', weburl = web_url, cleantext = text, display_svo = False)
                else:
                    flash('URL entered cannot be scraped !. Please enter the text from the weburl entered.', category='error')
                    text = 'NA'
                    return render_template('home.html', weburl = web_url, cleantext = text, display_svo = False)
            except:
                flash('URL entered cannot be scraped !. Please enter correct URL. e.g: https://textacy.readthedocs.io/', category='error')
                return render_template('home.html', weburl = web_url)


@actions.route('/extract', methods=['POST'])
def extract():
    # print(f'Extracting triplets {request.method}')
    if request.method == 'GET':
        return render_template('home.html')
    if request.method == 'POST':
        web_url = request.form.get('weburl')
        text = request.form.get('cleantext')
        # print(f'length of clean text: {len(text)}')
        # print(f'no of sentences before clean--> {len(text.splitlines())}')
        if len(text) > 0:
            text_df,text = cleaner.clean_data(text)
        # print(f'no of sentences after clean--> {len(text.splitlines())}')
        
        # if len(text_df) > 0:
        #     text_df.to_csv('text_df.csv', index = True)
        '''defining the pipeline'''        
        text = text.lower()
        text_doc = nlp(text)
        n_word_tokens = len(word_tokenize(text))

        svo_df, text_doc, sentence_embeddings = svo_extractor.svo(text_doc,text,nlp,qa_mpnet_base_model)
        # print(f' Len of svo_df --> {len(svo_df)}')
        svo_df, text_doc, sentence_embeddings = svo_extractor.extract_embeddings(qa_mpnet_base_model, text_doc, nlp, svo_df, text)
        # print(f'n_word_tokens : {n_word_tokens}')
        if len(svo_df) > 0:
            svo_df.to_csv('svo_df.csv', index = True)

        ## Saving NLP
        tmpdir = save_nlp_to_disk(text_doc)

        triplets_found = False
        headings = svo_df.columns[0:3]
        data_tuple = []
        # print(f'svo_df length - extract : {len(svo_df)}')
        if len(svo_df) >= 1:
            triplets_found = True
            for index, row in svo_df.iterrows():
                data_tuple.append((row.subject, row.verb, row.object))
            G, pos, edge_labels = kg_generator.plot_kg(svo_df)
            data_tuple = tuple(data_tuple)
            
            nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
            nx.draw_networkx_edge_labels(G, pos = pos,edge_labels = edge_labels)

            figfile = BytesIO()

            plt.savefig(figfile, format='png')
            plt.close()

            figfile.seek(0)
            figdata_png = base64.b64encode(figfile.getvalue()).decode('ascii')
            
            session['networkxGraph'] = G
            session['networkxGraphPos'] = pos
            session['networkxGraph_labels'] = edge_labels
            session['image_base64'] = figdata_png
            session['svo_df'] = svo_df.to_json(default_handler=str)

            # print(f'Saving sentence_embeddings to numpy')
            if isinstance(sentence_embeddings, torch.Tensor):
                # print('sentence_embeddings size -----> ', sentence_embeddings.size())
                sentence_embeddings_numpy = sentence_embeddings.cpu().numpy()
                session['sentence_embeddings'] = sentence_embeddings_numpy
                
            # result_optimization.print_temp_dir()
            print(f'Coding -----------********************')
            return render_template('home.html', weburl = web_url,
                                    image_base64 = figdata_png,
                                    matrix = str(nx.to_numpy_array(G)).replace('.',',').replace('\n',','),
                                    graph = G,
                                    cleantext = text, triplets_found = triplets_found, 
                                    headings = headings, data = data_tuple,
                                    svo_df = svo_df, 
                                    svo_table = [svo_df.to_html()], titles=[''],
                                    text_doc = text_doc,
                                    nlp = nlp,
                                    display_svo = True,
                                    tmpdir = tmpdir)
        else:
            
            session['svo_df'] = svo_df.to_json()
            cwd = os.getcwd()
            img = Image.open(str(cwd)+"/website/static/images/no-image-available.jpeg")
            buf = BytesIO()
            img.save(buf, format = 'jpeg')
            img.close()
            buf.seek(0)
            encoded_string = base64.b64encode(buf.getvalue()).decode('ascii')
            session['image_base64'] = encoded_string
            return render_template('home.html',image_base64=encoded_string, weburl = web_url, cleantext = text, 
                                    triplets_found = triplets_found, headings = headings, data = data_tuple, 
                                    svo_df = svo_df, 
                                    svo_table = [svo_df.to_html()], titles=[''],
                                    text_doc = text_doc,
                                    nlp = nlp,
                                    display_svo = True,
                                    tmpdir = tmpdir)

@actions.route('/queryQuestion', methods=['POST'])
def queryQuestion():
    # print(f'calling queryQuestion {request.method}')
    if request.method == 'POST':
        web_url = request.form.get('weburl')
        figdata_png = session.get('image_base64')
        text = request.form.get('cleantext')
        triplets_found = request.form.get('triplets_found')
        headings = request.form.get('headings')
        data_tuple = request.form.get('data')
        
        G = session.get('networkxGraph')
        pos = session.get('networkxGraphPos')
        edge_labels = session.get('networkxGraph_labels')
        # session['networkxGraph'] = G
        # session['networkxGraphPos'] = pos
        # session['networkxGraph_labels'] = edge_labels


        # print(create_temp_dir.get_temp_dir())
        ## Loading NLP
        try:
            if request.form.get('tmpdir') is None or request.form.get('tmpdir') == "None" \
                or request.form.get('tmpdir') == "":
                if create_temp_dir.get_temp_dir() is None or create_temp_dir.get_temp_dir() == "None" \
                    or create_temp_dir.get_temp_dir() == "":
                    text_doc = nlp(text)
                    save_nlp_to_disk(text_doc)
                else:
                    text_doc = Doc(nlp.vocab).from_disk(create_temp_dir.get_temp_dir()+'/spacy_nlp.nlp')
            else:
                text_doc = Doc(nlp.vocab).from_disk(request.form.get('tmpdir')+'/spacy_nlp.nlp')
        except Exception as e:
            print(f'Exception reading spacy nlp: {e}')
            text_doc = nlp(text)
            save_nlp_to_disk(text_doc)

        new_sentence_embeddings = None
        sentence_embeddings = None
        try:
            svo_df = pd.read_json(session.get('svo_df'), dtype=False)
        except Exception as e:
            print(f'Error svo_df actions - {e}')
            svo_df, text_doc, sentence_embeddings = svo_extractor.svo(text_doc,text,nlp,qa_mpnet_base_model)
            svo_df, text_doc, sentence_embeddings = svo_extractor.extract_embeddings(qa_mpnet_base_model, text_doc, nlp, svo_df, text)
            new_sentence_embeddings = sentence_embeddings

        try:
            print(f'Reading sentence embeddings from numpy and converting to torch')
            old_sentence_embeddings = torch.from_numpy(np.fromstring(session['sentence_embeddings'], dtype = np.float32).reshape(len(svo_df),768))
            sentence_embeddings = old_sentence_embeddings
            del old_sentence_embeddings
        except Exception as e:
            print(f'Error while retrieving the sentence embeddings from numpy: {e}')
            try:
                if new_sentence_embeddings is not None:
                    sentence_embeddings = new_sentence_embeddings
                    del new_sentence_embeddings
            except Exception as e:
                print(f'Error while assigning new_sentence_embeddings: {e}')
                svo_df, text_doc, sentence_embeddings = svo_extractor.svo(text_doc,text,nlp,qa_mpnet_base_model)
                svo_df, text_doc, sentence_embeddings = svo_extractor.extract_embeddings(qa_mpnet_base_model, text_doc, nlp, svo_df, text)
        
            

        headings = svo_df.columns[0:3]
        data_tuple = []
        if len(svo_df) >= 1:
            triplets_found = True
            for index, row in svo_df.iterrows():
                data_tuple.append((row.subject, row.verb, row.object))
            data_tuple = tuple(data_tuple)

        question = request.form.get('question')
        print(f'Question : {question}')
        short_answers = ''
        detailed_answers = ''
        question_svo_df, check_cause, question_embedding = stanza_svo_extractor.triplet_extraction('',nlp,qa_mpnet_base_model, question, output = ['result'])
        # print('question_svo_df --> ', question_svo_df)
        danswer, question_lemma_, question_pos_, danswer_index,danswers_df = query.detailed_answer_question(qa_mpnet_base_model,question,text_doc, nlp, svo_df, question_embedding, sentence_embeddings, question_svo_df, check_cause)
        # print(f'danswer_index --> {danswer_index}')
        # print(f'danswers_df --> {type(danswers_df)}')
        detailed_answers_found = False
        detailed_answer_length = 0
        if len(danswer) > 0:
            detailed_answers_found = True
            for ans in danswer:
                # print(f'detailed answers : {ans}')
                if ans.strip()[-1] == '?':
                    pass
                else:
                    detailed_answers +=  ans + '\n'
                    detailed_answer_length += 2
        answer, question_lemma_, question_pos_ = query.short_answer_question(question,text_doc, nlp, svo_df,danswer_index,danswers_df, G, pos, edge_labels, check_cause)
        # answer, question_lemma_, question_pos_ = [],[],[] #query.short_answer_question(question,text_doc, nlp, svo_df,danswer_index,danswers_df,check_cause)
        short_answers_found = False
        short_answer_length = 0
        if len(answer) > 0:
            short_answers_found = True
            for ans in answer:
                short_answers = '\n'.join(answer)
                short_answer_length += 1
        no_answer_found = False
        qanswer = ''
        if not short_answers_found:
            if not detailed_answers_found:
                no_answer_found = True
                qanswer = 'No answer found.'
        
        return render_template('home.html',image_base64=figdata_png, weburl = web_url, cleantext = text, 
                                        triplets_found = triplets_found, headings = headings, data = data_tuple, 
                                        svo_df = svo_df, 
                                        svo_table = [svo_df.to_html()], titles=[''],
                                        text_doc = text_doc,
                                        nlp = nlp,
                                        question = question,
                                        short_answers = short_answers,
                                        detailed_answers = detailed_answers,
                                        qanswer = qanswer,
                                        short_answers_found = short_answers_found,
                                        detailed_answers_found = detailed_answers_found,
                                        no_answer_found = no_answer_found,
                                        short_answer_length = short_answer_length,
                                        detailed_answer_length = detailed_answer_length,
                                        display_svo = True)

#############################################################################################
@actions.route('/api/v1/scrape', methods=['POST'])
def api_v1_scrape():
    print(f'calling scrape {request.method}')
    ## Cleaning tmp folder

    body=request.get_json(silent=True)
    web_url = body['url']
    print(web_url)

    try:
        text = scrapper.scrape_text(web_url)
        text_df,text = cleaner.clean_data(text)
        return jsonify({'msg':"OK",'cleanText': text})
    except Exception as e:
        print(f'ERROR scrap {e}')

    return jsonify({'msg': 'URL entered cannot be scraped!'})

@actions.route('/api/v1/extract', methods=['POST'])
def api_v1_extract():
    print(f'Extracting triplets {request.method}')

    body=request.get_json(silent=True)

    text = body['cleanText']
    # print(f'length of clean text: {len(text)}')
    # print(f'no of sentences before clean--> {len(text.splitlines())}')

    text_df,text = cleaner.clean_data(text)
    # print(f'no of sentences after clean--> {len(text.splitlines())}')

    text = text.lower()
    text_doc = nlp(text)
    n_word_tokens = len(word_tokenize(text))    
    svo_df, text_doc, sentence_embeddings = svo_extractor.svo(text_doc,text,nlp,qa_mpnet_base_model)
    # print(f' Len of svo_df --> {len(svo_df)}')
    svo_df, text_doc, sentence_embeddings = svo_extractor.extract_embeddings(qa_mpnet_base_model, text_doc, nlp, svo_df, text)
    # print(f'n_word_tokens : {n_word_tokens}')
    np.set_printoptions(threshold=np.inf)
    # print(f'Saving sentence_embeddings to numpy')
    if isinstance(sentence_embeddings, torch.Tensor):
        # print('sentence_embeddings size -----> ', sentence_embeddings.size())
        sentence_embeddings_numpy = sentence_embeddings.cpu().numpy()
        sentence_embeddings_str = np.array2string(sentence_embeddings_numpy)

    
    # print(sentence_embeddings_numpy)

    if len(svo_df) > 0:
        grf = svo_df.to_csv()

        # print(f'svo_df length - extract : {len(svo_df)}')
        if len(svo_df) >= 1:
            
            return jsonify({'msg':'OK','kgGraph':str(grf), 'embeddings':str(sentence_embeddings_str)})


    return jsonify({'msg':'No data extracted'})

@actions.route('/api/v1/queryQuestion', methods=['POST'])
def api_v1_queryQuestion():
    print(f'calling queryQuestion {request.method}')
    if request.method == 'POST':
        body=request.get_json(silent=True)
        question = body['question']
        embeddings = body['embeddings']
        svo_df = StringIO(body['kgGraph'].replace('\\n','\n'))
        svo_df = pd.read_csv(svo_df, sep = ",")
        svo_df = svo_df.iloc[:,1:]
        embeddings = str(embeddings).replace('[[','').replace('[','').replace(']','')
        sentence_embeddings = torch.from_numpy(np.fromstring(embeddings, dtype = np.float32, sep='  ').reshape(len(svo_df),768))
        G, pos, edge_labels = kg_generator.plot_kg(svo_df)
        print(f'Question : {question}')        
        
        short_answers = ''
        detailed_answers = ''
        question_svo_df, check_cause, question_embedding = stanza_svo_extractor.triplet_extraction('',nlp,qa_mpnet_base_model, question, output = ['result'])
        # print('question_svo_df --> ', question_svo_df)

        #return jsonify({'msg':'OK','shortAnswer':'Test short answer','sentence_embedding':str(sentence_embeddings)})

        danswer, question_lemma_, question_pos_, danswer_index,danswers_df = query.detailed_answer_question(qa_mpnet_base_model,question,None, nlp, svo_df, question_embedding, sentence_embeddings, question_svo_df, check_cause)
        # print(f'danswer_index --> {danswer_index}')
        # print(f'danswers_df --> {type(danswers_df)}')

        # return jsonify({'msg':'OK','shortAnswer':'Test short answer','sentence_embedding':str(sentence_embeddings)})

        detailed_answers_found = False
        detailed_answer_length = 0
        if len(danswer) > 0:
            detailed_answers_found = True
            for ans in danswer:
                # print(f'detailed answers : {ans}')
                if ans.strip()[-1] == '?':
                    pass
                else:
                    detailed_answers +=  ans + '\n'
                    detailed_answer_length += 2
        answer, question_lemma_, question_pos_ = query.short_answer_question(question,None, nlp, svo_df,danswer_index,danswers_df, G, pos, edge_labels, check_cause)
        short_answers_found = False
        short_answer_length = 0
        if len(answer) > 0:
            short_answers_found = True
            for ans in answer:
                short_answers = '\n'.join(answer)
                short_answer_length += 1
        no_answer_found = False
        qanswer = ''
        if not short_answers_found:
            if not detailed_answers_found:
                no_answer_found = True
                qanswer = 'No answer found.'


        
        # return jsonify({'msg':'No data extracted'})
        return jsonify({'msg':'OK','shortAnswer':str(short_answers),'detailedAnswer':str(detailed_answers)})