from flask import Blueprint, render_template, request, flash, redirect, url_for, send_file, session
from .nlp_kng import scrapper, cleaner, svo_extractor, kg_generator, query
# from .nlp_kng import * 
import spacy
import neuralcoref
from io import BytesIO
import networkx as nx
import matplotlib.pyplot as plt
import base64
import json 
from . import answers
import pandas as pd 
from PIL import Image
import io 
import os 


actions = Blueprint('actions',__name__)

print('calling actions')

@actions.route('/',methods=['GET'])
def home():
    print(f'calling home - {request.method}')
    return render_template('home.html')

@actions.route('/scrape', methods=['POST'])
def scrape():
    print(f'calling scrape {request.method}')
    if request.method == 'POST':
        web_url = request.form.get("weburl")
        print(web_url)
        if len(web_url) <1 :
            flash('URL is too short! Please enter correct URL. e.g: https://textacy.readthedocs.io/', category='error')
            return render_template('home.html')
        else:
            try:
                text = scrapper.scrape_text(web_url)
                text_df,text = cleaner.clean_data(text)
                # print(f'length of cleaned text scrape : {len(text)}')
                if len(text) > 1:
                    print(f'length of text scraped:{len(text)}')
                    return render_template('home.html', weburl = web_url, cleantext = text, display_svo = False)
            except:
                flash('URL entered cannot be scraped !. Please enter correct URL. e.g: https://textacy.readthedocs.io/', category='error')
                return render_template('home.html', weburl = web_url)


@actions.route('/extract', methods=['POST'])
def extract():
    print(f'Extracting triplets {request.method}')
    if request.method == 'GET':
        return render_template('home.html')
    if request.method == 'POST':
        web_url = request.form.get('weburl')
        nlp = spacy.load('en_core_web_sm')
        neuralcoref.add_to_pipe(nlp)
        # print(f'web_url:{web_url}')
        # text = request.form.get('cleantextarea')
        text = request.form.get('cleantext')
        print(f'length of clean text: {len(text)}')
        # print(text)
        '''defining the pipeline'''
        # nlp = spacy.load('en_core_web_sm',n_threads=LEMMATIZER_N_THREADS,  batch_size=LEMMATIZER_BATCH_SIZE)
        svo_df, text_doc = svo_extractor.svo(text,nlp)
        # print(f'type of text_doc ---> {type(text_doc)}')
        # print(len(svo_df))
        # print(svo_df.head(7))
        triplets_found = False
        headings = svo_df.columns
        data_tuple = []
        if len(svo_df) > 1:
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
                        
            session['image_base64'] = figdata_png
            # print('printing --- text_doc')
            # print(text_doc)
            session['text_doc'] = str(text_doc)
            # session['nlp'] = nlp
            # session['headings'] = headings
            # session['data_tuple'] = data_tuple

            session['svo_df'] = svo_df.to_json()

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
                                    display_svo = True)
        else:
            
            session['svo_df'] = svo_df.to_json()
            # full_fileName = '*/static/images/no-image-available.jpeg'
            cwd = os.getcwd()
            img = Image.open(str(cwd)+"/website/static/images/no-image-available.jpeg")
            # img = Image.open("/static/images/no-image-available.jpeg")
            buf = BytesIO()
            img.save(buf, format = 'jpeg')
            img.close()
            buf.seek(0)
            encoded_string = base64.b64encode(buf.getvalue()).decode('ascii')
            # encoded_string = base64.b64encode(img).decode('ascii')
            session['image_base64'] = encoded_string
            # print(session.get('image_base64'))
            # print(text_doc)
            session['text_doc'] = str(text_doc)
            # session['nlp'] = nlp
            # session['headings'] = headings
            # session['data_tuple'] = data_tuple
            return render_template('home.html',image_base64=encoded_string, weburl = web_url, cleantext = text, 
                                    triplets_found = triplets_found, headings = headings, data = data_tuple, 
                                    svo_df = svo_df, 
                                    svo_table = [svo_df.to_html()], titles=[''],
                                    text_doc = text_doc,
                                    nlp = nlp,
                                    display_svo = True)

@actions.route('/queryQuestion', methods=['POST'])
def queryQuestion():
    print(f'calling queryQuestion {request.method}')
    if request.method == 'POST':
        web_url = request.form.get('weburl')
        figdata_png = session.get('image_base64')
        text = request.form.get('cleantext')
        triplets_found = request.form.get('triplets_found')
        headings = request.form.get('headings')
        data_tuple = request.form.get('data')
        # text_doc = request.form.get('text_doc')
        # nlp = request.form.get('nlp')
        
        # headings = session.get('headings')
        # data_tuple = session.get('data_tuple')
        text_doc = session.get('text_doc')
        # nlp = session.get('nlp')

        nlp = spacy.load('en_core_web_sm')
        neuralcoref.add_to_pipe(nlp)
        text_doc = nlp(text_doc)

        svo_df = pd.read_json(session.get('svo_df'), dtype=False)
        headings = svo_df.columns
        data_tuple = []
        if len(svo_df) > 1:
            triplets_found = True
            for index, row in svo_df.iterrows():
                data_tuple.append((row.subject, row.verb, row.object))
            data_tuple = tuple(data_tuple)

        question = request.form.get('question')
        print(f'Question : {question}')
        
        short_answers = ''
        detailed_answers = ''
        answer, question_lemma_, question_pos_ = query.short_answer_question(question,text_doc, nlp, svo_df)
        # answers.print_answers(question, answer, question_lemma_, question_pos_)
        short_answers_found = False
        short_answer_length = 0
        if len(answer) > 0:
            short_answers_found = True
            for ans in answer:
                # short_answers = short_answers + '\n' + str(answer).replace('{','').replace('}','').replace
                short_answers = '\n'.join(answer)
                short_answer_length += 1
                # print(answer)
            # short_answers = short_answers.replace('\n','',1)
        answer, question_lemma_, question_pos_ = query.detailed_answer_question(question,text_doc, nlp, svo_df)
        # answers.print_answers(question, answer, question_lemma_, question_pos_)
        detailed_answers_found = False
        detailed_answer_length = 0
        if len(answer) > 0:
            detailed_answers_found = True
            for ans in answer:
                # detailed_answers = detailed_answers + '\n' + str(answer)
                # print(f'ans ------> {ans}')
                # ans1 = ''
                # for a in ans:
                #     ans1 = ''.join(str(a))
                detailed_answers = '\n'.join(answer)
                detailed_answer_length += 2
                # print(answer)
            # detailed_answers = detailed_answers.replace('\n','',1)
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