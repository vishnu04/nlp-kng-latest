from venv import create
from flask import Blueprint, render_template, request, flash, redirect, url_for, send_file, session
from .nlp_kng import scrapper, cleaner, svo_extractor, kg_generator, query, config, create_temp_dir, cleantmp
import spacy
# import neuralcoref
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

def save_nlp_to_disk(text_doc):
    try:
        tmpdir = create_temp_dir.createTempDir()
        text_nlp = tmpdir+'/spacy_nlp.nlp'
        text_doc.to_disk(text_nlp)
        text_vocab = tmpdir+'/spacy_vocab.voc'
        text_doc.vocab.to_disk(text_vocab)
        print(f'tmpdir : {tmpdir}')
        return tmpdir
    except Exception as e:
        print(f'Unable to save NLP to disk {e}')
        return None

actions = Blueprint('actions',__name__)
## Cleaning tmp folder
cleantmp.main()

nlp = spacy.load(config.BASE_MODEL)
# neuralcoref.add_to_pipe(nlp)
# nlp.add_pipe('sentencizer')
# nlp.add_pipe(nlp.create_pipe('sentencizer'))

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
        text = request.form.get('cleantext')
        print(f'length of clean text: {len(text)}')


        '''defining the pipeline'''

        
        text = text.lower()
        text_doc = nlp(text)
        n_word_tokens = len(word_tokenize(text))

        svo_df, text_doc = svo_extractor.svo(text_doc,nlp)
        print(f'n_word_tokens : {n_word_tokens}')

        ## Saving NLP
        tmpdir = save_nlp_to_disk(text_doc)
    

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
    print(f'calling queryQuestion {request.method}')
    if request.method == 'POST':
        web_url = request.form.get('weburl')
        figdata_png = session.get('image_base64')
        text = request.form.get('cleantext')
        triplets_found = request.form.get('triplets_found')
        headings = request.form.get('headings')
        data_tuple = request.form.get('data')
        
        print(create_temp_dir.get_temp_dir())
        ## Loading NLP
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

        print(f'queryQuestion: text_doc: {type(text_doc)}')

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
        short_answers_found = False
        short_answer_length = 0
        if len(answer) > 0:
            short_answers_found = True
            for ans in answer:
                short_answers = '\n'.join(answer)
                short_answer_length += 1
        answer, question_lemma_, question_pos_ = query.detailed_answer_question(question,text_doc, nlp, svo_df)
        detailed_answers_found = False
        detailed_answer_length = 0
        if len(answer) > 0:
            detailed_answers_found = True
            for ans in answer:
                detailed_answers = '\n'.join(answer)
                detailed_answer_length += 2
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

