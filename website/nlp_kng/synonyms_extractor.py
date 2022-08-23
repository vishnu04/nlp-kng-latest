## Code to download gensim text8 model as pickle object
# import gensim.downloader  as api
# from gensim.models import Word2Vec
# import pickle
# import os 

# dataset = api.load('text8')
# model = Word2Vec(dataset)

# current_path = os.path.abspath(os.path.dirname(__file__))
# model_path = os.path.join(current_path, '../models/')
# pickle.dump(model,open(model_path+"gensim_text8_model","wb"), pickle.HIGHEST_PROTOCOL)

# Code to load the downloaded gensim text8 model from pickle object

import pickle
import os
import gensim
from smart_open import open
current_path = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(current_path, '../models/')
model = pickle.load(open(model_path+"gensim_text8_model","rb"))


def get_synonyms(verbs, svo_df):
    syn_verbs = []
    ## for question verbs
    for verb in verbs:
        if verb in model.wv.key_to_index:
            syn_verbs.append(verb)
            for key, value in model.wv.most_similar(verb):
                if round(value) >= 0.5:
                    syn_verbs.append(key)
    print(syn_verbs)
    new_syn_verbs = []
    for syn_verb in syn_verbs:
        if syn_verb in model.wv.key_to_index:
            for verbs in svo_df.verb:
                for verb in verbs.split(" "):
                    if verb in model.wv.key_to_index:
                        if model.wv.similarity(syn_verb, verb) >= 0.5:
                            new_syn_verbs.append(verb)
    new_syn_verbs = list(set(new_syn_verbs))
    print(new_syn_verbs)
    return new_syn_verbs
