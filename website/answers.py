from .nlp_kng import cleaner, kg_generator, query, scrapper, svo_extractor
from .nlp_kng import config
import neuralcoref
import networkx as nx
import spacy

@config.timer
def print_answers(quest, unique_statements, question_lemma, question_pos):
    if len(unique_statements) > 0:
        answer_facts = []
        no_of_facts = len(unique_statements)
        n = 1
        for statement in unique_statements:
            answer_facts.append(f'\t{statement}')
            n += 1
        if len(answer_facts) >0 :
            print('\n')
            print(f'Question: {quest}')
            print("Answers:")
            for fact in answer_facts:
                print(fact)
        else:
            print('\n')
            print(f'Question: {quest}')
            print(f'Question Lemma: {question_lemma}\tQuestion Pos: {question_pos}')
            print(f"Answer: No answer derived from given Knowledge Graph.\n")
    else:
        print('\n')
        print(f'Question: {quest}')
        print(f'Question Lemma: {question_lemma}\tQuestion Pos: {question_pos}')
        print(f"Answer: No answer derived from given Knowledge Graph.\n")


@config.timer
def return_answers(quest, unique_statements, question_lemma, question_pos):
    if len(unique_statements) > 0:
        answer_facts = []
        no_of_facts = len(unique_statements)
        n = 1
        for statement in unique_statements:
            answer_facts.append(f'\t{statement}')
            # n += 1
        return answer_facts
    return None



# if __name__ == "__main__":
    
#     '''defining the pipeline'''
#     # nlp = spacy.load('en_core_web_sm',n_threads=LEMMATIZER_N_THREADS,  batch_size=LEMMATIZER_BATCH_SIZE)
#     nlp = spacy.load('en_core_web_sm')
#     neuralcoref.add_to_pipe(nlp)
    
#     text = scrapper.scrape_text(config.WEB_URL)
    
#     text_df, clean_text = cleaner.clean_data(text)
#     svo_df, text_doc = svo_extractor.svo(clean_text, nlp)
#     kng_G = kg_generator.plot_kg(svo_df)


#     question = "what is the impact of prices increase?"
#     answers, question_lemma_, question_pos_ = query.short_answer_question(question,text_doc, nlp, svo_df)
#     print_answers(question, answers, question_lemma_, question_pos_)
    
#     answers, question_lemma_, question_pos_ = query.detailed_answer_question(question,text_doc, nlp, svo_df)
#     print_answers(question, answers, question_lemma_, question_pos_)


    # question = "what causes Inflation?"
    # answer_question(question, text_doc)
    # question = "what policy can increase?"
    # answer_question(question, text_doc)