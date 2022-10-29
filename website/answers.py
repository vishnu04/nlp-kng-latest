from .nlp_kng import config

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
            n += 1
        return answer_facts
    return None
