from api_connector import api_connector

import json
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
connector = api_connector(
    api_key=''
)

connector.set_workspace('Llama70b')#or any other model

import json

def extract_questions(file_path):
    # Load the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Extract all questions and answers
    questions = [item['question'] for item in data]
    answers = [item['answer'] for item in data]
    
    return questions, answers

def calculate_metrics(reference, hypothesis):
    # Tokenize reference and hypothesis
    reference_tokens = nltk.word_tokenize(reference)
    hypothesis_tokens = nltk.word_tokenize(hypothesis)

    # Calculate BLEU score
    bleu = sentence_bleu([reference_tokens], hypothesis_tokens)

    # Calculate ROUGE score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(' '.join(reference_tokens), ' '.join(hypothesis_tokens))
    rouge_1 = rouge_scores['rouge1'].fmeasure
    rouge_2 = rouge_scores['rouge2'].fmeasure
    rouge_l = rouge_scores['rougeL'].fmeasure

    # Calculate METEOR score
    meteor = meteor_score([reference_tokens], hypothesis_tokens)

    return bleu, rouge_1, rouge_2, rouge_l, meteor
# Define the path to the JSON file
file_path = ''

# Extract all questions and answers
questions, answers = extract_questions(file_path)
bleu_list = []
rouge_1_list = []
rouge_2_list = []
rouge_l_list = []
meteor_list = []
counter = 0
# Query each question using the connector
for question, reference_answer in zip(questions, answers):
    counter +=1
    print(f'QUESTION {counter}:', question)
    out = connector.query(question)
    hypothesis_answer = out  # Assuming `connector.query(question)` returns the answer
    
    # Calculate metrics
    bleu, rouge_1, rouge_2, rouge_l, meteor = calculate_metrics(reference_answer, hypothesis_answer)
    bleu_list.append(bleu)
    rouge_1_list.append(rouge_1)
    rouge_2_list.append(rouge_2)
    rouge_l_list.append(rouge_l)
    meteor_list.append(meteor)
    # Print the results
    print('ANSWER:', hypothesis_answer)
    print('REFERENCE ANSWER:', reference_answer)
    print('BLEU:', bleu)
    print('ROUGE-1:', rouge_1)
    print('ROUGE-2:', rouge_2)
    print('ROUGE-L:', rouge_l)
    print('METEOR:', meteor)
    print('-----------------------------------------------')


    # Calculate averages
average_bleu = calculate_average(bleu_list)
average_rouge_1 = calculate_average(rouge_1_list)
average_rouge_2 = calculate_average(rouge_2_list)
average_rouge_l = calculate_average(rouge_l_list)
average_meteor = calculate_average(meteor_list)

# Print the averages
print(f"Average BLEU: {average_bleu}")
print(f"Average ROUGE-1: {average_rouge_1}")
print(f"Average ROUGE-2: {average_rouge_2}")
print(f"Average ROUGE-L: {average_rouge_l}")
print(f"Average METEOR: {average_meteor}")