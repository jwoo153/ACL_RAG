from pathlib import Path
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

import time

import nest_asyncio
nest_asyncio.apply()



from utils import get_doc_tools, set_openai_api_key

from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner




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


def calculate_average(lst):
    return sum(lst) / len(lst) if lst else 0


def main():
    #### Modify the code as needed for your purposes
    set_openai_api_key(
        #Put your OpenAI API key here
    )

    llm = OpenAI(model="gpt-4-turbo", temperature=0)
    dir_path = str(Path(__file__).parent.absolute())
    data_dir = f'{dir_path}/data'


    papers = [
        f'{data_dir}/aclcpg.pdf',
    ]

    paper_to_tools_dict = {}
    for paper in papers:
        print(f"Getting tools for paper: {paper}")
        vector_tool, summary_tool, retriever_tool = get_doc_tools(paper, Path(paper).stem)
        paper_to_tools_dict[paper] = [vector_tool, summary_tool, retriever_tool]

    initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

    print(len(initial_tools))







    agent_worker = FunctionCallingAgentWorker.from_tools(
        initial_tools, 
        llm=llm, 
        verbose=False
    )
    agent = AgentRunner(agent_worker)

    
    file_path = f'{data_dir}/ACL_Questions.json'

    # Extract all questions and answers
    questions, answers = extract_questions(file_path)
    bleu_list = []
    rouge_1_list = []
    rouge_2_list = []
    rouge_l_list = []
    meteor_list = []
    counter = 0

    with open('log.txt', 'w') as f:
        pass


    # Query each question using the connector
    for question, reference_answer in zip(questions, answers):
        
        
        counter +=1  
        try:
            print(f'QUESTION {counter}:', question)

            q = "Answer in a single-clause sentence or a few-word phrase. " + question

            out = str(agent.query(q))
            hypothesis_answer = out

            with open('log.txt', 'a') as f:
                f.write('QUESTION: ' + question + '\n')
                f.write('ANSWER: ' + hypothesis_answer + '\n')
                f.write('REFERENCE ANSWER: ' + reference_answer + '\n')
                f.write('-----------------------------------------------\n')



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

        except Exception as e:
            print(f'QUESTION {counter} is messed up')
            print(e)


        print('waiting 10 seconds...')
        time.sleep(10)


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



if __name__ == "__main__":
    main()

