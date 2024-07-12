import os
from dotenv import load_dotenv
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nest_asyncio
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq.chat_models import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from langchain.chains.combine_documents import stuff
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from IPython.core.display import Markdown
import json
import re
from langchain_core.runnables import (
    RunnableParallel,
    RunnableBranch,
    RunnablePassthrough,
)
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter
import asyncio


from PyPDF2 import PdfFileReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the PDF document
pdf_path = 'yourpath/aclcpg.pdf'
pdf_loader = PyPDFLoader(pdf_path)
pdf_docs = pdf_loader.load()

# Split the document into chunks
pdf_chunks = text_splitter.split_documents(pdf_docs)
pdf_chunks

# Swap these out as necessary 
embedding_function = HuggingFaceEmbeddings(show_progress=True, multi_process=True)

vector_store = Chroma.from_documents(documents=pdf_chunks, embedding=embedding_function)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = #input whatever llm you want, like llama3 opensource 


class AnswerGrader(BaseModel):
    "Binary score for an answer check based on a query."

    grade: Literal["yes", "no"] = Field(
        ...,
        description="'yes' if the provided answer is an actual answer to the query otherwise 'no'",
    )


answer_grader_system_prompt_template = (
    "You are a grader assessing whether a provided answer is in fact an answer to the given orthopedic related query.\n"
    "The answers should be sufficiently related to the query, although the answer to the query will be brief \n"
    "If the provided answer does not answer the query give a score of 'no' otherwise give 'yes'\n"
    "Just give the grade in json with 'grade' as a key and a binary value of 'yes' or 'no' without additional explanation"
)

answer_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_grader_system_prompt_template),
        ("human", "query: {query}\n\nanswer: {response}"),
    ]
)


answer_grader_chain = answer_grader_prompt | llm.with_structured_output(
    AnswerGrader, method="json_mode"
)

## Example usage of a possible query, response that you would get self-reflection
query = "What should a focused musculoskeletal exam include when assessing for an ACL injury?"
context = retriever.get_relevant_documents(query)
response = """Based on the provided context, a focused musculoskeletal exam when assessing for an ACL injury should include a relevant history, which at a minimum should include:

Mechanism and date of injury
History of hearing/feeling a popping sensation
Ability to bear weight
Ability to return to play
History of mechanical symptoms of locking or catching
Localization of pain if possible
Any history of prior knee injuries"""

response = answer_grader_chain.invoke({"response": response, "query": query})
##############################################################################################
#essentially this is where you make the agentic logic
#you use the workflow to add nodes and edges representing flow of logic, then compile
def rag_node(state: dict):
    query = state["query"]
    documents = state["documents"]

    generation = rag_chain.invoke({"query": query, "context": documents})
    return {"generation": generation}

def retrieve_node(state: dict) -> dict[str, list[Document] | str]:
    """
    Retrieve relevent documents from the vectorstore

    query: str

    return list[Document]
    """
    query = state["query"]
    documents = retriever.invoke(input=query)
    return {"documents": documents}
def fallback_node(state: dict):
    """
    Fallback to this node when there is no tool call
    """
    query = state["query"]
    chat_history = state["chat_history"]
    generation = fallback_chain.invoke({"query": query, "chat_history": chat_history})
    return {"generation": generation}
def question_router_node(state: dict):
    query = state["query"]
    try:
        response = question_router.invoke({"query": query})
    except Exception:
        return "llm_fallback"

    if "tool_calls" not in response.additional_kwargs:
        print("---No tool called---")
        return "llm_fallback"

    if len(response.additional_kwargs["tool_calls"]) == 0:
        raise "Router could not decide route!"

    route = response.additional_kwargs["tool_calls"][0]["function"]["name"]
    if route == "VectorStore":
        print("---Routing to VectorStore---")
        return "VectorStore"
    
def relevance_check(state: dict):
    llm_response = state["generation"]
    answer_relevance_grade = answer_grader_chain.invoke(
            {"response": llm_response, "query": query}
        )
        if answer_relevance_grade.grade == "yes":
            print("---Answer is relevant to question---\n")
            return "useful"
        else:
            print("---Answer is not relevant to question---")
            print(llm_response)
            return "not useful"
    
class AgentState(TypedDict):
    """The dictionary keeps track of the data required by the various nodes in the graph"""

    query: str
    chat_history:list[BaseMessage]
    generation: str
    documents: list[Document]
workflow = StateGraph(AgentState)
workflow.add_node("VectorStore", retrieve_node)
workflow.add_node("rag", rag_node)
workflow.add_node("filter_docs", filter_documents_node)
workflow.add_node("fallback", fallback_node)
#add edges for flow of logic
app = workflow.compile(debug=False)




#run the workflow after instantiating the metrics
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Ensure you have the necessary nltk data
# nltk.download('wordnet')
# nltk.download('punkt')

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


# Extract all questions and answers
questions, answers = extract_questions(file_path)
bleu_list = []
rouge_1_list = []
rouge_2_list = []
rouge_l_list = []
meteor_list = []
counter = 0

errors = []
# Query each question using the connector
for question, reference_answer in zip(questions, answers):
    
    counter +=1  
    try:
        print(f'QUESTION {counter}:', question)
        response = app.invoke({"query": question, "chat_history": []})
        hypothesis_answer = response["generation"]
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
        #if counter % 2==0:
        time.sleep(20)
    except Exception as e:
        print(f'QUESTION {counter} is messed up')
        print(e)
        errors.append(counter)