import os
import logging
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI

# --- 1. Initialization ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure the OpenAI client to point to your local HaluManage proxy.
# The proxy will use the OPENAI_API_KEY from its own environment.
client = OpenAI(api_key="halumanage", base_url="http://localhost:8000/v1")

# --- 2. Define the Task (Query, Answer, Model, and Context URLs) ---

# Que. No. 9 - 
FRAME_QUERY = (
    "The Pope born Pietro Barbo ended a long-running war two years after his papacy began, which famous conflict, immortalized in tapestry took place 400 years earlier?"
)

# Que. No. 75 - https://en.wikipedia.org/wiki/Milk_and_Honey_(poetry_collection [Content from en.wikipedia.org: Error fetching content: 404 Client Error: Not Found for url: https://en.wikipedia.org/wiki/Milk_and_Honey_(poetry_collection])
'''FRAME_QUERY = (
    "How old would the 1975 winner of the Lenore Marshall Poetry Prize have been if they were still alive on the date when Rupi Kaur released her book titled, \"Milk and Honey\"?"  
)'''

# Que. No. 104 - 
'''FRAME_QUERY = (
    " \"The Terminator\" was released on October 26th exactly how many years after the famous gunfight at the O.K. Corral occurred?"
)'''

GROUND_TRUTH_ANSWER = "The Battle of Hastings."
#GROUND_TRUTH_ANSWER = "90"
#GROUND_TRUTH_ANSWER = "103"

WIKI_LINKS = [
        'https://en.wikipedia.org/wiki/Barbara_Kingsolver', 
        'https://en.wikipedia.org/wiki/The_Poisonwood_Bible', 
        'https://en.wikipedia.org/wiki/Belgian_Congo'
]

'''WIKI_LINKS = [
        'https://en.wikipedia.org/wiki/List_of_winners_of_the_Lenore_Marshall_Poetry_Prize', 
        'https://en.wikipedia.org/wiki/Cid_Corman', 
        'https://en.wikipedia.org/wiki/Milk_and_Honey_(poetry_collection)'
]'''

'''WIKI_LINKS = [
    'https://en.wikipedia.org/wiki/Gunfight_at_the_O.K._Corral', 
    'https://en.wikipedia.org/wiki/The_Terminator'
]'''

MODEL_NAME = "gpt-4o-mini"

# --- 3. Helper Functions for API Calls ---

def get_llm_response(prompt: str, model: str) -> str:
    """
    Sends the request to the HaluManage proxy with the RAG approach.
    The `extra_body` parameter is the key instruction for the proxy.
    """
    try:
        logging.info(f"Invoking HaluManage proxy with model: {model} and RAG approach.")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.7,
            extra_body={"halumanage_approach": "readurls&memory"}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error getting LLM response: {e}")
        return ""

def evaluate_response(question: str, llm_response: str, ground_truth: str, model: str) -> Dict[str, str]:
    """
    Uses the same LLM as a judge to evaluate if the model's response is correct.
    """
    evaluation_prompt = f"""===Task===
        I need your help in evaluating an answer provided by an LLM against a ground
        truth answer. Your task is to determine if the ground truth answer is present in the LLM's
        response. Please analyze the provided data and make a decision.
        ===Instructions===
        1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer".
        2. Consider the substance of the answers - look for equivalent information or correct answers.
        Do not focus on exact wording unless the exact wording is crucial to the meaning.
        3. Your final decision should be based on whether the meaning and the vital facts of the
        "Ground Truth Answer" are present in the "Predicted Answer:"
        ===Input Data===
        - Question: {question}
        - Predicted Answer: {llm_response}
        - Ground Truth Answer: {ground_truth}
        ===Output Format===
        Provide your final evaluation in the following format:
        "Explanation:" (How you made the decision?)
        "Decision:" ("TRUE" or "FALSE" )
        Please proceed with the evaluation."""

    try:
        logging.info("Invoking LLM as an evaluator.")
        evaluation_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert evaluator."},
                {"role": "user", "content": evaluation_prompt}
            ],
            max_tokens=300,
            n=1,
            stop=None,
            temperature=0.3,
        )
        evaluation_text = evaluation_response.choices[0].message.content.strip()
        
        lines = evaluation_text.split('\n')
        decision = "FALSE"
        explanation = ""
        for line in lines:
            if line.lower().startswith("decision:"):
                decision = line.split(":", 1)[1].strip().upper()
            elif line.lower().startswith("explanation:"):
                explanation = line.partition(":")[2].strip()
        return {"decision": decision, "explanation": explanation}
    except Exception as error:
        logging.error(f"An error occurred during evaluation: {error}")
        return {"decision": "ERROR", "explanation": str(error)}

# --- 4. Main Execution Block ---
# Note: Removed URL to avoid potential licensing concerns.

if __name__ == "__main__":
    print(f"--- FRAME Query ---\n{FRAME_QUERY}\n")
    
    # The prompt includes the URLs, which the 'readurls' plugin will detect and process.
    prompt = f"{FRAME_QUERY}\n\nRelevant URLs:\n" + "\n".join(WIKI_LINKS)
    
    llm_response = get_llm_response(prompt, MODEL_NAME)
    print(f"\n[HaluManage RAG Response for {MODEL_NAME}]:\n{llm_response}")
    
    evaluation = evaluate_response(FRAME_QUERY, llm_response, GROUND_TRUTH_ANSWER, MODEL_NAME)
    print(f"\n[Evaluation Decision]: {evaluation.get('decision')}")
    print(f"[Evaluation Explanation]: {evaluation.get('explanation')}")