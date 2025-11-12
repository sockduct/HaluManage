import os
import logging
from typing import Dict
import litellm
from dotenv import load_dotenv
import traceback

# --- 1. Initialization ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 2. Define the Task (Query, Answer, Model, and Context URLs) ---

# You can swap out the question and corresponding links/answer here.
# A multi-hop, challenging question, as defined by the FRAMES paper methodology.
# Que. No. 1 - Geting FALSE
'''FRAME_QUERY = (
    "If my future wife has the same first name as the 15th first lady of the United States' mother and her surname"
    "is the same as the second assassinated president's mother's maiden name, what is my future wife's name?"
)'''

# Que. No. 3 - 
'''FRAME_QUERY = (
    "How many years earlier would Punxsutawney Phil have to be canonically alive to have made a Groundhog Day prediction in the same state as the US capitol?"
)'''

# Que. No. 9 - 
FRAME_QUERY = (
    "The Pope born Pietro Barbo ended a long-running war two years after his papacy began, which famous conflict, immortalized in tapestry took place 400 years earlier?"
)

# Que. No. 13 - TRUE
'''FRAME_QUERY = (
    "One of Barbara Kingsolver's best known novels is about an American missionary family which moves to Africa."
    "At the time, the country they move to was a Belgian colony. Which year did it become independent?"
)'''

# Que. No. 75 - Getting FALSE - https://en.wikipedia.org/wiki/Milk_and_Honey_(poetry_collection [Content from en.wikipedia.org: Error fetching content: 404 Client Error: Not Found for url: https://en.wikipedia.org/wiki/Milk_and_Honey_(poetry_collection])
'''FRAME_QUERY = (
    "How old would the 1975 winner of the Lenore Marshall Poetry Prize have been if they were still alive on the date when Rupi Kaur released her book titled, \"Milk and Honey\"?"  
)'''

# Que. No. 104 - TRUE
'''FRAME_QUERY = (
    " \"The Terminator\" was released on October 26th exactly how many years after the famous gunfight at the O.K. Corral occurred?"
)'''


#GROUND_TRUTH_ANSWER = "Jane Ballou"
#GROUND_TRUTH_ANSWER = "87"
GROUND_TRUTH_ANSWER = "The Battle of Hastings."
#GROUND_TRUTH_ANSWER = "1960"
#GROUND_TRUTH_ANSWER = "90"
#GROUND_TRUTH_ANSWER = "103"

# The list of URLs to be retrieved by the 'readurls' plugin.

'''WIKI_LINKS = [
        'https://en.wikipedia.org/wiki/President_of_the_United_States', 
        'https://en.wikipedia.org/wiki/James_Buchanan', 
        'https://en.wikipedia.org/wiki/Harriet_Lane', 
        'https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States_who_died_in_office', 
        'https://en.wikipedia.org/wiki/James_A._Garfield'
]'''

'''WIKI_LINKS = [
        'https://en.wikipedia.org/wiki/Punxsutawney_Phil',
        'https://en.wikipedia.org/wiki/United_States_Capitol',
]'''

WIKI_LINKS = [
        'https://en.wikipedia.org/wiki/Barbara_Kingsolver', 
        'https://en.wikipedia.org/wiki/The_Poisonwood_Bible', 
        'https://en.wikipedia.org/wiki/Belgian_Congo'
]

'''WIKI_LINKS = [
        'https://en.wikipedia.org/wiki/Barbara_Kingsolver', 
        'https://en.wikipedia.org/wiki/The_Poisonwood_Bible', 
        'https://en.wikipedia.org/wiki/Belgian_Congo'
]'''

'''WIKI_LINKS = [
        'https://en.wikipedia.org/wiki/List_of_winners_of_the_Lenore_Marshall_Poetry_Prize', 
        'https://en.wikipedia.org/wiki/Cid_Corman', 
        'https://en.wikipedia.org/wiki/Milk_and_Honey_(poetry_collection)'
]'''

'''WIKI_LINKS = [
        'https://en.wikipedia.org/wiki/Gunfight_at_the_O.K._Corral', 
        'https://en.wikipedia.org/wiki/The_Terminator'
]'''



MODEL_NAME = "gemini/gemini-2.5-flash-lite"


# --- 3. Helper Functions for API Calls ---

def get_llm_response(prompt: str, model: str) -> str:
    """
    Sends the request to the HaluManage proxy with the RAG approach.
    Uses the proxy's 'readurls&memory' plugin by building the model string.
    Returns the assistant content or "" on failure.
    """
    try:
        logging.info(f"Invoking HaluManage proxy with model: {model} and RAG approach.")
        # The model name is prefixed with the approach for the HaluManage proxy.
        response = litellm.completion(
            model=f"litellm_proxy/{model}",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt}
            ],
            # A dummy api_key is needed for the proxy, which handles the actual provider keys.
            api_key="halumanage",
            base_url="http://localhost:8000/v1", # Point to your local HaluManage server
            extra_body={"halumanage_approach": "readurls&memory"},
            max_tokens=1500,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error getting LLM response: {e}")
        logging.debug(traceback.format_exc())
        return ""

def evaluate_response(question: str, llm_response: str, ground_truth: str, model: str) -> Dict[str, str]:
    """
    Uses the same LLM as a judge to evaluate if the model's response is correct.
    This makes a direct call to the model (bypasses readurls plugin).
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
        logging.info("Invoking LLM as an evaluator (direct call).")
        response = litellm.completion(
            model=f"litellm_proxy/{model}",  # direct model call to the underlying model
            messages=[
                {"role": "system", "content": "You are an expert evaluator."},
                {"role": "user", "content": evaluation_prompt}
            ],
            # A dummy api_key is needed for the proxy.
            api_key="halumanage",
            base_url="http://localhost:8000/v1", # Point to your local HaluManage server
            max_tokens=500,
            temperature=0.1
        )

        # Safely access response content, works for both object and dict responses
        evaluation_text = ""
        if response and hasattr(response, 'choices') and response.choices:
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                evaluation_text = response.choices[0].message.content.strip()

        if not evaluation_text:
            logging.error("Could not extract evaluation text from the response.")
            return {"decision": "ERROR", "explanation": "Empty or invalid response from evaluator."}

        lines = evaluation_text.split('\n')
        decision = "FALSE"
        explanation = ""
        for line in lines:
            if line.lower().startswith("decision:"):
                decision = line.split(":", 1)[1].strip().upper()
            elif line.lower().startswith("explanation:"):
                explanation = line.partition(":")[2].strip()
        return {"decision": decision, "explanation": explanation}
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        logging.debug(traceback.format_exc())
        return {"decision": "ERROR", "explanation": str(e)}

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    print(f"--- FRAME Query ---\n{FRAME_QUERY}\n")
    prompt = f"{FRAME_QUERY}\n\nRelevant URLs:\n" + "\n".join(WIKI_LINKS)
    llm_response = get_llm_response(prompt, MODEL_NAME)
    print(f"\n[HaluManage RAG Response for {MODEL_NAME}]:\n{llm_response}")
    evaluation = evaluate_response(FRAME_QUERY, llm_response, GROUND_TRUTH_ANSWER, MODEL_NAME)
    print(f"\n[Evaluation Decision]: {evaluation.get('decision')}")
    print(f"[Evaluation Explanation]: {evaluation.get('explanation')}")