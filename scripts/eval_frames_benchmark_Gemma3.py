import argparse
import json
import os
import time
from typing import List, Dict
import logging

from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
import litellm
from typing import Tuple

load_dotenv()  # Load environment variables from .env file

sleep_interval = 5  # seconds
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the model name for LiteLLM (Google Cloud AI Platform)
MODEL_NAME = "gemini/gemma-3-27b-it"  # The model identifier for Gemma 3 27b on Google Cloud AI Platform
#MODEL_NAME = "vertex_ai/gemma2-27b-it"  # Correct model identifier for Gemma 2 27b on Vertex AI

# Get Google Cloud API key from environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# For Vertex AI LiteLLM calls, get project and location from environment variables
#VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT")
#VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION")

# Verify that the API key is set
if not GEMINI_API_KEY:
    logging.error("GEMINI API key not found. Please set the GEMINI_API_KEY environment variable.")
    exit() 
# Verify that the project and location are set
'''if not VERTEX_PROJECT or not VERTEX_LOCATION:
    logging.error("VERTEX_PROJECT and VERTEX_LOCATION environment variables must be set for Vertex AI.")
    exit() '''
# Function to check if the Google Cloud token is valid by making a simple API call
def is_vertex_ai_configured():
    try:
        litellm.completion(
            model=MODEL_NAME,  # Use a basic model to test the token
            messages=[{"role": "user", "content": "test"}],
            #vertex_project=VERTEX_PROJECT,
            #vertex_location=VERTEX_LOCATION,
            timeout=10  # Add a timeout to prevent indefinite hanging
        )
        return True
    except Exception as e:
        logging.error(f"Vertex AI configuration check failed: {e}")
        logging.error("Ensure you have run 'gcloud auth application-default login' and set the correct project/location.")
        return False

# Validate the Vertex AI configuration
if not is_vertex_ai_configured():
    exit()

def load_existing_results(filename: str) -> List[Dict]:
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_result(filename: str, result: Dict):
    results = load_existing_results(filename)
    results.append(result)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def get_last_processed_index(results: List[Dict]) -> int:
    if not results:
        return -1
    return max(int(r.get('index', -1)) for r in results)

def count_tokens(text: str, model_name: str) -> int:
    """Counts the number of tokens in a given text using litellm's token counter."""
    try:
        # Use litellm's built-in token counter
        return litellm.token_counter(model=model_name, text=text)
    except Exception as e:
        logging.warning(f"Could not count tokens for model {model_name}: {e}. Falling back to character count / 4.")
        # Fallback to a rough estimate if litellm fails
        return len(text) // 4

def generate_llm_prompt(prompt: str, wiki_links: List[str]) -> Tuple[str, int]:
    """
    Generates the LLM prompt and counts RAG tokens.
    """
    rag_content = "\n".join(wiki_links)
    rag_tokens = count_tokens(rag_content, MODEL_NAME)
    logging.info(f"RAG tokens: {rag_tokens}")
    return f"Here are the relevant Wikipedia articles:\n{rag_content}\n\nBased on all the information, answer the query. \n\nQuery: {prompt}\n\n", rag_tokens

def get_llm_response(prompt: str, model_name: str) -> Tuple[str, int, int]:
    """
    Gets the LLM response and counts input and output tokens.
    """
    input_tokens = count_tokens(prompt, model_name)
    logging.info(f"Input tokens: {input_tokens}")

    try:
        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
           # api_key=GEMINI_API_KEY,  # Explicitly pass the API key
           # provider="google",
            # Pass project and location for Vertex AI
           # vertex_project=VERTEX_PROJECT,
           # vertex_location=VERTEX_LOCATION,
            timeout=60  # Add a timeout to prevent indefinite hanging
        )
        output_text = response.choices[0].message.content
        output_tokens = count_tokens(output_text, model_name)
        logging.info(f"Output tokens: {output_tokens}")
        return output_text, input_tokens, output_tokens
    except Exception as e:
        logging.error(f"Error getting LLM response: {e}")
        return "", input_tokens, 0

def evaluate_response(question: str, llm_response: str, ground_truth: str, model_name: str) -> Tuple[Dict[str, str], int, int]:
    evaluation_prompt = f"""
    Given the question: {question}
    The model responded: {llm_response}
    The correct answer is: {ground_truth}
    Is the model's response correct? Answer with 'Yes' or 'No', and then explain your reasoning.
    """
    evaluation, eval_input_tokens, eval_output_tokens = get_llm_response(evaluation_prompt, model_name)

    if "Yes" in evaluation:
        decision = "TRUE"
    else:
        decision = "FALSE"

    explanation = evaluation  # The full response is the explanation
    return {"decision": decision, "explanation": explanation}, eval_input_tokens, eval_output_tokens

def main(model: str):
    # Load the dataset
    dataset = load_dataset("google/frames-benchmark", split="test")

    filename = f"evaluation_results_{model.replace('/', '_')}.json"
    existing_results = load_existing_results(filename)
    last_processed_index = get_last_processed_index(existing_results)

    total_rag_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for item in tqdm(dataset, desc="Processing samples"):
        index = int(item['Unnamed: 0'])
        if index <= last_processed_index:
            continue

        prompt, rag_tokens = generate_llm_prompt(item['Prompt'], item['wiki_links'])
        total_rag_tokens += rag_tokens

        llm_response, input_tokens, output_tokens = get_llm_response(prompt, model)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        evaluation, eval_input_tokens, eval_output_tokens = evaluate_response(item['Prompt'], llm_response, item['Answer'], model)
        total_input_tokens += eval_input_tokens
        total_output_tokens += eval_output_tokens

        result = {
            "index": index,
            "prompt": item['Prompt'],
            "ground_truth": item['Answer'],
            "llm_response": llm_response,
            "evaluation_decision": evaluation['decision'],
            "evaluation_explanation": evaluation['explanation'],
            "reasoning_type": item['reasoning_types']
        }

        save_result(filename, result)
        # print(f"Index: {index}, Decision: {result['evaluation_decision']}")
        # Rate limiting sleep (outside the retry loop)
        if sleep_interval > 0:
            time.sleep(sleep_interval)

    # Calculate and print summary statistics
    results = load_existing_results(filename)
    total_samples = len(results)
    correct_answers = sum(1 for r in results if r['evaluation_decision'] == 'TRUE')
    accuracy = correct_answers / total_samples

    print(f"Model: {model}")
    print(f"Total samples: {total_samples}")
    print(f"Correct answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.2%}")

    # Print accuracy by reasoning type
    reasoning_types = set(r['reasoning_type'] for r in results)
    for rt in reasoning_types:
        rt_samples = [r for r in results if r['reasoning_type'] == rt]
        rt_correct = sum(1 for r in rt_samples if r['evaluation_decision'] == 'TRUE')
        rt_accuracy = rt_correct / len(rt_samples)
        print(f"Accuracy for {rt}: {rt_accuracy:.2%}")

    # Print total token counts
    print("\n--- Token Usage Summary ---")
    print(f"Total RAG tokens: {total_rag_tokens}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on google/frames-benchmark")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help=f"Model to use (e.g., {MODEL_NAME})")
    args = parser.parse_args()

    main(args.model)