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

load_dotenv()  # Load environment variables from .env file

sleep_interval = 5  # seconds
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the model name for LiteLLM (Google Cloud AI Platform)
MODEL_NAME = "gemini-2.5-flash-lite"  # Replace with the correct  model identifier on Google Cloud AI Platform

# Get Google Cloud API key from environment variable
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Verify that the API key is set
if not GOOGLE_API_KEY:
    logging.error("Google Cloud API key not found. Please set the GOOGLE_API_KEY environment variable.")
    exit()

# Function to check if the Google Cloud token is valid by making a simple API call
def is_google_cloud_token_valid(token):
    try:
        litellm.completion(
            model="gemini-2.5-flash-lite",  # Use a basic model to test the token
            messages=[{"role": "user", "content": "test"}],
            api_key=token,
            provider="google",
            timeout=10  # Add a timeout to prevent indefinite hanging
        )
        return True
    except Exception as e:
        logging.error(f"Google Cloud token validation failed: {e}")
        return False

# Validate the Google Cloud token
if not is_google_cloud_token_valid(GOOGLE_API_KEY):
    logging.error("Google Cloud API token is invalid. Please check your token and permissions.")
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

def generate_llm_prompt(prompt: str, wiki_links: List[str]) -> str:
    """
    Generates the LLM prompt and counts RAG tokens.
    """
    rag_content = "\n".join(wiki_links)
    rag_tokens = count_tokens(rag_content, MODEL_NAME)
    logging.info(f"RAG tokens: {rag_tokens}")
    return f"Here are the relevant Wikipedia articles:\n{rag_content}\n\nBased on all the information, answer the query. \n\nQuery: {prompt}\n\n"

def get_llm_response(prompt: str, model_name: str) -> str:
    """
    Gets the LLM response and counts input and output tokens.
    """
    input_tokens = count_tokens(prompt, model_name)
    logging.info(f"Input tokens: {input_tokens}")
        
    try:
        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            api_key=GOOGLE_API_KEY,  # Explicitly pass the API key
            provider="google",
            timeout=60  # Add a timeout to prevent indefinite hanging
        )
        output_text = response.choices[0].message.content
        output_tokens = count_tokens(output_text, model_name)
        logging.info(f"Output tokens: {output_tokens}")
        return output_text
    except Exception as e:
        logging.error(f"Error getting LLM response: {e}")
        return ""

def evaluate_response(question: str, llm_response: str, ground_truth: str, model_name: str) -> Dict[str, str]:
    evaluation_prompt = f"""
    Given the question: {question}
    The model responded: {llm_response}
    The correct answer is: {ground_truth}
    Is the model's response correct? Answer with 'Yes' or 'No', and then explain your reasoning.
    """
    evaluation = get_llm_response(evaluation_prompt, model_name)

    if "Yes" in evaluation:
        decision = "TRUE"
    else:
        decision = "FALSE"

    explanation = evaluation  # The full response is the explanation
    return {"decision": decision, "explanation": explanation}

def main(model: str):
    # Load the dataset
    dataset = load_dataset("google/frames-benchmark", split="test")

    filename = f"evaluation_results_{model.replace('/', '_')}.json"
    existing_results = load_existing_results(filename)
    last_processed_index = get_last_processed_index(existing_results)

    for item in tqdm(dataset, desc="Processing samples"):
        index = int(item['Unnamed: 0'])
        if index <= last_processed_index:
            continue

        prompt = generate_llm_prompt(item['Prompt'], item['wiki_links'])
        llm_response = get_llm_response(prompt, model)
        evaluation = evaluate_response(item['Prompt'], llm_response, item['Answer'], model)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on google/frames-benchmark")
    parser.add_argument("--model", type=str, required=True, help="Model to use (e.g., gemini-2.5-flash-lite, gemma-2-27b-it)")
    args = parser.parse_args()

    main(args.model)