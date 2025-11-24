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
MODEL_NAME = "gemini/gemini-2.5-flash-lite"  # Replace with the correct  model identifier on Google Cloud AI Platform

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
            model=MODEL_NAME,  # Use a basic model to test the token
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
    if not text:
        return 0
    try:
        # Use litellm's built-in token counter
        return litellm.token_counter(model=model_name, text=text)
    except Exception as e:
        logging.warning(f"Could not count tokens for model {model_name}: {e}. Falling back to character count / 4.")
        # Fallback to a rough estimate if litellm fails
        return len(text) // 4

def generate_llm_naive_prompt(prompt: str, wiki_links: List[str]) -> str:
    """
    Generates the naive LLM prompt without RAG/wiki_links for baseline evaluation.
    """
    # For baseline evaluation, we ignore the wiki_links and only use the prompt.
    logging.info("Generating naive prompt without RAG content for baseline.")
    return prompt

def get_llm_response(prompt: str, model_name: str) -> str:
    try:
        response = litellm.completion(
            #model=f"litellm_proxy/{model_name}",
            model=model_name,
            messages=[ 
                {"role": "user", "content": prompt}
            ],
            #api_key="halumanage",  # A placeholder key is sufficient for the proxy
            api_key=GOOGLE_API_KEY,
            provider="google",
            # Use lower temperature for factual/reasoning tasks, aligning with mitigation best practices
            temperature=0.1
        )
        output_text = response.choices[0].message.content
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

    filename = f"eval_results_{model.replace('/', '_')}_baseline.json"
    existing_results = load_existing_results(filename)
    last_processed_index = get_last_processed_index(existing_results)
    total_input_tokens = 0
    total_output_tokens = 0

    for item in tqdm(dataset, desc="Processing samples"):
        index = int(item['Unnamed: 0'])
        if index <= last_processed_index:
            continue

        prompt = generate_llm_naive_prompt(item['Prompt'], item['wiki_links']) # Renamed from generate_llm_prompt
        
        input_tokens = count_tokens(prompt, MODEL_NAME)
        logging.info(f"Naive Prompt Input tokens: {input_tokens}")
        total_input_tokens += input_tokens

        llm_response = get_llm_response(prompt, model)
        output_tokens = count_tokens(llm_response, MODEL_NAME)
        logging.info(f"Output tokens: {output_tokens}")
        total_output_tokens += output_tokens

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

    # Print total token counts
    print("\n--- Token Usage Summary ---")
    print(f"Total input tokens for naive prompts: {total_input_tokens}")
    print(f"Total output tokens for naive prompts: {total_output_tokens}")

    # Print accuracy by reasoning type
    reasoning_types = set(r['reasoning_type'] for r in results)
    for rt in reasoning_types:
        rt_samples = [r for r in results if r['reasoning_type'] == rt]
        rt_correct = sum(1 for r in rt_samples if r['evaluation_decision'] == 'TRUE')
        rt_accuracy = rt_correct / len(rt_samples)
        print(f"Accuracy for {rt}: {rt_accuracy:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on google/frames-benchmark")
    # Make the --model argument optional by providing a default value.
    # This fixes the "arguments are required: --model" error.
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Model to use (e.g., {MODEL_NAME})",
    )
    args = parser.parse_args()

    main(args.model)