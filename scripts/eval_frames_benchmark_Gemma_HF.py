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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the model name for LiteLLM
MODEL_NAME = "gemma-2-9b-it"  # Replace with the correct gemma-2-27b model identifier for LiteLLM
# Get Hugging Face API token from environment variable
HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# Verify that the API token is set
if not HUGGINGFACE_API_TOKEN:
    logging.error("Hugging Face API token not found. Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")
    exit()

# Function to check if the Hugging Face token is valid by making a simple API call
def is_huggingface_token_valid(token):
    try:
        litellm.completion(
            model="huggingface/google/gemma-2-9b-it",  # Use a basic model to test the token
            messages=[{"role": "user", "content": "test"}],
            api_key=token,
            timeout=10  # Add a timeout to prevent indefinite hanging
        )
        return True
    except Exception as e:
        logging.error(f"Hugging Face token validation failed: {e}")
        return False

# Validate the Hugging Face token
if not is_huggingface_token_valid(HUGGINGFACE_API_TOKEN):
    logging.error("Hugging Face API token is invalid. Please check your token and permissions.")
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

def generate_llm_prompt(prompt: str, wiki_links: List[str]) -> str:
    return f"Here are the relevant Wikipedia articles:\n{wiki_links}\n\nBased on all the information, answer the query. \n\nQuery: {prompt}\n\n"

def get_llm_response(prompt: str, model_name: str) -> str:
    try:
        response = litellm.completion(
            model=f"huggingface/google/{model_name}",
            messages=[{"role": "user", "content": prompt}],
            api_key=HUGGINGFACE_API_TOKEN,  # Explicitly pass the API key
            timeout=60  # Add a timeout to prevent indefinite hanging
        )
        return response.choices[0].message.content
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
        # time.sleep(SLEEP_INTERVAL)

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
    parser.add_argument("--model", type=str, required=True, help="Model to use (e.g., gemma-2-9b-it, gemma-2-7b, google/gemma-2-27b)")
    args = parser.parse_args()

    main(args.model)