# This script evaluates the performance of a Gemini LLM model on the google/frames-benchmark dataset
# using a Retrieval-Augmented Generation (RAG) approach with a memory plugin. The meory plugin is part of utility/memory_plugin.py in HaluManage.

import argparse
import json
import os
import sys
import time
from typing import List, Dict
import logging
import traceback

from tqdm import tqdm
from dotenv import load_dotenv
import litellm


# Suppress INFO messages from the litellm library
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# Add the project root to the Python path to allow importing from the halumanage package
# The script is in 'HaluManage/scripts', so we need to go up one level to 'HaluManage' to make the 'halumanage' package importable.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from halumanage.litellm_wrapper import LiteLLMWrapper
# Import the run function from the readurls_plugin
from utility.memory_plugin import run as run_memory_plugin


load_dotenv()  # Load environment variables from .env file

#sleep_interval = 5  # seconds
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the model name for LiteLLM (Google Cloud AI Platform)
MODEL_NAME = "gemini/gemini-2.5-flash-lite"  # Replace with the correct  model identifier on Google Cloud AI Platform

# Get Google Cloud API key from environment variable
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

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

def generate_llm_oracle_prompt(prompt: str, context: str) -> str:
    """
    Generates the oracle LLM prompt with wiki_links for RAG evaluation.
    """
    #logging.info("Generating oracle prompt with RAG URLs.")
    return f"Here are the relevant Wikipedia articles context:\n{context}\n\nBased on all the information, answer the query. \n\nQuery: {prompt}\n\n"

def evaluate_response(question: str, llm_response: str, ground_truth: str, model_name: str) -> Dict[str, str]:

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
        #logging.info("Invoking LLM as an evaluator (direct call).")

        response = litellm.completion( # direct model call to the underlying model
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert evaluator."},
                {"role": "user", "content": evaluation_prompt}
            ],
            max_tokens=500,
            temperature=0.3
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

def main(model: str):
    # Verify that the API key is set
    if not GOOGLE_API_KEY:
        logging.error("Google Cloud API key not found. Please set the GOOGLE_API_KEY environment variable.")
        exit()

    # Rest of your code...
    json_file_path = os.path.join(project_root, 'frames_with_full_context_readurl_v1.0.json')
    # Load the dataset from the local JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    filename = f"evaluation_results_{model.replace('/', '_')}.json"
    existing_results = load_existing_results(filename)
    last_processed_index = get_last_processed_index(existing_results)
    total_input_tokens = 0
    total_output_tokens = 0


    # Instantiate the LiteLLMWrapper to call the LLM directly
    # This avoids using the HaluManage proxy server.
    llm_client = LiteLLMWrapper(api_key=GOOGLE_API_KEY)

    for item in tqdm(dataset, desc="Processing samples"):
        index = int(item['QuestionNo'])
        if index <= last_processed_index:
            continue

        prompt = generate_llm_oracle_prompt(item['Prompt'], item['generated_context']) # Renamed from generate_llm_prompt
        
        input_tokens = count_tokens(prompt, MODEL_NAME)
        logging.info(f"Oracle Prompt Input tokens: {input_tokens}")
        total_input_tokens += input_tokens

        # Call the memory plugin directly instead of get_llm_response
        # The system prompt is passed but the memory plugin uses its own internal prompts.
        system_prompt = "You are a helpful assistant."
        llm_response, plugin_tokens = run_memory_plugin(system_prompt, prompt, llm_client, model)

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