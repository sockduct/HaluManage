"""
Evaluates a Gemini model on the FRAMES benchmark using the native Google AI SDK (new client).

This script implements a simplified RAG pipeline:
1. Read Data: Fetches prompts and pre-generated context from a local JSON file.
2. Generate Answer: Calls the Gemini model once with the full context to get an answer.
3. Evaluate: Calls the Gemini model a second time to evaluate the answer against the ground truth.
"""
import argparse
import json
import os
import logging
import traceback
from typing import List, Dict, Tuple
import re

from tqdm import tqdm
from dotenv import load_dotenv
from google import genai # Correct import for the new SDK
import google.genai.types as types # Import types for configuration

load_dotenv()  # Load environment variables from .env file

# --- Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure the Google AI SDK
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# Create the client object with the API key (REPLACING genai.configure)
# This resolves the 'module 'google.genai' has no attribute 'configure'' error.
client = genai.Client(api_key=api_key)

# Set the model name for the Google AI SDK
MODEL_NAME = "gemini-2.5-flash-lite" # Use an appropriate, available model


# --- Prompts ---

GEMINI_SYSTEM_PROMPT='''You are an expert **Multi-Hop Reasoner** and Final Answer Synthesizer powered by Google's Gemini 2.5 Flash Lite.

Your sole purpose is to answer the complex, multi-hop question using the provided set of "Key Facts" (distilled information).

### Rules for Answering:
1.  **Chain-of-Thought (CoT):** You must first provide a clear, numbered, step-by-step chain of reasoning that logically connects the facts to reach the final answer.
2.  **Final Answer:** After your reasoning, provide the final answer as a single, concise statement under the heading "Final Answer:".
'''

EVAL_SYSTEM_PROMPT = (
    "You are a helpful assistant."
)

EVALUATION_USER_PROMPT = '''===Task===
I need your help in evaluating an answer provided by an LLM against a ground truth
answer. Your task is to determine if the ground truth answer is present in the LLM's response.
Please analyze the provided data and make a decision.

===Instructions===
1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer".
2. Consider the substance of the answers - look for equivalent information or correct answers. Do
not focus on exact wording unless the exact wording is crucial to the meaning.
3. Your final decision should be based on whether the meaning and the vital facts of the "Ground
Truth Answer" are present in the "Predicted Answer:"

===Input Data===
- Question: {question}
- Predicted Answer: {LLM_response}
- Ground Truth Answer: {ground_truth_answer}

===Output Format===
Respond ONLY with a valid JSON object, no explanation or text before/after. Example:
{{"decision": "TRUE", "explanation": "The predicted answer matches the ground truth."}}

Please proceed with the evaluation. '''


# --- LLM Call Functions ---

def call_llm(
    client: genai.Client, # Accept the client object
    system_prompt: str, 
    user_prompt: str, 
    model: str, 
    maxtoken: int
) -> Tuple[str, int, int]:
    """
    Wrapper for the Google AI SDK client.models.generate_content
    """
    try:
        # The new SDK uses a unified client for all operations
        response = client.models.generate_content(
            model=model,
            contents=[
                # The Gemini API expects a list of Contents/Parts. 
                # For system prompts, setting the first part of the user role to the system prompt
                # is the common workaround for API clients that don't directly support a system role.
                # Alternatively, you can use the system_instruction config if available:
                # config=types.GenerateContentConfig(system_instruction=system_prompt, ...)
                {"role": "user", "parts": [
                    {"text": system_prompt + "\n\n" + user_prompt}
                ]}
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=maxtoken,
                temperature=0.0 # Setting temperature to 0.0 is crucial for factual accuracy
            )
        )
        
        # Extract response text and token counts
        answer_text = response.text
        
        # Token calculation uses the usage_metadata from the response
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        
        return answer_text, input_tokens, output_tokens
        
    except Exception as e:
        logging.error(f"Error calling LLM {model}: {e}")
        traceback.print_exc()
        # Return empty values for error state
        return f"Error: {e}", 0, 0

def save_result(filename: str, result: Dict):
    """Saves the result to a JSON file."""
    results = load_existing_results(filename)
    results.append(result)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def load_existing_results(filename: str) -> List[Dict]:
    """Loads existing results from a JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def get_last_processed_index(results: List[Dict]) -> int:
    """Finds the index of the last processed item."""
    if not results:
        return -1
    return max(int(r.get('index', -1)) for r in results)

MAX_OUTPUT_TOKENS = 1014 # Generous limit for the detailed answer and evaluation

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM with pre-compiled context from a JSON file using Google AI SDK.")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model identifier (e.g., 'gemini-2.5-flash-lite')")
    parser.add_argument("--input_file", type=str, default="frames_with_context_readurl.json", help="Path to the input JSON file with pre-generated context.")
    args = parser.parse_args()
    
    model = args.model
    input_file = args.input_file
    output_filename = f"results_google_sdk_{model.replace('/', '_')}_rag.json"

    # --- Load Data and Resume ---
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_file}. Please ensure you have the FRAMES data with context.")
        return

    results = load_existing_results(output_filename)
    start_index = get_last_processed_index(results) + 1
    total_items = len(data)

    logging.info(f"Starting evaluation with model: {model}")
    logging.info(f"Loaded {total_items} items. Resuming from index: {start_index}")

    total_rag_input_tokens = sum(r.get('rag_input_tokens', 0) for r in results)
    total_rag_output_tokens = sum(r.get('rag_output_tokens', 0) for r in results)
    total_eval_input_tokens = sum(r.get('eval_input_tokens', 0) for r in results)
    total_eval_output_tokens = sum(r.get('eval_output_tokens', 0) for r in results)


    # --- Main Loop ---

    for index, item in enumerate(tqdm(data[start_index:], initial=start_index, total=total_items)):
        
        # 1. RAG Step (Answer Generation)
        rag_user_prompt = f"Question: {item['Prompt']}\n\nContext:\n'''{item['generated_context']}'''"
        
        answer, rag_input_tokens, rag_output_tokens = call_llm(
            client, # PASS CLIENT
            GEMINI_SYSTEM_PROMPT, rag_user_prompt, model, MAX_OUTPUT_TOKENS
        )
        
        total_rag_input_tokens += rag_input_tokens
        total_rag_output_tokens += rag_output_tokens

        eval_user_prompt = EVALUATION_USER_PROMPT.format(question={item['Prompt']}, LLM_response={answer},
                                  ground_truth_answer={item['Answer']})
        
        # Request the evaluation in JSON format
        evaluation_response, eval_input_tokens, eval_output_tokens = call_llm(
            client, # PASS CLIENT
            EVAL_SYSTEM_PROMPT, 
            eval_user_prompt, 
            model, 
            maxtoken=500 # Smaller output limit for evaluation JSON
        )
        
        total_eval_input_tokens += eval_input_tokens
        total_eval_output_tokens += eval_output_tokens

        # 3. Parse and Save Result
        try:
            # Try to find JSON object in the response
            match = re.search(r'\{.*\}', evaluation_response, re.DOTALL)
            if match:
                evaluation = json.loads(match.group(0))
            else:
                raise json.JSONDecodeError("No JSON found", evaluation_response, 0)
        except json.JSONDecodeError:
            # Handle cases where the LLM fails to return valid JSON
            logging.warning(f"Failed to parse JSON for index {index}. Response: {evaluation_response[:100]}")
            evaluation = {'decision': 'ERROR', 'explanation': f'Failed to parse JSON: {evaluation_response}'}
        
        result = {
            "index": index,
            "prompt": item['Prompt'],
            "ground_truth": item['Answer'],
            "full_context": item['generated_context'],
            "model_answer": answer,
            "rag_input_tokens": rag_input_tokens,
            "rag_output_tokens": rag_output_tokens,
            "eval_input_tokens": eval_input_tokens,
            "eval_output_tokens": eval_output_tokens,
            "evaluation_decision": evaluation.get('decision', 'ERROR'),
            "evaluation_explanation": evaluation.get('explanation', 'N/A')
            #"reasoning_type": item['reasoning_types']
        }
        
        save_result(output_filename, result)


    # --- Final Summary ---
    
    # Calculate and print summary statistics
    results = load_existing_results(output_filename)
    
    if not results:
        logging.info("No results to summarize.")
        return
        
    correct = sum(1 for r in results if r['evaluation_decision'] == 'TRUE')
    accuracy = correct / len(results) if results else 0

    print("\n" + "="*30 + "\n      EVALUATION SUMMARY\n" + "="*30)
    print(f"Model: {model}")
    print(f"Total samples: {len(results)}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\n--- Token Usage Summary ---")
    print(f"Total RAG Input Tokens:  {total_rag_input_tokens:,}")
    print(f"Total RAG Output Tokens: {total_rag_output_tokens:,}")
    print(f"Total EVAL Input Tokens: {total_eval_input_tokens:,}")
    print(f"Total EVAL Output Tokens:{total_eval_output_tokens:,}")
    print("="*30)

if __name__ == "__main__":
    main()