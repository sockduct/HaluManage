"""
Updated script to evaluate the Gemini model on the FRAMES benchmark
using a pre-compiled context from a JSON file.

This script implements a simplified RAG pipeline:
1. Read Data: Fetches prompts and pre-generated context from a local JSON file.
2. Generate Answer: Calls the LLM once with the full context to get an answer.
3. Evaluate: Calls the LLM a second time to evaluate the answer against the ground truth.
"""
import argparse
import json
import os
import logging
import traceback
from typing import List, Dict, Tuple

from tqdm import tqdm
from dotenv import load_dotenv
import litellm

load_dotenv()  # Load environment variables from .env file

# --- Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress INFO messages from the litellm library
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# Set the model name for LiteLLM
MODEL_NAME = "gemini/gemini-2.5-flash-lite"

# --- LLM Call Functions ---

def call_llm(system_prompt: str, user_prompt: str, model: str, maxtoken: int) -> Tuple[str, int, int]:
    """
    Wrapper for litellm.completion to get response and token counts.
    Returns (response_text, input_tokens, output_tokens)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        # Ensure the API key is set for the call if not globally configured
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")

        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=0.0,
            api_key=api_key,
            max_tokens=maxtoken
        )
        
        response_text = response.choices[0].message.content.strip()
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        
        return response_text, input_tokens, output_tokens

    except Exception as e:
        logging.error(f"Error calling LLM {model}: {e}")
        logging.error(traceback.format_exc())
        return f"Error: {e}", 0, 0

# --- Main RAG and Evaluation Functions ---

ANSWER_SYSTEM_PROMPT = """
You are an expert question-answering assistant. Your task is to answer the user's question based *only* on the provided "Context".
- Read the context carefully.
- Synthesize a final, concise answer to the question.
- Do not use any external knowledge.
- If the answer cannot be found in the context, state: "The answer is not contained in the provided context."
"""

def get_answer_from_context(context: str, original_question: str, model: str) -> Tuple[str, int, int]:
    """Calls the LLM with the pre-fetched context to get the final answer."""
    user_prompt = f"""
    **Context:**
    ---
    {context}
    ---

    **Question:**
    "{original_question}"

    Based *only* on the context above, what is the final answer to the question?
    """
    maxtoken=2000
    return call_llm(ANSWER_SYSTEM_PROMPT, user_prompt, model, maxtoken)

def evaluate_llm_response(prompt: str, llm_response: str, ground_truth: str, model: str) -> Tuple[str, str, int, int]:
    """Uses an LLM to evaluate the correctness of the generated answer."""
    evaluator_prompt = f"""
    Please act as an impartial evaluator.
    Compare the "Ground Truth" with the "LLM's Answer" based on the "Question".
    The LLM's Answer must be factually correct and semantically equivalent to the Ground Truth.
    
    Question: {prompt}
    Ground Truth: {ground_truth}
    LLM's Answer: {llm_response}

    Respond with your decision (TRUE or FALSE) and a brief explanation in this JSON format:
    {{"decision": "TRUE/FALSE", "explanation": "Your reason here."}}
    """
    
    eval_system_prompt = "You are an impartial, expert evaluator."
    max_tokens=1000
    response_text, input_tokens, output_tokens = call_llm(
        eval_system_prompt, 
        evaluator_prompt,
        model,
        max_tokens
    )
    
    try:
        # Clean the response if it includes markdown
        if response_text.startswith("```json"):
            response_text = response_text.strip("```json\n").strip("```")
        
        eval_json = json.loads(response_text)
        decision = eval_json.get("decision", "FALSE")
        explanation = eval_json.get("explanation", "Error parsing evaluation.")
        return decision, explanation, input_tokens, output_tokens
    except Exception as e:
        logging.error(f"Failed to parse eval JSON: {response_text}. Error: {e}")
        return "FALSE", f"Error parsing evaluation response: {e}", input_tokens, output_tokens

# --- File I/O and Main Execution Logic ---

def load_json_data(filename: str) -> List[Dict]:
    """Loads data from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Input file not found: {filename}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Could not decode JSON from {filename}.")
        return []

def save_result(filename: str, result: Dict):
    """Appends a result to a JSON file."""
    results = load_json_data(filename)
    results.append(result)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def get_last_processed_index(results: List[Dict]) -> int:
    """Finds the last processed index to allow resuming."""
    if not results:
        return -1
    return max(int(r.get('index', -1)) for r in results)

def main(model: str, input_file: str, max_samples: int):
    # Load dataset from the specified JSON file
    dataset = load_json_data(input_file)
    if not dataset:
        return

    if max_samples:
        dataset = dataset[:max_samples]
        logging.info(f"Running in test mode with {max_samples} samples.")

    # Setup output file
    output_filename = f"results/eval_from_json_context_{model.replace('/', '_')}.json"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    existing_results = load_json_data(output_filename)
    last_processed_index = get_last_processed_index(existing_results)
    
    # Global token counters
    total_rag_input_tokens, total_rag_output_tokens = 0, 0
    total_eval_input_tokens, total_eval_output_tokens = 0, 0

    try:
        for i, item in enumerate(tqdm(dataset, desc="Evaluating prompts from JSON")):
            if i <= last_processed_index:
                continue

            question = item.get('Prompt')
            ground_truth = item.get('Answer')
            context = item.get('generated_context')

            if not all([question, ground_truth, context]):
                logging.warning(f"Skipping item #{i} due to missing data.")
                continue

            logging.info(f"\nProcessing [#{i}]: {question}")

            # 1. Generate Answer
            llm_answer, rag_in_tok, rag_out_tok = get_answer_from_context(context, question, model)
            total_rag_input_tokens += rag_in_tok
            total_rag_output_tokens += rag_out_tok

            # 2. Evaluate the response
            eval_decision, eval_explanation, eval_in_tok, eval_out_tok = evaluate_llm_response(
                question, llm_answer, ground_truth, model
            )
            total_eval_input_tokens += eval_in_tok
            total_eval_output_tokens += eval_out_tok
            
            # 3. Save result
            result = {
                "index": i,
                "prompt": question,
                "llm_response": llm_answer,
                "ground_truth": ground_truth,
                "evaluation_decision": eval_decision,
                "evaluation_explanation": eval_explanation,
                "rag_input_tokens": rag_in_tok,
                "rag_output_tokens": rag_out_tok,
                "eval_input_tokens": eval_in_tok,
                "eval_output_tokens": eval_out_tok
            }
            
            save_result(output_filename, result)
            logging.info(f"Result for [#{i}]: {eval_decision} - {llm_answer}")

    except KeyboardInterrupt:
        logging.info("Evaluation interrupted by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

    finally:
        # Final summary
        results = load_json_data(output_filename)
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
        print(f"Total RAG Input Tokens:  {total_rag_input_tokens}")
        print(f"Total RAG Output Tokens: {total_rag_output_tokens}")
        print(f"Total EVAL Input Tokens: {total_eval_input_tokens}")
        print(f"Total EVAL Output Tokens:{total_eval_output_tokens}")
        print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM with pre-compiled context from a JSON file.")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model identifier")
    parser.add_argument("--input_file", type=str, default="frames_with_fulll_context_readurl.json", help="Path to the input JSON file with pre-generated context.")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    args = parser.parse_args()
    
    main(model=args.model, input_file=args.input_file, max_samples=args.max_samples)