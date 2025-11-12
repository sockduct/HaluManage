"""
Updated eval_frames_benchmark_Gemini_RAG.py

This script implements an advanced standalone RAG pipeline
to fairly evaluate the Gemini model on the FRAMES benchmark.

It replicates the "Map-Reduce" logic from the memory_plugin.py:
1. Read URLs: Fetches content from all wiki_links.
2. Chunk: Splits the combined text into manageable chunks.
3. Map: Calls the LLM on each chunk to extract relevant "key info".
4. Reduce: Calls the LLM one final time on the combined "key info"
   to synthesize the final answer.
"""
import argparse
import json
import os
import time
import ast
import logging
import traceback
import requests
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup

from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
import litellm

load_dotenv()  # Load environment variables from .env file

# --- Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress INFO messages from the litellm library
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# Set the model name for LiteLLM
# Using Gemini 2.5 Flash Lite as in the original script
MODEL_NAME = "gemini/gemini-2.5-flash"

# Get Google Cloud API key from environment variable
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logging.error("Google Cloud API key not found. Please set the GOOGLE_API_KEY environment variable.")
    exit()

# Configure litellm to use the API key
litellm.api_key = GOOGLE_API_KEY 

# --- RAG Helper Functions ---

# 1. Fetching (from readurls_plugin logic)
def fetch_and_clean_text(url: str, max_length: int = 100000) -> str:
    """Fetches and cleans text content from a URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the main content body of a Wikipedia page
        content_div = soup.find(id='bodyContent')
        
        if content_div:
            # Extract text primarily from paragraphs
            paragraphs = content_div.find_all('p', recursive=True)
            text = "\n".join([p.get_text() for p in paragraphs])
        else:
            text = soup.get_text() # Fallback

        # Truncate to max_length
        if len(text) > max_length:
            text = text[:max_length] + '...'
        
        return text
    except Exception as e:
        logging.warning(f"Error fetching {url}: {str(e)}")
        return ""

# 2. Chunking
def text_to_chunks(text: str, chunk_size: int = 10000, overlap: int = 200) -> List[str]:
    """Splits text into overlapping chunks."""
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# --- LLM Call Functions ---

def call_llm(system_prompt: str, user_prompt: str, model: str) -> Tuple[str, int, int]:
    """
    Wrapper for litellm.completion to get response and token counts.
    Returns (response_text, input_tokens, output_tokens)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=0.0,
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

# 3. Map Step (from memory_plugin logic)
EXTRACT_SYSTEM_PROMPT = """
You are a highly efficient research assistant. Your task is to read the following text chunk and extract *only* the key facts, names, dates, numbers, and relationships that are directly relevant to answering the user's question.
- Be concise. Output *only* bullet points of facts.
- If no relevant information is found in the text, respond with "N/A".
- Do not answer the question, only extract information.
"""

def extract_key_info_from_chunk(chunk: str, original_question: str, model: str) -> Tuple[str, int, int]:
    """MAP step: Calls LLM on one chunk to extract key info."""
    user_prompt = f"""
    **Original Question:**
    "{original_question}"
    
    **Text Chunk:**
    ---
    {chunk}
    ---
    
    Extract key information relevant to the question from the chunk above.
    """
    return call_llm(EXTRACT_SYSTEM_PROMPT, user_prompt, model)

# 4. Reduce Step (from memory_plugin logic)
SYNTHESIZE_SYSTEM_PROMPT = """
You are an expert question-answering assistant. Your task is to answer the user's question based *only* on the provided "Margin Notes".
- Read all the notes carefully.
- Synthesize a final, concise answer to the question.
- Do not use any external knowledge.
- If the answer cannot be found in the notes, state: "The answer is not contained in the provided notes."
"""

def synthesize_final_answer(all_notes: str, original_question: str, model: str) -> Tuple[str, int, int]:
    """REDUCE step: Calls LLM on all "notes" to get the final answer."""
    user_prompt = f"""
    **Original Question:**
    "{original_question}"

    **Collected Margin Notes:**
    ---
    {all_notes}
    ---

    Based *only* on the notes above, what is the final answer to the question?
    """
    return call_llm(SYNTHESIZE_SYSTEM_PROMPT, user_prompt, model)

# --- Original Script Functions (Modified) ---

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

# The evaluator LLM call (as in the original)
def evaluate_llm_response(prompt: str, llm_response: str, ground_truth: str, model: str) -> Tuple[str, str, int, int]:
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
    
    response_text, input_tokens, output_tokens = call_llm(
        eval_system_prompt, 
        evaluator_prompt,
        model  # Using the same model for evaluation
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

# --- Main Execution Logic ---

def main(model: str, max_samples: int):
    # Load dataset from Hugging Face
    try:
        dataset = load_dataset("google/frames-benchmark", split="test")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}. Make sure 'datasets' is installed.")
        return

    if max_samples:
        dataset = dataset.select(range(max_samples))
        logging.info(f"Running in test mode with {max_samples} samples.")

    # Setup output file
    filename = f"results/eval_frames_benchmark_gemini-2.5-flash-lite_RAG_LiteLLM.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    results = load_existing_results(filename)
    last_processed_index = get_last_processed_index(results)
    
    # Global token counters
    total_rag_input_tokens = 0
    total_rag_output_tokens = 0
    total_eval_input_tokens = 0
    total_eval_output_tokens = 0

    try:
        for i, item in enumerate(tqdm(dataset, desc="Evaluating RAG prompts")):
            if i <= last_processed_index:
                continue

            # --- This is the new RAG Pipeline ---
            question = item['Prompt']
            ground_truth = item['Answer']
            
            # 1. Read URLs
            try:
                wiki_links = ast.literal_eval(item['wiki_links'])
                if not isinstance(wiki_links, list):
                    wiki_links = []
            except Exception:
                wiki_links = []

            logging.info(f"\nProcessing [#{i}]: {question}")
            logging.info(f"Fetching {len(wiki_links)} URLs...")
            
            full_context = ""
            for url in wiki_links:
                full_context += fetch_and_clean_text(url) + "\n\n"

            if not full_context.strip():
                logging.warning("No context retrieved. Skipping to eval with empty answer.")
                llm_answer = "Error: No context could be retrieved."
                rag_input_tokens, rag_output_tokens = 0, 0
            
            else:
                # 2. Chunk
                chunks = text_to_chunks(full_context)
                logging.info(f"Context fetched ({len(full_context)} chars), split into {len(chunks)} chunks.")
                
                all_notes = []
                current_rag_input_tokens = 0
                current_rag_output_tokens = 0
                
                # 3. Map
                for chunk in chunks:
                    note, in_tok, out_tok = extract_key_info_from_chunk(chunk, question, model)
                    current_rag_input_tokens += in_tok
                    current_rag_output_tokens += out_tok
                    if note != "N/A":
                        all_notes.append(note)
                
                logging.info(f"Extracted {len(all_notes)} relevant notes.")
                combined_notes = "\n".join(all_notes)
                
                # 4. Reduce
                if not combined_notes:
                    llm_answer = "The answer is not contained in the provided notes."
                    rag_input_tokens, rag_output_tokens = 0, 0 # No final call
                else:
                    llm_answer, in_tok, out_tok = synthesize_final_answer(combined_notes, question, model)
                    current_rag_input_tokens += in_tok
                    current_rag_output_tokens += out_tok
                
                rag_input_tokens = current_rag_input_tokens
                rag_output_tokens = current_rag_output_tokens
                
                total_rag_input_tokens += rag_input_tokens
                total_rag_output_tokens += rag_output_tokens

            # --- End of new RAG Pipeline ---

            # Evaluate the response
            eval_decision, eval_explanation, eval_in_tok, eval_out_tok = evaluate_llm_response(
                question, llm_answer, ground_truth, model
            )
            
            total_eval_input_tokens += eval_in_tok
            total_eval_output_tokens += eval_out_tok
            
            # Save result
            result = {
                "index": i,
                "prompt": question,
                "rag_llm_response": llm_answer,
                "ground_truth": ground_truth,
                "evaluation_decision": eval_decision,
                "evaluation_explanation": eval_explanation,
                "reasoning_type": item['reasoning_types'],
                "rag_input_tokens": rag_input_tokens,
                "rag_output_tokens": rag_output_tokens,
                "eval_input_tokens": eval_in_tok,
                "eval_output_tokens": eval_out_tok
            }
            
            save_result(filename, result)
            logging.info(f"Result for [#{i}]: {eval_decision} - {llm_answer}")

            # time.sleep(1) # Optional: rate limiting

    except KeyboardInterrupt:
        logging.info("Evaluation interrupted by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        logging.error(traceback.format_exc())

    finally:
        # Calculate and print summary statistics
        results = load_existing_results(filename)
        total_samples = len(results)
        if total_samples == 0:
            logging.info("No results to summarize.")
            return
            
        correct_answers = sum(1 for r in results if r['evaluation_decision'] == 'TRUE')
        accuracy = correct_answers / total_samples

        print("\n" + "="*30)
        print("      EVALUATION SUMMARY")
        print("="*30)
        print(f"Model: {model} (with RAG)")
        print(f"Total samples: {total_samples}")
        print(f"Correct answers: {correct_answers}")
        print(f"Accuracy: {accuracy:.2%}")

        # Print total token counts
        print("\n--- Token Usage Summary ---")
        print(f"Total RAG Input Tokens:  {total_rag_input_tokens}")
        print(f"Total RAG Output Tokens: {total_rag_output_tokens}")
        print(f"Total EVAL Input Tokens: {total_eval_input_tokens}")
        print(f"Total EVAL Output Tokens:{total_eval_output_tokens}")

        # Print accuracy by reasoning type
        print("\n--- Accuracy by Reasoning Type ---")
        reasoning_types = set()
        for r in results:
            for rt in r['reasoning_type'].split(' | '):
                reasoning_types.add(rt.strip())

        for rt in sorted(list(reasoning_types)):
            rt_samples = [r for r in results if rt in r['reasoning_type']]
            if rt_samples:
                rt_correct = sum(1 for r in rt_samples if r['evaluation_decision'] == 'TRUE')
                rt_accuracy = rt_correct / len(rt_samples)
                print(f"- {rt:<30} {rt_accuracy:>6.2%}  ({rt_correct}/{len(rt_samples)})")
        
        print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM RAG performance on google/frames-benchmark")
    parser.add_argument(
        "--model", 
        type=str, 
        default=MODEL_NAME, 
        help="The model identifier to use (e.g., 'gemini/gemini-2.5-flash-lite')"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )
    args = parser.parse_args()
    
    main(model=args.model, max_samples=args.max_samples)