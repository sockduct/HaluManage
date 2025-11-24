import argparse
import json
import os
import time
from typing import List, Dict
import logging

from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()  # Load environment variables from .env file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Replace with your actual Hugging Face API token
HUGGING_FACE_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

print(f"HUGGING_FACE_API_TOKEN: {HUGGING_FACE_API_TOKEN}")  


# Check CUDA availability
if torch.cuda.is_available():
    device = "cuda"
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("CUDA is not available. Using CPU.")

# Load model and tokenizer
'''The from_pretrained method from the transformers library is designed to:
1. Check a local cache directory (usually ~/.cache/huggingface/hub) for the model files.
2. If the files for "google/gemma-7b-it" are not found, it will download them from the Hugging Face Hub.
3. Once downloaded, the model will be cached locally, so subsequent runs of the script will load it from your disk instead of downloading it again.'''
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it", token=HUGGING_FACE_API_TOKEN)

model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", token=HUGGING_FACE_API_TOKEN, torch_dtype=torch.bfloat16, device_map="auto")

SLEEP_INTERVAL = 300

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
    input_text = prompt
    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**input_ids, max_new_tokens=200)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

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
    parser.add_argument("--model", type=str, required=True, help="Model to use (e.g., gemma-7b-it)")
    args = parser.parse_args()

    main(args.model)