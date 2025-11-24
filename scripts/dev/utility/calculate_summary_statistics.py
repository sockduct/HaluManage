import json
import argparse
from typing import List, Dict


def load_results(filename: str) -> List[Dict]:
    """Loads evaluation results from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filename}")
        return []


def main(filename: str):
    """
    Loads evaluation results and prints summary statistics including overall
    accuracy and accuracy broken down by reasoning type.
    """
    results = load_results(filename)
    if not results:
        print("No results to analyze.")
        return

    # The model name is not stored in the results file, so we extract it from the filename
    # assuming the format 'evaluation_results_model_name.json'
    model_name_from_file = filename.replace('evaluation_results_', '').replace('.json', '').replace('_', '/')

    total_samples = len(results)
    correct_answers = sum(1 for r in results if r.get('evaluation_decision') == 'TRUE')
    
    if total_samples == 0:
        accuracy = 0.0
    else:
        accuracy = correct_answers / total_samples

    print(f"Model: {model_name_from_file}")
    print(f"Total samples: {total_samples}")
    print(f"Correct answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.2%}")

    # Print accuracy by reasoning type
    reasoning_types = sorted(list(set(r['reasoning_type'] for r in results if 'reasoning_type' in r)))
    if reasoning_types:
        print("\n--- Accuracy by Reasoning Type ---")
        for rt in reasoning_types:
            rt_samples = [r for r in results if r.get('reasoning_type') == rt]
            rt_correct = sum(1 for r in rt_samples if r.get('evaluation_decision') == 'TRUE')
            rt_accuracy = rt_correct / len(rt_samples) if rt_samples else 0.0
            print(f"Accuracy for {rt}: {rt_accuracy:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate summary statistics from LLM evaluation results. If no filename is provided, it defaults to 'evaluation_results_gemini_gemini-2.5-flash-lite.json'.")
    parser.add_argument(
        'filename',
        nargs='?',
        default="evaluation_results_gemini_gemini-2.5-flash-lite.json",
        type=str,
        help="Path to the evaluation results JSON file. Defaults to 'evaluation_results_gemini_gemini-2.5-flash-lite.json' if not provided."
    )
    args = parser.parse_args()
    main(args.filename)
