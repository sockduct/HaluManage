# This script reads a CSV file containing prompts, answers, and associated Wikipedia links,
# fetches the content of those links using optillm readurl plugin, and saves the augmented data to a new JSON file

import pandas as pd
import ast
import json
import os
from tqdm import tqdm
import logging
import sys
import os
import argparse

# Add the project root to the Python path to allow importing from the halumanage package
# The script is in 'HaluManage/scripts', so we need to go up one level to 'HaluManage' to make the 'halumanage' package importable.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the run function from the readurls_plugin
from utility.readurls_plugin_extract_content import run as run_readurls

# --- Configuration ---
#INPUT_CSV_PATH = r'd:\vscworkspace\HaluManage\frames_benchmark_data_test.csv'
#OUTPUT_JSON_PATH = r'd:\vscworkspace\HaluManage\frames_with_context_readurl1.json'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_existing_results(filename: str) -> list:
    """Loads previously saved results from a JSON file."""
    if not os.path.exists(filename):
        return []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        logging.warning(f"Could not decode JSON from {filename}. Starting fresh.")
        return []

def save_results(filename: str, results: list):
    """Saves a list of results to a JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving results to JSON: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract context for FRAMES using readurls plugin")
    parser.add_argument("--input_csv", default=r'd:\vscworkspace\HaluManage\frames_benchmark_dataset.csv', help="Path to input CSV")
    parser.add_argument("--output_json", default=r'd:\vscworkspace\HaluManage\frames_with_context_readurl_final.json', help="Output JSON path")
    args = parser.parse_args()

    input_path = args.input_csv
    output_path = args.output_json

    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        logging.error("Please verify path. Common locations to check:\n - scripts/data/\n - scripts/async/data/\n - repository root")
        return

    print(f"Loading data from {input_path}...")
    logging.info(f"Loading data from {input_path}...")

    try:
        # Read all columns as strings to avoid dtype issues and NaN conversion for empty cells.
        # na_filter=False prevents pandas from interpreting empty strings as NaN.
        df = pd.read_csv(input_path, dtype=str, na_filter=False)
    except Exception as e:
        logging.error(f"Failed to read CSV {input_path}: {e}")
        return

    # Check if output file exists to handle resuming progress
    results = load_existing_results(output_path)
    processed_prompts = set()

    if results:
        logging.info(f"Output file found at {output_path}. Resuming progress.")
        processed_prompts = {res['Prompt'] for res in results}
        logging.info(f"Found {len(processed_prompts)} already processed records to skip.")
    
    # Use tqdm to create a progress bar for the loop
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Fetching URL Content"):
        questionNo = row['Unnamed: 0']
        prompt = row['Prompt']
        logging.info(f"Processing Question {questionNo} Started")
        
        # Skip if this prompt has already been processed and saved
        if prompt in processed_prompts:
            continue
            
        answer = row['Answer']
        wiki_link1 = row['wikipedia_link_1']
        wiki_link2 = row['wikipedia_link_2']
        wiki_link3 = row['wikipedia_link_3']
        wiki_link4 = row['wikipedia_link_4']
        wiki_link5 = row['wikipedia_link_5']
        wiki_link6 = row['wikipedia_link_6']
        wiki_link7 = row['wikipedia_link_7']
        wiki_link8 = row['wikipedia_link_8']
        wiki_link9 = row['wikipedia_link_9']
        wiki_link10 = row['wikipedia_link_10']
        wiki_link11 = row['wikipedia_link_11+']
        wiki_links_str = row['wiki_links']
        reasoning_types = row['reasoning_types']

        
        generated_context = ""
        
        try:
            # The 'wiki_links' column is a string representation of a list, so we use ast.literal_eval
            url_list = ast.literal_eval(wiki_links_str)
            
            if isinstance(url_list, list):
                # Join URLs into a single string to pass to the plugin
                urls_as_query = " ".join(url_list)
                
                # Use the run function from the plugin to fetch content.
                # The function returns (new_query, completion_tokens). We only need the new_query.
                generated_context, _ = run_readurls(system_prompt="", initial_query=urls_as_query)
            else:
                logging.warning(f"Row {index}: 'wiki_links' is not a list. Skipping.")
            logging.info(f"Processing Question {questionNo} Completes")       
        except (ValueError, SyntaxError) as e:
            logging.error(f"Row {index}: Could not parse 'wiki_links' column. Error: {e}")
            generated_context = f"Error parsing URLs: {e}"
        
        # Append the new result to our list
        results.append({
            'QuestionNo': questionNo,
            'Prompt': prompt,
            'Answer': answer,
            'wiki_links': wiki_links_str,
            'generated_context': generated_context,
            'wikipedia_link_1': wiki_link1,
            'wikipedia_link_2': wiki_link2,
            'wikipedia_link_3': wiki_link3,
            'wikipedia_link_4': wiki_link4,
            'wikipedia_link_5': wiki_link5,
            'wikipedia_link_6': wiki_link6,
            'wikipedia_link_7': wiki_link7,
            'wikipedia_link_8': wiki_link8,
            'wikipedia_link_9': wiki_link9,
            'wikipedia_link_10': wiki_link10,
            'wikipedia_link_11+': wiki_link11,
            'reasoning_types': reasoning_types
        })

        # Save the updated list of results to the JSON file after each record
        save_results(output_path, results)

    logging.info(f"Processing complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
