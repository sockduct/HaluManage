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

# Add the project root to the Python path to allow importing from the halumanage package
# The script is in 'HaluManage/scripts', so we need to go up one level to 'HaluManage' to make the 'halumanage' package importable.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the run function from the readurls_plugin
from halumanage.plugins.readurls_plugin import run as run_readurls

# --- Configuration ---
INPUT_CSV_PATH = r'd:\vscworkspace\HaluManage\frames_benchmark_data.csv'
OUTPUT_JSON_PATH = r'd:\vscworkspace\HaluManage\frames_with_context_readurl.json'

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
    """
    Main function to read benchmark data, fetch URL content, and save to a JSON file.
    """
    logging.info(f"Loading data from {INPUT_CSV_PATH}...")
    
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        logging.error(f"Input file not found: {INPUT_CSV_PATH}")
        return

    # Check if output file exists to handle resuming progress
    results = load_existing_results(OUTPUT_JSON_PATH)
    processed_prompts = set()

    if results:
        logging.info(f"Output file found at {OUTPUT_JSON_PATH}. Resuming progress.")
        processed_prompts = {res['Prompt'] for res in results}
        logging.info(f"Found {len(processed_prompts)} already processed records to skip.")
    
    # Use tqdm to create a progress bar for the loop
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Fetching URL Content"):
        prompt = row['Prompt']
        
        # Skip if this prompt has already been processed and saved
        if prompt in processed_prompts:
            continue
            
        answer = row['Answer']
        wiki_links_str = row['wiki_links']
        
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
                
        except (ValueError, SyntaxError) as e:
            logging.error(f"Row {index}: Could not parse 'wiki_links' column. Error: {e}")
            generated_context = f"Error parsing URLs: {e}"
        
        # Append the new result to our list
        results.append({
            'Prompt': prompt,
            'Answer': answer,
            'wiki_links': wiki_links_str,
            'generated_context': generated_context,
        })

        # Save the updated list of results to the JSON file after each record
        save_results(OUTPUT_JSON_PATH, results)

    logging.info(f"Processing complete. Results saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
