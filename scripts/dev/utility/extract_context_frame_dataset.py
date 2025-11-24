# This script reads a CSV file containing prompts, answers, and associated Wikipedia links,
# fetches the content of those links, and saves the augmented data to a new CSV file. The formatting of the data is not good for LLMs,
# but this is a simple implementation to extract context for the frames benchmark dataset.

# You should use the readurls_plugin for a better implementation which is adopted in extract_context_frame_dataset_readurl.py

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import ast
from tqdm import tqdm
import logging

# --- Configuration ---
INPUT_CSV_PATH = r'd:\vscworkspace\HaluManage\frames_benchmark_data.csv'
OUTPUT_CSV_PATH = r'd:\vscworkspace\HaluManage\frames_with_context.csv'
HALUMANAGE_VERSION = "1.0" # Version for User-Agent header

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Core Functions (adapted from readurls_plugin.py) ---

def fetch_webpage_content(url: str, max_length: int = 100000) -> str:
    """
    Fetches and cleans the text content of a given URL.
    """
    try:
        headers = {
            'User-Agent': f'HaluManage-Context-Capture/{HALUMANAGE_VERSION}'
        }
        # Using verify=True is the default and recommended for security.
        response = requests.get(url, headers=headers, timeout=15, verify=True)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "aside"]):
            script.decompose()
        
        # Get text from main content tags first
        main_content = soup.select_one('article, main, div[role="main"], .main-content')
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            # Fallback to body if no main content tags are found
            body = soup.find('body')
            if body:
                text = body.get_text(separator=' ', strip=True)
            else:
                return f"Error: Could not find body content for {url}"

        # Clean up whitespace and remove bracketed citations (e.g., [1], [edit])
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\[\d+\]|\[edit\]', '', text)
        
        # Truncate to max_length
        if len(text) > max_length:
            text = text[:max_length] + '...'
        
        return text
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching content for {url}: {e}")
        return f"Error fetching content: {e}"
    except Exception as e:
        logging.error(f"An unexpected error occurred for {url}: {e}")
        return f"Unexpected error: {e}"

def main():
    """
    Main function to read the benchmark data, fetch URL content, 
    and save the augmented data to a new CSV.
    """
    logging.info(f"Loading data from {INPUT_CSV_PATH}...")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        logging.error(f"Input file not found: {INPUT_CSV_PATH}")
        return

    results = []
    
    # Use tqdm to create a progress bar for the loop
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Fetching URL Content"):
        prompt = row['Prompt']
        answer = row['Answer']
        wiki_links_str = row['wiki_links']
        
        generated_context = ""
        
        try:
            # The 'wiki_links' column is a string representation of a list, so we use ast.literal_eval
            url_list = ast.literal_eval(wiki_links_str)
            
            if isinstance(url_list, list):
                content_parts = []
                for url in url_list:
                    logging.info(f"Fetching content for: {url}")
                    content = fetch_webpage_content(url)
                    content_parts.append(f"--- Content from {url} ---\n{content}\n")
                
                generated_context = "\n".join(content_parts)
            else:
                logging.warning(f"Row {index}: 'wiki_links' is not a list. Skipping.")
                
        except (ValueError, SyntaxError) as e:
            logging.error(f"Row {index}: Could not parse 'wiki_links' column. Error: {e}")
        
        results.append({
            'Prompt': prompt,
            'Answer': answer,
            'wiki_links': wiki_links_str,
            'generated_context': generated_context
        })

    logging.info("Saving results to new CSV file...")
    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_CSV_PATH, index=False)
    logging.info(f"Successfully created {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()
