import pandas as pd
import ast
import requests
from bs4 import BeautifulSoup
import tiktoken
import logging
import time

# --- Configuration ---
DATASET_FILE = "frames_benchmark_dataset.csv"
TARGET_URL_COUNT = 11

# Set up logging to see progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_links_list(links_str: str) -> list:
    """Safely parses a string representation of a list."""
    try:
        links_list = ast.literal_eval(links_str)
        if isinstance(links_list, list):
            return links_list
        return []
    except (ValueError, SyntaxError):
        return []

def fetch_and_clean_text(url: str) -> str:
    """
    Fetches content from a URL, parses it with BeautifulSoup,
    and returns only the text from the main content paragraphs.
    """
    if not url or not url.startswith('http'):
        logging.warning(f"Skipping invalid URL: {url}")
        return ""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        # Raise an exception for bad status codes (like 404 or 403)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the main content body of a Wikipedia page
        content_div = soup.find(id='bodyContent')
        
        if content_div:
            # Extract text primarily from <p> (paragraph) tags
            paragraphs = content_div.find_all('p', recursive=True)
            page_text = "\n".join([p.get_text() for p in paragraphs])
            return page_text
        else:
            # Fallback if the structure is unexpected
            logging.warning(f"Could not find 'bodyContent' div for {url}. Falling back to all text.")
            return soup.get_text()
            
    except requests.RequestException as e:
        logging.error(f"Could not fetch URL {url}: {e}")
        return ""
    except Exception as e:
        logging.error(f"Error parsing URL {url}: {e}")
        return ""

def count_tokens(text: str, encoding) -> int:
    """Counts tokens in a string using the provided tiktoken encoding."""
    return len(encoding.encode(text))

def main():
    logging.info("Starting token count analysis for local execution...")
    
    # 1. Load the tokenizer
    try:
        # cl100k_base is the standard for gpt-4, gpt-3.5-turbo, 
        # and a strong proxy for Gemini models.
        encoding = tiktoken.get_encoding("cl100k_base")
        logging.info("Loaded 'cl100k_base' tokenizer.")
    except Exception as e:
        logging.error(f"Could not load tiktoken encoding: {e}")
        return

    # 2. Load and filter dataset
    try:
        df = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        logging.error(f"Error: The file '{DATASET_FILE}' was not found in this directory.")
        return

    # Clean and parse the 'wiki_links' column
    df['wiki_links'] = df['wiki_links'].fillna('[]')
    df['links_list'] = df['wiki_links'].apply(parse_links_list)
    df['num_links'] = df['links_list'].apply(len)

    # Filter for rows with the exact target URL count
    target_rows = df[df['num_links'] == TARGET_URL_COUNT]

    if target_rows.empty:
        logging.warning(f"No questions found with exactly {TARGET_URL_COUNT} links.")
        return

    logging.info(f"Found {len(target_rows)} question(s) with {TARGET_URL_COUNT} links.")
    
    grand_total_tokens = 0
    
    # 3. Process each target row (question)
    for index, row in target_rows.iterrows():
        print("\n" + "="*80)
        print(f"Processing Question (Index {index})")
        print(f"Question: \"{row['Prompt']}\"")
        print("="*80)
        
        urls_to_fetch = row['links_list']
        question_total_tokens = 0
        page_token_counts = {}

        # 4. Fetch, clean, and count tokens for each URL
        for i, url in enumerate(urls_to_fetch, 1):
            logging.info(f"  ({i}/{len(urls_to_fetch)}) Fetching: {url}")
            
            page_text = fetch_and_clean_text(url)
            
            if page_text:
                tokens = count_tokens(page_text, encoding)
                page_token_counts[url] = tokens
                question_total_tokens += tokens
                logging.info(f"    -> Fetched and Cleaned. Token Count: {tokens:,}")
            else:
                page_token_counts[url] = 0
                logging.warning(f"    -> Failed to fetch or no text found for this URL.")
            
            # Brief pause to be polite to Wikipedia's servers
            time.sleep(0.5) 

        # 5. Print the report for this question
        print("\n--- Token Count Report ---")
        print(f"Question: \"{row['Prompt']}\"")
        print("\nIndividual Page Token Counts (using 'cl100k_base'):")
        
        for url, count in page_token_counts.items():
            print(f"- {count:7,} tokens | {url}")
            
        print("------------------------")
        print(f"Total Combined Token Size: {question_total_tokens:,}")
        print("------------------------")
        
        grand_total_tokens += question_total_tokens

    # 6. Print the final grand total
    print("\n" + "="*80)
    print("========= FINAL SUMMARY =========")
    print(f"Processed {len(target_rows)} question(s) with {TARGET_URL_COUNT} URLs.")
    print(f"Grand Total Combined Tokens: {grand_total_tokens:,}")
    print("=================================")

if __name__ == "__main__":
    main()