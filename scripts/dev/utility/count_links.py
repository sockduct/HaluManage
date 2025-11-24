import pandas as pd
import ast
import logging

# Configure basic logging to show any potential issues.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def count_total_links(csv_path: str) -> int:
    """
    Reads a CSV file, parses the 'wiki_links' column, and returns the
    total count of all individual Wikipedia links.

    Args:
        csv_path: The path to the input CSV file.

    Returns:
        The total number of links found across all rows.
    """
    try:
        # Read the CSV file. We use dtype=str and na_filter=False to handle
        # potential empty cells gracefully without creating NaN values.
        df = pd.read_csv(csv_path, dtype=str, na_filter=False)
        logging.info(f"Successfully loaded data from {csv_path}")
    except FileNotFoundError:
        logging.error(f"The file was not found at the specified path: {csv_path}")
        return 0
    except Exception as e:
        logging.error(f"An error occurred while reading the CSV file: {e}")
        return 0

    total_links_count = 0
    
    # Check if the 'wiki_links' column exists
    if 'wiki_links' not in df.columns:
        logging.error("The CSV file does not contain a 'wiki_links' column.")
        return 0

    def parse_and_count(row):
        index, links_string = row
        try:
            if not links_string:
                return 0
            url_list = ast.literal_eval(links_string)
            return len(url_list) if isinstance(url_list, list) else 0
        except (ValueError, SyntaxError):
            logging.error(f"Row {index}: Could not parse the string in 'wiki_links'. Content: {links_string}")
            return 0

    # Use sum() with a generator expression for a more concise way to count.
    # This iterates through each row, applies the parsing and counting logic, and sums the results.
    total_links_count = sum(parse_and_count(row) for row in df['wiki_links'].items())

    return total_links_count

if __name__ == "__main__":
    # Path to your CSV file
    csv_file_path = r'd:\vscworkspace\HaluManage\frames_benchmark_dataset.csv'
    
    # Get the total count
    total_count = count_total_links(csv_file_path)
    
    # Print the final result
    if total_count > 0:
        print(f"\nTotal number of Wikipedia links found in the 'wiki_links' column: {total_count}")
