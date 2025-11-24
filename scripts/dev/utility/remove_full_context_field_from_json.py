import json
import argparse

def remove_full_context(input_file: str, output_file: str) -> None:
    """
    Read a JSON file, remove entries with 'full_context' key, and write to a new JSON file.
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to the output JSON file
    """
    try:
        # Read the input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            # If data is a list of objects, remove 'full_context' from each
            cleaned_data = []
            for item in data:
                if isinstance(item, dict):
                    # Create a new dict without 'full_context'
                    cleaned_item = {k: v for k, v in item.items() if k != 'full_context'}
                    cleaned_data.append(cleaned_item)
                else:
                    cleaned_data.append(item)
        elif isinstance(data, dict):
            # If data is a single object, remove 'full_context' from it
            cleaned_data = {k: v for k, v in data.items() if k != 'full_context'}
        else:
            print("Unsupported JSON format")
            return
        
        # Write the cleaned data to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully processed {input_file}")
        print(f"Output written to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: '{input_file}' is not a valid JSON file.")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove 'full_context' field from a JSON file.")
    parser.add_argument("--input_file", type=str, default="input.json", help="Path to the input JSON file with pre-generated context.")
    args = parser.parse_args()

    input_file = args.input_file
    output_filename = f"output_without_full_context.json"

    # Specify your input and output file paths
    input_json = input_file  # Change this to your input file path
    output_json = output_filename  # Change this to your desired output file path
    
    remove_full_context(input_json, output_json)