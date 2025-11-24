#! /usr/bin/env python


import argparse
import ast
import json
### import sys
from pathlib import Path
from typing import Any, Iterable, List

'''
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
'''

from context_reducer import SemanticContextReducer


# Default is take 75% of this * 1,024 * 2.5 (Conservative words/token ratio)
MAX_CONTEXT_BYTES = 128
INPUT_FILE = 'data/frames_with_context.json'
OUTPUT_FILE = 'data/frames_with_context_{SIZE}.json'


def parse_wiki_links(raw_value: Any) -> List[str]:
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value]
    if isinstance(raw_value, str):
        try:
            parsed = ast.literal_eval(raw_value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (SyntaxError, ValueError):
            pass
    return []


def ensure_limit(record: dict, reducer: SemanticContextReducer, limit_bytes: int) -> bool:
    prompt = record.get('Prompt') or record.get('prompt') or ''
    context = record.get('Context', '')
    # links = parse_wiki_links(record.get("wiki_links", []))

    reduced_context = reducer.reduce(
        question=prompt,
        context=context,
        # wiki_links=links,
        limit_bytes=limit_bytes,
    )

    record['Context'] = reduced_context
    return len(reduced_context.encode('utf-8')) <= limit_bytes


def process_dataset(input_path: Path, output_path: Path, *,
                    limit_context: int=MAX_CONTEXT_BYTES, verbose: bool=True,
                    include_model_answer: bool=False) -> dict[str, Any]:
    reducer = SemanticContextReducer()
    limit_bytes = int(limit_context * 1024 * 0.75 * 2.5)

    if verbose:
        print(f"Limiting context to {limit_bytes:,} bytes - 75% of {limit_context} "
              "KB using 2.5 word/token ratio.")

    if verbose:
        print(f"Loading dataset {input_path}...")
    with input_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    total_items = len(dataset)
    trimmed_items = 0
    original_total = 0
    reduced_total = 0

    if verbose:
        print(f"Processing {total_items} records:", end='', flush=True)
    for idx, record in enumerate(dataset):
        if verbose:
            print('.', end='', flush=True)
        docs = record['wiki_content']
        articles = [doc['article'] for doc in docs]
        total_size = sum(len(article.encode('utf-8')) for article in articles)
        if include_model_answer:
            ma_size = len(record['ModelAnswer'].encode('utf-8'))
            total_size += ma_size

        abridged_articles = []
        for article in articles:
            original_size = len(article.encode('utf-8'))
            article_limit = int(original_size/total_size * limit_bytes)
            original_total += original_size

            article_record = {'Prompt': record['Prompt'], 'Context': article}
            if ensure_limit(article_record, reducer, article_limit):
                reduced_size = len(article_record.get('Context', '').encode('utf-8'))
                reduced_total += reduced_size
                if reduced_size < original_size:
                    trimmed_items += 1
            else:
                raise ValueError("Failed to enforce limit for a dataset row.")

            # Collect updated/abridged articles
            abridged_articles.append(article_record['Context'])

        if include_model_answer:
            ma_limit = int(ma_size/total_size * limit_bytes)
            ma_record = {'Prompt': record['Prompt'], 'Context': record['ModelAnswer']}
            ma_final_offset = record['ModelAnswer'].find('Final answer:')
            if ma_final_offset == -1:
                raise ValueError(f"Failed to find final answer in model answer for record {idx}.")
            ma_final_answer = record['ModelAnswer'][ma_final_offset:]
            if ensure_limit(ma_record, reducer, ma_limit):
                reduced_size = len(ma_record.get('Context', '').encode('utf-8'))
                reduced_total += reduced_size
                if reduced_size < ma_size:
                    trimmed_items += 1
            else:
                raise ValueError("Failed to enforce limit for a dataset row.")

        # Update dataset
        for i, doc in enumerate(record['wiki_content']):
            doc['article'] = abridged_articles[i]

        if include_model_answer:
            record['ModelAnswer'] = ma_record['Context'] + f'\n\n{ma_final_answer}'

    if verbose:
        print()

    if verbose:
        print(f"Saving trimmed dataset to {output_path}...")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=True, indent=2)

    return {
        "records": total_items,
        "trimmed_records": trimmed_items,
        "original_bytes": original_total,
        "reduced_bytes": reduced_total,
    }


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim FRAMES record contexts with "
                                     "OptiLLM semantic reducer.")
    parser.add_argument(
        "--input_file",
        type=Path,
        default=Path(INPUT_FILE),
        help=f"Path to the source JSON dataset (default: {INPUT_FILE}).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=OUTPUT_FILE,
        help=f"Where to write the trimmed JSON dataset (default: {OUTPUT_FILE}).",
    )
    parser.add_argument(
        "--include_model_answer",
        action="store_true",
        help="Include the model answer in the context trimming (default: False).",
    )
    parser.add_argument(
        "-l", "--limit-kb",
        type=int,
        default=MAX_CONTEXT_BYTES,
        help=f"Maximum size context window in KB (default: {MAX_CONTEXT_BYTES}).",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Overwrite the existing output file if present."
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    output_file = Path(args.output_file.format(SIZE=args.limit_kb))

    if not args.force and output_file.exists():
        print(f"Output file already exists: {output_file} - aborting...")
        return

    stats = process_dataset(args.input_file, output_file, limit_context=args.limit_kb,
                            include_model_answer=args.include_model_answer)

    print(
        "Processed {records} records; trimmed {trimmed_records}; size reduced "
        "from {original_bytes:,} to {reduced_bytes:,} bytes.".format(**stats)
    )


if __name__ == "__main__":
    main()
