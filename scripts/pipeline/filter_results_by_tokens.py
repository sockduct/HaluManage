#!/usr/bin/env python
"""Filter model result records by QuestionOutputTokens."""

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List, Tuple


MAX_OUTPUT_TOKENS = 4000

def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove records whose QuestionOutputTokens exceed a limit."
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        help="Path to the model results JSON file to process.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        help="Where to write the filtered results JSON.",
    )
    parser.add_argument(
        "--output_token_limit",
        type=int,
        default=MAX_OUTPUT_TOKENS,
        help=f"Maximum allowed QuestionOutputTokens value (default: {MAX_OUTPUT_TOKENS}).",
    )
    return parser.parse_args(args)


def load_records(path: Path) -> List[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected list at root of {path}, got {type(data).__name__}")
    return data


def filter_by_output_tokens(
    records: List[dict[str, Any]],
    limit: int,
) -> Tuple[List[dict[str, Any]], int]:
    kept: List[dict[str, Any]] = []
    removed = 0

    for record in records:
        tokens = record.get("QuestionOutputTokens")
        if isinstance(tokens, (int, float)) and tokens > limit:
            idx = record.get("Index", "<unknown>")
            print(
                f"Index {idx}: QuestionOutputTokens {tokens} exceeds limit {limit}; "
                "removing."
            )
            removed += 1
            continue

        kept.append(record)

    return kept, removed


def write_records(path: Path, records: List[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=True, indent=2)


def main() -> None:
    args = parse_args()
    records = load_records(args.input_file)
    filtered_records, removed_count = filter_by_output_tokens(
        records, args.output_token_limit
    )
    write_records(args.output_file, filtered_records)
    print(
        f"Completed. Removed {removed_count} record(s); "
        f"wrote {len(filtered_records)} record(s) to {args.output_file}."
    )


if __name__ == "__main__":
    main()
