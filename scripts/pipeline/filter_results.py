#!/usr/bin/env python
"""Filter model result records by specified criteria."""

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List, Tuple


MAX_OUTPUT_TOKENS = 4000

def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove records matching specified criteria or exceeding a specified limit."
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
    parser.add_argument(
        "--no_output_token_limit",
        action="store_true",
        default=False,
        help="Do not apply QuestionOutputTokens limit.",
    )
    parser.add_argument(
        "--prune_empty_grades",
        action="store_true",
        default=False,
        help="Prune records with empty GraderDecision value.",
    )
    return parser.parse_args(args)


def load_records(path: Path) -> List[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected list at root of {path}, got {type(data).__name__}")
    return data


def filter_by_output_tokens(records: List[dict[str, Any]], limit: int
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


def filter_by_grade(records: List[dict[str, Any]]) -> Tuple[List[dict[str, Any]], int]:
    kept: List[dict[str, Any]] = []
    removed = 0

    for record in records:
        if record.get("GraderDecision") == "":
            idx = record.get("Index", "<unknown>")
            print(f"Index {idx}: GraderDecision is empty; removing.")
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
    total_filtered = 0
    if not args.no_output_token_limit:
        records, removed_count = filter_by_output_tokens(
            records, args.output_token_limit
        )
        total_filtered += removed_count

    if args.prune_empty_grades:
        records, removed_count = filter_by_grade(records)
        total_filtered += removed_count

    write_records(args.output_file, records)
    print(
        f"Completed. Removed {total_filtered} record(s); "
        f"wrote {len(records)} record(s) to {args.output_file}."
    )


if __name__ == "__main__":
    main()
