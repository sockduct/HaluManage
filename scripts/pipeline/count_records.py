#!/usr/bin/env python
"""Count records in a JSON list of dicts."""

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count records in a JSON file containing a list of dicts."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the JSON file to inspect.",
    )
    return parser.parse_args(args)


def load_records(path: Path) -> List[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected list at root of {path}, got {type(data).__name__}")
    return data


def main() -> None:
    args = parse_args()
    records = load_records(args.input_file)
    print(f"{args.input_file} contains {len(records)} records")


if __name__ == "__main__":
    main()
