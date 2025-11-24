#! /usr/bin/env python
"""Sample a percentage of results-oracle-Qwen25-with-context.json and save sorted by Index."""

import argparse
import json
import math
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_file",
        type=Path,
        help="Path to input JSON list file to sample from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=20.0,
        help="Percentage of records to sample (0-100]. Default: 20",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.percent <= 0 or args.percent > 100:
        raise ValueError("--percent must be in the range (0, 100].")

    input_path = args.input_file
    percent_label = int(args.percent) if args.percent.is_integer() else args.percent
    output_path = input_path.with_name(f"{input_path.stem}-{percent_label}p{input_path.suffix}")

    with input_path.open() as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records.")

    sample_size = max(1, math.ceil(len(data) * (args.percent / 100.0)))
    sampled = random.sample(data, sample_size)
    sampled.sort(key=lambda x: x.get("Index", 0))

    with output_path.open("w") as f:
        json.dump(sampled, f, indent=2)

    print(f"Wrote {sample_size} records ({args.percent}% target) to {output_path}")


if __name__ == "__main__":
    main()
