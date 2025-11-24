#!/usr/bin/env python3
"""Generate accuracy report from results.json with per-category breakdowns."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict

RESULTS_PATH = Path(__file__).with_name("results.json")

BASE_CATEGORIES = [
    "Multiple Constraints",
    "Numerical Reasoning",
    "Post Processing",
    "Tabular Reasoning",
    "Temporal Reasoning",
]

CANONICAL = {label.lower(): label for label in BASE_CATEGORIES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Report on FRAMES benchmark results.'
    )
    parser.add_argument(
        '-r', '--results_file',
        default=RESULTS_PATH,
        help=(f'Path to results file (default: {RESULTS_PATH})')
    )
    parser.add_argument(
        '-d', '--display',
        default=False,
        action='store_true',
        help='Display the report in the terminal (default: False)'
    )
    return parser.parse_args()


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def parse_reasoning_types(value: str | None) -> set[str]:
    if not value:
        return set()
    parsed: set[str] = set()
    for part in value.split("|"):
        normalized = part.strip().lower()
        if not normalized:
            continue
        canonical = CANONICAL.get(normalized)
        if canonical:
            parsed.add(canonical)
    return parsed


def update_stats(stats: Dict[str, Dict[str, int]], category: str, is_correct: bool) -> None:
    entry = stats[category]
    entry["total"] += 1
    if is_correct:
        entry["correct"] += 1


def format_accuracy(correct: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{(correct / total):.2%}"


def main(results_file: Path) -> str:
    solutions = load_results(results_file)
    total_samples = len(solutions)
    correct_answers = sum(1 for s in solutions if s.get("GraderDecision") == "TRUE")

    base_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    multi_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})

    for sample in solutions:
        is_correct = sample.get("GraderDecision") == "TRUE"
        reasoning_field = sample.get("ReasoningTypes") or sample.get("reasoning_types")
        types = parse_reasoning_types(reasoning_field)
        for category in types:
            update_stats(base_stats, category, is_correct)
        if len(types) >= 2:
            for category in types:
                update_stats(multi_stats, f"{category}+", is_correct)

    output = ''
    output += f"Total samples: {total_samples}\n"
    output += f"Correct answers: {correct_answers}\n"
    overall_accuracy = correct_answers / total_samples if total_samples else 0.0
    output += f"Accuracy: {overall_accuracy:.2%}\n"
    output += "\nPer-category accuracy:\n"
    for category in BASE_CATEGORIES:
        stats = base_stats[category]
        output += f"  {category}: {format_accuracy(stats['correct'], stats['total'])}\n"
    output += "\nMulti-category accuracy (counts overlap by design):\n"
    for category in BASE_CATEGORIES:
        label = f"{category}+"
        stats = multi_stats[label]
        output += f"  {label}: {format_accuracy(stats['correct'], stats['total'])}\n"

    return output


if __name__ == "__main__":
    args = parse_args()
    results_file = Path(args.results_file)
    results = main(results_file)

    if args.display:
        print(results)
