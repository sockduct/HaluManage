import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optillm import SemanticContextReducer


MAX_CONTEXT_BYTES = 96 * 1024  # 96 KB


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
    prompt = record.get("Prompt") or record.get("prompt") or ""
    context = record.get("generated_context", "")
    links = parse_wiki_links(record.get("wiki_links", []))

    reduced_context = reducer.reduce(
        question=prompt,
        context=context,
        wiki_links=links,
        limit_bytes=limit_bytes,
    )

    record["generated_context"] = reduced_context
    return len(reduced_context.encode("utf-8")) <= limit_bytes


def process_dataset(
    input_path: Path,
    output_path: Path,
    *,
    limit_bytes: int = MAX_CONTEXT_BYTES,
) -> dict[str, Any]:
    reducer = SemanticContextReducer()

    with input_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    total_items = len(dataset)
    trimmed_items = 0
    original_total = 0
    reduced_total = 0

    for record in dataset:
        original_size = len(record.get("generated_context", "").encode("utf-8"))
        original_total += original_size

        if ensure_limit(record, reducer, limit_bytes):
            reduced_size = len(record.get("generated_context", "").encode("utf-8"))
            reduced_total += reduced_size
            if reduced_size < original_size:
                trimmed_items += 1
        else:
            raise ValueError("Failed to enforce limit for a dataset row.")

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=True, indent=2)

    return {
        "records": total_items,
        "trimmed_records": trimmed_items,
        "original_bytes": original_total,
        "reduced_bytes": reduced_total,
    }


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim FRAMES contexts with OptiLLM semantic reducer.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/frames_with_context_readurl.json"),
        help="Path to the source JSON dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/frames_with_context_readurl_optillm.json"),
        help="Where to write the trimmed JSON dataset.",
    )
    parser.add_argument(
        "--limit-kb",
        type=int,
        default=96,
        help="Maximum size per generated_context entry in KB (default: 96 KB).",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    limit_bytes = args.limit_kb * 1024

    stats = process_dataset(args.input, args.output, limit_bytes=limit_bytes)

    print(
        "Processed {records} records; trimmed {trimmed_records}; size reduced "
        "from {original_bytes:,} to {reduced_bytes:,} bytes.".format(**stats)
    )


if __name__ == "__main__":
    main()
