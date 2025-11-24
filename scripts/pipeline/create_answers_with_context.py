#! /usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List, Sequence


def build_context(wiki_content: Sequence[dict[str, Any]]) -> str:
    """Format wiki_content docs into the context block used by the QA pipeline."""
    parts: List[str] = []
    for doc in wiki_content:
        parts.append(
            f'<<<BEGIN DOC [{doc["doc_index"]}]>>>\n'
            f'Source: Wikipedia | URL: {doc["url"]}\n'
            f'Title: {doc["title"]}\n'
            f'{doc["article"]}\n'
            f'<<<END DOC [{doc["doc_index"]}]>>>\n'
        )
    return "\n".join(parts)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach formatted context to QA records using a context file."
    )
    parser.add_argument(
        "--answer_file",
        type=Path,
        default=None,
        help="Path to the model output JSON file (list of question/answer dicts).",
    )
    parser.add_argument(
        "--context_file",
        type=Path,
        default=None,
        help="Path to the question context JSON file (list of dicts containing wiki_content).",
    )
    parser.add_argument(
        "--merged_file",
        type=Path,
        default=None,
        help=("Path to the merged model output/question context JSON file (list of dicts "
              "containing wiki_content)."),
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Where to write the augmented JSON file.",
    )
    parser.add_argument(
        "--preserve_context",
        action="store_true",
        help=("Preserve the original context in the output (default is to merge "
              "context docs together)."),
    )
    parser.add_argument(
        "--allow_missing_records",
        action="store_true",
        help="Skip missing records (default is to raise an error).",
    )
    return parser.parse_args(args)


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=4)


def augment_records(base_records: list[dict[str, Any]], context_records: list[dict[str, Any]],
                    preserve_context: bool=False) -> list[dict[str, Any]]:
    if len(base_records) != len(context_records):
        raise ValueError(
            f"Input and context files differ in length: {len(base_records)} vs {len(context_records)}"
        )

    answer_cols = ['Index', 'Prompt', 'Answer', 'ModelAnswer', 'GraderDecision',
                   'GraderExplanation', 'ReasoningTypes']

    augmented: list[dict[str, Any]] = []
    for idx, (base, ctx) in enumerate(zip(base_records, context_records)):
        record = {col: base[col] for col in answer_cols}

        wiki_content = ctx.get("wiki_content") or []
        if not isinstance(wiki_content, list):
            raise ValueError(f"Record {idx} context wiki_content is not a list.")
        if preserve_context:
            record['wiki_content'] = wiki_content
        else:
            record["Context"] = build_context(wiki_content)

        augmented.append(record)
    return augmented


def augment_trunc_records(base_records: list[dict[str, Any]], context_records: list[dict[str, Any]],
                          preserve_context: bool=False) -> list[dict[str, Any]]:
    answer_cols = ['Index', 'Prompt', 'Answer', 'ModelAnswer', 'GraderDecision',
                   'GraderExplanation', 'ReasoningTypes']

    augmented: list[dict[str, Any]] = []
    ctx = iter(context_records)
    for idx, base in enumerate(base_records):
        record = {col: base[col] for col in answer_cols}

        while (ctx_record := next(ctx)) and base['Index'] != ctx_record['Unnamed: 0']:
            pass

        wiki_content = ctx_record.get("wiki_content") or []
        if not isinstance(wiki_content, list):
            raise ValueError(f"Record {idx} context wiki_content is not a list.")
        if preserve_context:
            record['wiki_content'] = wiki_content
        else:
            record["Context"] = build_context(wiki_content)

        augmented.append(record)
    return augmented


def alt_augment_records(merged_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for record in merged_records:
        record['Context'] = build_context(record['wiki_content'])
        del record['wiki_content']
        augmented.append(record)
    return augmented

def main() -> None:
    args = parse_args()
    if args.answer_file and args.context_file:
        answer_records = load_json(args.answer_file)
        context_records = load_json(args.context_file)
        try:
            augmented = augment_records(answer_records, context_records, args.preserve_context)
        except ValueError as err:
            if args.allow_missing_records:
                augmented = augment_trunc_records(
                    answer_records, context_records, args.preserve_context
                )
            else:
                print(f'\nError:  {err}\n')
                return
    elif args.merged_file:
        merged_records = load_json(args.merged_file)
        augmented = alt_augment_records(merged_records)
    else:
        raise ValueError("Must specify either --answer_file and --context_file or --merged_file.")

    save_json(args.output_file, augmented)


if __name__ == "__main__":
    main()
