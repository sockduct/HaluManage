#! /usr/bin/env python


import argparse
import json
from pathlib import Path

import tiktoken


DATASET_FILE = Path(__file__).parent/'results'/'results-oracle-Qwen25-with-context.json'
MAX_TOKENS = 32000


def parse_args():
    parser = argparse.ArgumentParser(description='Check total context token size of each '
                                     'dataset record.')
    parser.add_argument('--dataset', default=DATASET_FILE, type=Path)
    parser.add_argument('--limit', default=MAX_TOKENS, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.dataset, 'r') as infile:
        data = json.load(infile)

    # Conservatively, don't send more than 75% of maximum input token size:
    max_tokens = int(args.limit * 0.75)

    enc = tiktoken.get_encoding('cl100k_base')

    over = {}
    for question in data:
        index = question['Index'] + 1
        # Don't need to count this:
        # index_tokens = len(enc.encode(index))
        prompt = question['Prompt']
        prompt_tokens = len(enc.encode(prompt))
        answer = question['Answer']
        # Don't need to count this:
        answer_tokens = len(enc.encode(answer))
        model_answer = question['ModelAnswer']
        model_answer_tokens = len(enc.encode(model_answer))
        context = question['Context']
        context_tokens = len(enc.encode(context))

        total_tokens = (
            # index_tokens + prompt_tokens + answer_tokens + model_answer_tokens + context_tokens
            prompt_tokens + model_answer_tokens + context_tokens
        )
        if total_tokens > max_tokens:
            print(f'Warning:  File Index {index - 1} over maximum input token size!')
            over[index] = total_tokens

        print(f'{index:>3}:\t{total_tokens:,}')

    if over:
        print(f'\n{len(over)} records over maximum input token size:')
        print('\n'.join(f'  {index}:  {tokens:,} tokens' for index, tokens in over.items()))
