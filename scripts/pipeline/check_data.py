#! /usr/bin/env python


import argparse
import json
from pathlib import Path

import tiktoken


DATASET_FILE = Path(__file__).parent/'data'/'frames_with_context.json'


def parse_args():
    parser = argparse.ArgumentParser(description='Check token size of each dataset record.')
    parser.add_argument('--dataset', default=DATASET_FILE, type=Path)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.dataset, 'r') as infile:
        data = json.load(infile)

    enc = tiktoken.get_encoding('cl100k_base')

    for question in data:
        index = question['Unnamed: 0'] + 1
        docs = question['wiki_content']
        content = [doc['article'] for doc in docs]
        articles = len(docs)
        size = sum(len(c) for c in content)
        tokens = sum(len(enc.encode(c)) for c in content)

        # print(f'{index:>3}:\t{articles} articles\t{size:,} characters\t{tokens:,} tokens')
        print(f'{index:>3}:\t{articles}\t{size:,}\t{tokens:,}')
        '''
        for i, c in enumerate(content, 1):
            print(f'==> Article {i} - {len(c):,} characters ({(len(c)/size) * 100:.1f}%)')
        '''
