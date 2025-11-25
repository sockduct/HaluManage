#! /usr/bin/env python
'''
Retrieve Hugging Face dataset for Google Frames Benchmark
'''


# Standard library:
import argparse
import json
from pathlib import Path
from pprint import pprint, pformat

# 3rd party libraries:
from datasets import (
    get_dataset_config_info, get_dataset_config_names, get_dataset_split_names, load_dataset
)


DATASET_ID = 'google/frames-benchmark'
DATASET_DIR = Path('data')
DATASET_FILE = 'frames.json'


def parse_args():
    parser = argparse.ArgumentParser(description='Retrieve FRAMES dataset.')
    parser.add_argument('--output_file', default=DATASET_FILE, type=Path)
    parser.add_argument('--output_path', default=DATASET_DIR, type=Path)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset_dir = args.output_path
    dataset_file = args.output_file

    # Dataset info:
    ds_info = get_dataset_config_info(DATASET_ID)
    print(f'Google FRAMES dataset info:\n{pformat(ds_info)}')

    # What configurations (sub-datasets) exist?
    configs = get_dataset_config_names(DATASET_ID)

    # Only default config:
    print(f'\nConfigs for dataset:  {configs}')

    # For each config, what splits exist?
    print('\nDataset splits for each config:')
    for cfg in configs:
        splits = get_dataset_split_names(DATASET_ID, cfg)
        # Only test split:
        print(f'* {cfg} -> splits: {splits}')

    # Load dataset:
    print('\nLoading dataset...')
    dataset = load_dataset(DATASET_ID)

    # Example dataset record:
    print('\nExample dataset record:')
    if 'test' in dataset:
        pprint(dataset['test'][0])

    # Save locally in JSON format:
    print(f'\nSaving locally in JSON format to {DATASET_DIR/DATASET_FILE}...')
    dataset_dir.mkdir(exist_ok=True)
    with open(dataset_dir/dataset_file, 'w', encoding='utf-8') as outfile:
        json.dump(dataset['test'].to_list(), outfile, indent=4)
