# HaluManage
CIS 583 Term Project - Hallucination Detection and Mitigation

## Install Python 3.12.11

## Install git

## Clone this repo

* cd into repo directory

## Create virtual environment (from within repo directory)

### Windows:
py -3.12 -m venv .venv

### Linux:
...

## Activate virtual environment - Windows:
.\.venv\Scripts\activate

## Activate virtual environment - Linux:
...

## Install the dependencies specified in requirements.txt
pip install -r requirements.txt

## Upgrade new pip release
python -m pip install --upgrade pip

## Python scripts:

cd scripts/pipeline

### How to generate data/frames.json?
#### Looks like need basic script...

### Generate data/frames_with_context.json:
./fetch_urls.py

### Process FRAMES dataset with naive prompting approach:
./eval_frames.py --dataset data/frames.json --mode naive --output_file results/results-naive-FlashLite-FlashLite-final.json --report_file results/results-naive-FlashLite-FlashLite-final.txt

### Process FRAMES dataset with oracle prompting approach:
./eval_frames.py --dataset data/frames_with_context.json --mode oracle --output_file results/results-oracle-FlashLite-FlashLite-final.json --report_file results/results-oracle-FlashLite-FlashLite-final.txt

### Remove resulting records with answer > 4k tokens:
./filter_results_by_tokens.py --input_file results/results-oracle-FlashLite-FlashLite-final.json --output_file results/results-oracle-FlashLite-FlashLite-final-trunc.json

### Trim/summarize FRAMES dataset (data/frames_with_context.json) to fit within 32k context window supported by Osiris (reserve 4k for model answer):
./trim_record_context.py -l 28

### Create dataset for Osiris based on model results and trimmed FRAMES dataset:
./create_answers_with_context.py --answer_file results/results-oracle-FlashLite-FlashLite-final-trunc.json --context_file data/frames_with_context_28.json --output_file data/frames_with_ma_context_28.json --allow_missing_records

### Upload data/frames_with_ma_context_28.json to Osiris VM
### Run Osiris Jupyter Notebook to analyze data and report on results

### Explain optional utilities:

* check_context.py
* check_data.py
* count_records.py
* report_results.py
* sample_results.py
