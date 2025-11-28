# HaluManage

CIS 583 Term Project - Hallucination Detection and Mitigation

## Install Python 3.12

* For Windows download the appropriate [Python installer](https://www.python.org/downloads/release/python-31210/)
* For Linux, we recommend using [pyenv](https://github.com/pyenv/pyenv) and retrieving 3.12.11 using it:

```bash
curl -fsSL https://pyenv.run | bash

# Set up your shell environment for Pyenv
#
# Add the commands to ~/.bashrc by running the following in your terminal:
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc

# If you have ~/.profile, ~/.bash_profile or ~/.bash_login, add the commands there
# as well. If you have none of these, create a ~/.profile and add the commands there.
#
# to add to ~/.profile:
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init - bash)"' >> ~/.profile

# to add to ~/.bash_profile:
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init - bash)"' >> ~/.bash_profile

# Restart your shell for the PATH changes to take effect:
exec "$SHELL"

# Install Python build dependencies - for example with Ubuntu Linux:
sudo apt update
sudo apt install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl git \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Install Python 3.12.11:
pyenv install 3.12.11
```

## Install git

* For Windows download the [git client](https://git-scm.com/install/windows)
* For Linux use your package manager to install it:

```bash
# For example, install git client on Ubuntu Linux:
sudo apt update -y
# Note:  You may find it's already installed:
sudo apt install git -y
```

## Clone this repo

```bash
git clone https://github.com/sockduct/HaluManage.git halumanage

# cd into repo directory:
cd halumanage
```

## Create virtual environment (from within repo directory)

```bash
# Windows:
py -3.12 -m venv .venv

# Linux:
pyenv virtualenv 3.12 halumanage
```

## Activate virtual environment

```bash
# Windows:
.\.venv\Scripts\activate

# Linux:
pyenv activate halumanage
```

## Install the required Python dependencies

```bash
# Ensure the latest version of pip (Python package manager) is installed:
python -m pip install --upgrade pip

# Install Python dependencies needed by project:
pip install -r requirements.txt
```

## Change into the Pipeline scripts directory

```bash
# Note that all scripts have help to explain the script and its options
cd scripts/pipeline

# For example:
./getdata.py -h
```

### Note on running Python scripts

* These scripts were tested from PowerShell on Windows and bash on Linux
* If you have trouble running the scripts like this: `./script.py` try: `python script.py`
* Ensure you are using Python 3.12 when you run the scripts

### Retrieve FRAMES dataset from Hugging Face

```bash
# Retrieve FRAMES dataset and store in data/frames.json:
./getdata.py
```

### Create dataset with all Wikipedia article data present

```bash
# For each Wikipedia URL in the FRAMES record, retrieve the article content and
# embed it as a distinct document within the record.  This script also deals
# with observed issues and purges records referencing deleted articles.  The
# updated dataset is stored in data/frames_with_context.json:
#
# Note:  This is based on optillm/plugins/readurls_plugin.py from the OptiLLM repo:
./fetch_urls.py
```

### Processing a dataset with a model (LLM) - Requirements

* The eval_frames.py script uses [Open Router](https://openrouter.ai/docs/quickstart) to provide API access to hosted LLMs (models)
* You will need to [purchase Open Router API credits](https://openrouter.ai/settings/credits?ref=pricing-table-payg) to run this script
* In the directory where the eval_frames.py script is located, create an .env file with your API key:

```bash
OPENROUTER_API_KEY=sk-or-v1-sdlfkjslfksdjf...
```

```bash
# Test run to validate OPEN ROUTER API Key:
./eval_frames.py -l 1
```

### Process FRAMES dataset with naive prompting approach

```bash
# Ask question to model based on model's innate knowledge - no data supplied:
./eval_frames.py --dataset data/frames.json --mode naive --output_file results/results-naive-FlashLite-FlashLite-final.json --report_file results/results-naive-FlashLite-FlashLite-final.txt
```

### Process FRAMES dataset with oracle prompting approach

```bash
# Ask question to model supplying all referenced Wikipedia articles (so called
# oracle prompt) and instruct model to use chain-of-thought reasoning to reach
# answer.
./eval_frames.py --dataset data/frames_with_context.json --mode oracle --output_file results/results-oracle-FlashLite-FlashLite-final.json --report_file results/results-oracle-FlashLite-FlashLite-final.txt
```

### Remove resulting records with answer > 4k tokens and empty grades

```bash
# Optionally examine Question Output Tokens for records - note most records < 4k:
# Windows:
Select-String -pattern 'questionoutputtokens' -path 'results\results-oracle-FlashLite-FlashLite-final.json' | ForEach-Object { ($_.line -split ':')[1] -replace '^\s+', '' -replace ',\s$', '' } | Sort-Object { [int]$_ }

# Linux:
egrep -i 'questionoutputtokens' results/results-oracle-FlashLite-FlashLite-final.json | gawk -F ':' '{print $2}' | sed -e 's/^[[:space:]]*//' -e 's/,*$//' | sort -n

# Generate new dataset file with records > 4k removed:
# Also purge records where Grader Decision is empty
./filter_results.py --input_file results/results-oracle-FlashLite-FlashLite-final.json --output_file results/results-oracle-FlashLite-FlashLite-final-trunc.json --prune_empty_grades
```

### Trim/summarize FRAMES dataset to fit within 32k context window supported by Osiris

```bash
# We're selecting a target context window of 28k (reserving 4k for model answers
# to questions) - default input file is data/frames_with_context.json:
#
# Based on optillm/plugins/memory_plugin.py from the OptiLLM repo:
./trim_record_context.py -l 28
```

### Create dataset for Osiris based on model results and trimmed FRAMES dataset

```bash
./create_answers_with_context.py --answer_file results/results-oracle-FlashLite-FlashLite-final-trunc.json --context_file data/frames_with_context_28.json --output_file data/frames_with_ma_context_28.json --allow_missing_records
```

### Setup environment to run Osiris model

* Even the quantized Osiris model wants 24 GB of VRAM and a dedicated GPU
* We used DigitalOcean's Paperspace Gradient to run Osiris on
* We created a Notebook
  * Use Start from Scratch template
* We used a machine with 8 CPUs, 45 GB of RAM, an Nvidia A6000 GPU with 48 GB of VRAM
* We used an auto-shutdown timeout of 2 hours
* We used the data source to host the Osiris dataset
  * Gradient Dataset
  * Display Name:  frames-with-ma-context
  * Upload data/frames_with_ma_context_28.json from above
* Mount dataset
* Open in JupyterLab
  * Open a terminal

```bash
# Install prerequisites
sudo apt-get update
sudo apt-get install -y build-essential curl git \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev wget llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev

# Install pyenv
curl https://pyenv.run | bash

# Add to shell (e.g. ~/.bashrc)
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc

# Install and use Python 3.12
pyenv install 3.12.11
pyenv virtualenv 3.12.11 osiris
pyenv activate osiris

# Install Jupyter and create a kernel:
pip install jupyterlab ipykernel ipywidgets

# Setup Jupyter Kernel:
python -m ipykernel install --user --name "py312" --display-name "Python 3.12 (pyenv)"

# Setup repo:
git clone https://github.com/sockduct/HaluManage.git halumanage

# Setup environment:
pip install -r halumanage/requirements.txt

# Create data directory:
mkdir data

# Install notebook:
cp halumanage/scripts/pipeline/results/osiris_notebook.ipynb .
```

### Run Osiris Jupyter Notebook to analyze data and report on results

* From the DigitalOcean Paperspace Gradient Notebook JupyterLab console - open osiris_notebook.ipynb
* Ensure the kernel is set to Python 3.12 (pyenv) in the upper right hand corner
* Ensure the values are set correct and run all the cells (Run, Run All Cells)
* Download the results (/notebooks/data/results-oracle-FlashLite-FlashLite-final-trunc-processed.json)
* Optionally download the notebook

### Additional optional utilities:

* check_context.py - check total context token size of each dataset record
* check_data.py - check token size of each dataset record
* count_records.py - count records in JSON dataset file
* report_results.py - report on FRAMES benchmark results file
* sample_results.py - create a percentage random sample of records from a dataset file
