# Mitigating Hallucinations in Complex Question-Answering

This project evaluates and compares various Retrieval-Augmented Generation (RAG) strategies to mitigate hallucinations in Large Language Models (LLMs) when performing complex, multi-hop question-answering tasks.

## Overview

Large Language Models often "hallucinate" or generate factually incorrect information, a problem that is particularly pronounced in tasks requiring multi-step reasoning across multiple documents. This project uses the **FRAMES dataset**, a benchmark designed to test factuality and reasoning, to assess the effectiveness of different RAG-based approaches.

The experiments compare Google's `gemini-2.5-flash-lite` model against OpenAI's `gpt-4o-mini` model, each deployed with different context-providing strategies. The**HaluManage proxy server**, is used to intermediate requests for the OpenAI evaluations.

## Dataset

The **FRAMES** dataset is used for all evaluations. It contains 824 challenging multi-hop questions that require synthesizing information from 2 to 15 different Wikipedia articles to arrive at a correct answer. This makes it an ideal benchmark for testing an LLM's ability to reason over a large context without generating incorrect facts.

## Setup and Execution

### 1. Initial Environment Setup

First, set up your local environment to run the evaluation scripts.

*   **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd HaluManage
    ```

*   **Create and activate a Python virtual environment:**
    ```bash
    # For Windows
    py -3 -m venv .venv
    .venv\scripts\activate

    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

*   **Install required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

*   **Configure Environment Variables:**
    The API key creation is not a part of the documentation.
    Create a `.env` file in the root of the project (`HaluManage/`) and add your API keys:
    ```env
    GOOGLE_API_KEY="your-google-api-key"
    OPENAI_API_KEY="your-openai-api-key"
    ```
### 2. Running Direct Evaluations (Gemini)

The Gemini evaluation scripts call the Google AI APIs directly or through a wrapper, without the need for the proxy server. The eval_results_* file will be created under root project folder.

*   **Run an Evaluation:**
    Execute one of the Gemini evaluation scripts from the `scripts/` directory.
    ```bash
    # Example for Full-Context RAG with CoT
    python scripts/eval_frames_benchmark_gemini_rag_full_context_cot.py
    ```

*   **Calculate Statistics:**
    If the summery statistics is not present in evluation results, run run the `calculate_summary_statistics.py` script on the generated JSON file to get the final accuracy report.
    ```bash
    # Example for the full-context results
    python scripts/calculate_summary_statistics.py results/eval_results_gemini-2.5-flash_lite_rag_full_context_cot.json
    ```

### 3. Running Evaluations via HaluManage Proxy (OpenAI)

The `eval_frames_benchmark_openai_rag_optillm.py` script is designed to send requests through the local HaluManage proxy server. The eval_results_* file will be created under root project folder.

*   **Step 1: Start the HaluManage Proxy Server**
    Open a new terminal, activate the virtual environment, and start the proxy server. (Note: The exact command may vary based on the server's implementation, but it will be similar to the following).
    ```bash
    # This command assumes the server is started via a 'server' module within the 'halumanage' package
    python -m halumanage.server OR
    python .\halumanage\server.py
    ```
    The server should now be running and listening on `http://localhost:8000`.

*   **Step 2: Run the OpenAI RAG Evaluation Script**
    In your original terminal, run the script. It is hardcoded to connect to the local proxy, as seen in the following lines from the script:
    ```python
    # HaluManage proxy will read the API key from environment variable.
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="http://localhost:8000/v1")
    ```
    Execute the script with the desired model:
    ```bash
    python scripts/eval_frames_benchmark_openai_rag_optillm.py --model gpt-4o-mini
    ```
    The script will now send its requests to your local proxy, which will then forward them to the OpenAI API.

## Methodologies & Experiments

Three primary methodologies were tested using `gemini-2.5-flash-lite`, and two were tested using `gpt-4o-mini`.

### 1. Gemini 2.5 Flash Lite Experiments

#### a. Naive Baseline
*   **Script:** `eval_frames_benchmark_gemini_naive_baseline.py`
*   **Description:** This approach provides the LLM with only the question and a list of Wikipedia URLs. The model is expected to answer based on its internal knowledge without the factual content from the articles, serving as a baseline to measure the impact of RAG.
*   **Output Files:**
    *   `eval_results_gemini-2.5-flash-lite_baseline.json`
    *   `eval_results_gemini-2.5-flash-lite_baseline.txt`

#### b. Memory Plugin RAG
*   **Script:** `eval_frames_benchmark_gemini_rag_memory.py`
*   **Description:** This script implements a selective, multi-step RAG pipeline. A `memory_plugin` first processes the source articles, distills what it determines to be "key facts," and then passes this smaller, condensed context to the LLM to generate the final answer. This tests the efficacy of an intermediary context-distillation component.
*   **Output Files:**
    *   `eval_results_gemini-2.5-flash-lite_rag_memory.json`
    *   `eval_results_gemini-2.5-flash-lite_rag_memory.txt`

#### c. Full-Context RAG with Chain-of-Thought (CoT)
*   **Script:** `eval_frames_benchmark_gemini_rag_full_context_cot.py`
*   **Description:** This approach leverages the large context window of the model by injecting the *entire* text of all relevant articles directly into the prompt. It is guided by a strong Chain-of-Thought (CoT) system prompt, instructing the model to reason step-by-step to synthesize the final answer.
*   **Output Files:**
    *   `eval_results_gemini-2.5-flash_lite_rag_full_context_cot.json`
    *   `eval_results_gemini-2.5-flash_lite_rag_full_context_cot.txt`

### 2. GPT-4o-mini (OptiLLM) Experiments

These experiments utilize `gpt-4o-mini` with a focus on comparing a naive baseline against a RAG approach intermediated by the HaluManage proxy.

#### a. Naive Baseline
*   **Script:** `eval_frames_benchmark_openai_naive_baseline_optillm.py`
*   **Description:** Similar to the Gemini baseline, this script provides the model with only the question, testing its pre-trained knowledge without immediate factual context.
*   **Output Files:**
    *   `eval_results_gpt-4o-mini_baseline_optillm.json`
    *   `eval_results_gpt-4o-mini_baseline_optillm.txt`

#### b. RAG via HaluManage Proxy
*   **Script:** `eval_frames_benchmark_openai_rag_optillm.py`
*   **Description:** This script implements a RAG approach where the request is sent to the local HaluManage proxy server. The proxy then orchestrates the retrieval and generation steps.
*   **Output Files:**
    *   `eval_results_gpt-4o-mini_rag_optillm.json`
    *   `eval_results_gpt-4o-mini_rag_optillm.txt`
    *   `eval_results_gpt-4o-mini_rag1_optillm.json`
    *   `eval_results_gpt-4o-mini_rag1_optillm.txt`

## Results Summary

The experiments show a significant performance difference between the various approaches, highlighting the effectiveness of providing full context with a Chain-of-Thought prompt.

| Model | Method | Accuracy |
| :--- | :--- | :--- |
| `gemini-2.5-flash-lite` | Naive Baseline | 27.55% |
| `gemini-2.5-flash-lite` | Memory Plugin RAG | 58.37% |
| **`gemini-2.5-flash-lite`** | **Full-Context + CoT** | **76.17%** |
| `gpt-4o-mini` | Naive Baseline (OptiLLM) | 32.04% |
| `gpt-4o-mini` | RAG (OptiLLM) | 59.95% |

The results clearly indicate that for `gemini-2.5-flash-lite`, providing the full, unaltered context and guiding the model with a Chain-of-Thought prompt yields the highest accuracy, outperforming both the naive baseline and the selective memory plugin approach.

## Dependencies - The other depedencies present in requirements.txt is used by HaluManage proxy and it's plugins and features.

*   `litellm`
*   `python-dotenv`
*   `tqdm`
*   `numpy`
*   `scikit-learn`
*   `openai`
*   `datasets`
*   `google-generativeai`
