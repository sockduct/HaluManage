#! /usr/bin/env python
'''
Asynchronous Proof of Concept for FRAMES dataset
* Query specified model with Open Router API Endpoint
* Grade response using Google's Gemini 1.5 Pro model (Open Router)
* Report results
'''


# Standard Library:
import argparse
import asyncio
import json
import os
from pathlib import Path
import time

# 3rd party libraries:
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError

# Local libraries:
from report_results import main as supplemental_report


# Set your API key / endpoint
API_BASE = 'https://openrouter.ai/api/v1'
API_PROVIDER = 'OPENROUTER'
#
# Identifier used by provider:
MODEL1a = 'google/gemma-3-27b-it:free'
MODEL1b = 'google/gemma-3-27b-it'  # Watch cost incurred...
MODEL2 = 'google/gemma-2-9b-it:free'
MODEL3 = 'google/gemma-2-27b-it'  # About 7x more expensive then gemma-3-27b-it!
MODEL4 = 'google/gemini-pro-1.5'  # Used as grading model
MODEL5 = 'google/gemini-pro-2.5'  # Powerful but expensive!
MODEL6 = 'openai/gpt-4o-mini'  # Backup instead of Gemini Pro for grading
MODEL7 = 'google/gemini-2.5-flash-lite'  # Used instead of Gemini Pro for grading
MODEL8 = 'google/gemini-2.0-flash-001'
MODEL9 = 'qwen/qwen-2.5-7b-instruct'  # Model Osiris is based on
#
# Other models considered:
# gpt-4.1-mini - too expensive
# gpt-5-mini - too expensive
# gemini-*-pro - too expensive
#
#
NAIVE_Q_PROMPT = '''You are a helpful and accurate assistant.
Please answer the following question based on your knowledge.
Question: {question}
Answer:'''
#
# Good:
ORACLE_Q_PROMPT = '''Background information:\n{background_info}\n\n
You are a helpful and accurate assistant.
Please answer the following question based on your knowledge and the background information.
Question: {question}
Answer:'''
#
# Better but now some output tokens are too long for Osiris:
ORACLE_Q2_PROMPT = '''You are an expert **Multi-Hop Reasoner** and Final Answer Synthesizer.

Your sole purpose is to answer the complex, multi-hop question using only the provided background information.

### Rules for Answering:
1.  **Chain-of-Thought (CoT):** You must first provide a clear, numbered, step-by-step chain of reasoning that lists each doc title used, and logically connects the facts within and between docs to reach the final answer.
2.  **Final Answer:** After your reasoning, provide the final answer as a single, concise statement under the heading "Final Answer:".


Background information:
{background_info}


Multi-hop Question: {question}'''
#
# Revised to do chain-of-thought with concise final answer - even worse results:
ORACLE_Q3_PROMPT = '''You are an expert **Multi-Hop Reasoner** and Final Answer Synthesizer.

Your sole purpose is to answer the complex, multi-hop question using only the provided background information.

### Rules for Answering:
1. **Chain-of-Thought (CoT):** First, provide a clear, numbered step-by-step chain of reasoning **summarizing the essential facts** from each relevant document (mention each doc title). **Keep each step concise, focusing only on key information** that logically connects to the next step, to reach the final answer.
2. **Final Answer:** After the reasoning, provide the final answer as a single, concise statement under the heading "Final Answer:".
3. **Conciseness:** *Ensure your entire response (Chain-of-Thought + Final Answer) is under 4000 tokens.* Avoid unnecessary details or verbatim long quotes - **focus on high-level reasoning** that can be audited for correctness.


Background information:
{background_info}


Multi-hop Question: {question}'''
#
GRADER_PROMPT = '''===Task===
I need your help in evaluating an answer provided by an LLM against a ground truth
answer. Your task is to determine if the ground truth answer is present in the LLM's response.
Please analyze the provided data and make a decision.

===Instructions===
1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer".
2. Consider the substance of the answers - look for equivalent information or correct answers. Do
not focus on exact wording unless the exact wording is crucial to the meaning.
3. Your final decision should be based on whether the meaning and the vital facts of the "Ground
Truth Answer" are present in the "Predicted Answer:"

===Input Data===
- Question: {question}
- Predicted Answer: {LLM_response}
- Ground Truth Answer: {ground_truth_answer}

===Output Format===
Provide your final evaluation in the following format:
"Explanation:" (How you made the decision?)
"Decision:" ("TRUE" or "FALSE" )

Please proceed with the evaluation. '''
#
DATASET = 'google/frames-benchmark'
#
LIMITER: int|None = None
#
# Maximum concurrent requests:
GRADER_GATE = asyncio.Semaphore(15)
Q_GATE = asyncio.Semaphore(15)
MAX_RETRIES = 3
#
DATASET_FILE = Path(__file__).parent/'data'/'frames_with_context.json'
OUTPUT_FILE = 'results.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FRAMES benchmark queries and grading asynchronously."
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging (default: False)'
    )
    parser.add_argument(
        '--dataset',
        default='default',
        help=('Defaults to FRAMES dataset or specify compatible dataset with '
              f'context (e.g., {DATASET_FILE})')
    )
    parser.add_argument(
        '-l', '--limit',
        type=int,
        default=LIMITER,
        help='Limit number of records to process (default: no limit)'
    )
    parser.add_argument(
        '--mode',
        choices=['naive', 'oracle'],
        default='naive',
        help='Mode to run: naive (no context) or oracle (full context) (default: naive)'
    )
    parser.add_argument(
        '--output_file',
        default=OUTPUT_FILE,
        help=f"Path to save results (default: {OUTPUT_FILE})"
    )
    parser.add_argument(
        '--report_file',
        default=None,
        help="Path to save report (default: None)"
    )
    parser.add_argument(
        '--question_model',
        default=MODEL7,
        help=f"LLM model to use for questions (default: {MODEL7})"
    )
    parser.add_argument(
        '--grader_model',
        default=MODEL7,
        help=f"LLM model to use for grading (default: {MODEL7})"
    )
    return parser.parse_args()


async def ask_1q(question: str, *, client: AsyncOpenAI, temperature: float=0.0,
                 max_completion_tokens: int|None=None, model: str=MODEL7,
                 prompt_template: str=NAIVE_Q_PROMPT, prompt_vars: dict[str, str]|None=None,
                 verbose: bool=False) -> tuple[str, dict[str, str]]:
    '''Send one question using the provided prompt template and return the
       model's answer.'''
    variables = {'question': question}
    if prompt_vars:
        variables |= prompt_vars
    prompt = prompt_template.format(**variables)
    if verbose:
        print(f'User Prompt:  {prompt}')

    async with Q_GATE:
        delay = 1.0
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    extra_headers = {},
                    extra_body = {},
                    model = model,
                    messages = [
                        ### Make this dynamic depending on model???
                        # Removing the system prompt for gemma-3-27b-it - model
                        # doesn't support it:
                        # {'role': 'system', 'content': 'You are a highly competent assistant.'},
                        {'role': 'user', 'content': prompt},
                    ],
                    temperature = temperature,  # optional: 0 temperature for deterministic answers
                    max_completion_tokens = max_completion_tokens,
                    n = 1,
                )
                break
            except RateLimitError:
                if attempt == MAX_RETRIES - 1:
                    raise
                # If we hit rate limit, wait briefly and retry once
                await asyncio.sleep(delay)
                delay *= 2

    if response.choices[0].finish_reason == 'length':
        print('WARNING: ask_1q() - The model\'s response was truncated due to reaching '
              f'the maximum alloted tokens. ({response.usage.completion_tokens:,})')

    token_stats = {
        'input_tokens': response.usage.prompt_tokens,
        'output_tokens': response.usage.completion_tokens
    }
    return response.choices[0].message.content.strip(), token_stats


async def grade_response(*, question: str, actual_answer: str, model_answer: str,
                         client: AsyncOpenAI, temperature: float=0.3,
                         max_completion_tokens: int|None=None, model: str=MODEL7,
                         verbose: bool=False) -> dict[str, str|int]:
    '''
    Prompt parameters:
    Question: {question}
    Predicted Answer: {LLM_response}
    Ground Truth Answer: {ground_truth_answer}

    Note:  In FRAMES paper, they used Google Gemini Pro 1.5 as the grading model

    Look at eval_frames_benchmark.py in the codelion/optillm repo for example
    evalu against FRAMES dataset.  Per this example, setting temperature=0.3,
    and max_completion_tokens=300.
    '''
    prompt = GRADER_PROMPT.format(question=question, LLM_response=model_answer,
                                  ground_truth_answer=actual_answer)

    async with GRADER_GATE:
        delay = 1.0
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    extra_headers = {},
                    extra_body = {},
                    model = model,
                    messages = [
                        {'role': 'system', 'content': 'You are a helpful and accurate assistant.'},
                        {'role': 'user', 'content': prompt},
                    ],
                    temperature = temperature,
                    max_completion_tokens = max_completion_tokens,
                    n = 1,
                )
                break
            except RateLimitError:
                if attempt == MAX_RETRIES - 1:
                    raise
                # If we hit rate limit, wait briefly and retry once
                await asyncio.sleep(delay)
                delay *= 2

    ### Modifications required for async???
    ### Look at Codex suggestions...
    if response.choices[0].finish_reason == 'length':
        print('WARNING: grade_response() - The model\'s response was truncated due to '
              'reaching the maximum alloted tokens.')

    eval_text = response.choices[0].message.content.strip()

    # Extract the decision and explanation
    lines = eval_text.split('\n')
    ### Fix this to be boolean:
    decision = 'FALSE'
    explanation = ''
    for line in lines:
        if line.startswith('Decision:'):
            decision = line.split(':')[1].strip().upper()
        elif line.startswith('Explanation:'):
            explanation = line.split(':', 1)[1].strip()

    return {
        'decision': decision,
        'explanation': explanation,
        'input_tokens': response.usage.prompt_tokens,
        'output_tokens': response.usage.completion_tokens
    }


async def process_question(*, index: int, question: dict[str, str], client: AsyncOpenAI,
                           mode: str='naive', status_queue: asyncio.Queue|None=None,
                           question_model: str=MODEL7, grader_model: str=MODEL7,
                           verbose: bool=False) -> dict[str, str]:
    if status_queue:
        await status_queue.put(('ask_start', index))

    if mode == 'naive':
        model_answer, token_stats = await ask_1q(question['Prompt'], client=client, verbose=verbose)
    elif mode == 'oracle':
        question['background_info'] = ''
        for doc in question['wiki_content']:
            question['background_info'] += (
                f'<<<BEGIN DOC [{doc["doc_index"]}]>>>\n'
                f'Source: Wikipedia | URL: {doc["url"]}\n'
                f'Title: {doc["title"]}\n'
                f'{doc["article"]}\n'
                f'<<<END DOC [{doc["doc_index"]}]>>>\n\n'
            )
        model_answer, token_stats = await ask_1q(
            question['Prompt'], client=client, prompt_template=ORACLE_Q2_PROMPT,
            prompt_vars={'background_info': question['background_info']}, model=question_model,
            verbose=verbose
        )
    else:
        raise ValueError(f'Unknown mode: {mode}')

    if status_queue:
        await status_queue.put(('ask_done', index))

    if status_queue:
        await status_queue.put(('grade_start', index))

    grade = await grade_response(question=question['Prompt'], actual_answer=question['Answer'],
                                 model_answer=model_answer, client=client, model=grader_model)
    if status_queue:
        await status_queue.put(('grade_done', index))

    return {
        'Index': question.get('Unnamed: 0', index),
        'Prompt': question['Prompt'],
        'Answer': question['Answer'],
        'ModelAnswer': model_answer,
        'QuestionInputTokens': token_stats['input_tokens'],
        'QuestionOutputTokens': token_stats['output_tokens'],
        'GraderDecision': grade['decision'],
        'GraderExplanation': grade['explanation'],
        'GraderInputTokens': grade['input_tokens'],
        'GraderOutputTokens': grade['output_tokens'],
        'ReasoningTypes': question.get('reasoning_types', 'N/A'),
    }


async def status_monitor(total: int, queue: asyncio.Queue) -> None:
    completed = 0
    while True:
        event, index = await queue.get()
        if event == 'done':
            queue.task_done()
            break
        question_number = index + 1 if index is not None else '?'
        if event == 'ask_start':
            print(f'Starting ask_1q for question {question_number}...')
        elif event == 'ask_done':
            print(f'Completed ask_1q for question {question_number}.')
        elif event == 'grade_start':
            print(f'Starting grade_response for question {question_number}...')
        elif event == 'grade_done':
            completed += 1
            print(f'Completed grade_response for question {question_number} '
                  f'({completed}/{total}).')
        queue.task_done()


async def main(qa: list[dict[str, str]], client: AsyncOpenAI, *, limit: int|None=None,
               mode: str='naive', question_model: str=MODEL7, grader_model: str=MODEL7,
               verbose: bool=False) -> list[dict[str, str]]:
    total = min(len(qa), limit) if limit else len(qa)
    status_queue: asyncio.Queue = asyncio.Queue()
    monitor = asyncio.create_task(status_monitor(total, status_queue))

    tasks = []
    for index, question in enumerate(qa):
        if limit and index >= limit:
            break
        tasks.append(asyncio.create_task(
            process_question(index=index, question=question, client=client,
                             status_queue=status_queue, mode=mode, question_model=question_model, grader_model=grader_model, verbose=verbose)
        ))

    try:
        solutions = await asyncio.gather(*tasks)
    finally:
        await status_queue.put(('done', None))
        await status_queue.join()
        await monitor

    return solutions


def display(solutions: list[dict[str, str]]) -> None:
    for i, solution in enumerate(solutions):
        print(f'Index {solution["Index"]}, Question ({i + 1}):')
        print(f'Question:  {solution["Prompt"]}')
        print(f'Model Answer:  {solution["ModelAnswer"]}')
        print(f'Actual Answer:  {solution["Answer"]}')
        print(f'Grader Decision:  {solution["GraderDecision"]}')
        print(f'Grader Explanation:  {solution["GraderExplanation"]}')
        print(f'Reasoning Types:  {solution["ReasoningTypes"]}')
        print('\n')


def get_client() -> AsyncOpenAI:
    '''Model provider'''
    load_dotenv()
    api_key = os.getenv(f'{API_PROVIDER}_API_KEY')

    return AsyncOpenAI(
        base_url = API_BASE,
        api_key = api_key,
    )


def report(solutions: list[dict[str, str]], *, display: bool=True,
           results_file: Path|None=None, report_file: str|None=None, mode: str='naive',
           question_model: str=MODEL7, grader_model: str=MODEL7) -> None:
    '''Calculate and print summary statistics

       Types:
       * Multiple Constraints
       * Numerical Reasoning
       * Post Processing
       * Tabular Reasoning
       * Temporal Reasoning

        'QuestionInputTokens': token_stats['input_tokens'],
        'QuestionOutputTokens': token_stats['output_tokens'],
        'GraderInputTokens': grade['input_tokens'],
        'GraderOutputTokens': grade['output_tokens'],

    '''
    total_samples = len(solutions)
    correct_answers = sum(1 for s in solutions if s['GraderDecision'] == 'TRUE')
    accuracy = correct_answers/total_samples
    total_q_intokens = sum(int(s['QuestionInputTokens']) for s in solutions)
    total_q_outtokens = sum(int(s['QuestionOutputTokens']) for s in solutions)
    total_grader_intokens = sum(int(s['GraderInputTokens']) for s in solutions)
    total_grader_outtokens = sum(int(s['GraderOutputTokens']) for s in solutions)
    total_intokens = total_q_intokens + total_grader_intokens
    total_outtokens = total_q_outtokens + total_grader_outtokens
    total_tokens = total_intokens + total_outtokens

    # Output:
    output = f'FRAMES Dataset Evaluation Report\n\nMode:  {mode}\n'
    output += f'Question Model:  {question_model}\n'
    output += f'Grader Model:  {grader_model}\n\n'

    output += f'Total samples: {total_samples}\n'
    output += f'Correct answers: {correct_answers}\n'
    output += f'Accuracy: {accuracy:.2%}\n'

    # Print accuracy by reasoning type
    reasoning_types = {s['ReasoningTypes'] for s in solutions}
    for rt in reasoning_types:
        rt_samples = [s for s in solutions if s['ReasoningTypes'] == rt]
        rt_correct = sum(1 for r in rt_samples if r['GraderDecision'] == 'TRUE')
        rt_accuracy = rt_correct / len(rt_samples)
        output += f'Accuracy for {rt}: {rt_accuracy:.2%}\n'

    output += f'\nTotal question input tokens: {total_q_intokens:,}\n'
    output += f'Total question output tokens: {total_q_outtokens:,}\n'
    output += f'Total grader input tokens: {total_grader_intokens:,}\n'
    output += f'Total grader output tokens: {total_grader_outtokens:,}\n'
    output += f'Total input tokens: {total_intokens:,}\n'
    output += f'Total output tokens: {total_outtokens:,}\n'
    output += f'Total tokens: {total_tokens:,}\n'

    if results_file:
        output += '\nSupplemental:\n'
        output += supplemental_report(results_file)

    if display:
        print(f'\n{output}')

    if report_file:
        with open(report_file, 'w') as outfile:
            outfile.write(output)


def save(solutions: list[dict[str, str]], *, output_file: str=OUTPUT_FILE) -> None:
    with open(output_file, 'w') as f:
        json.dump(solutions, f, indent=4)


def get_dataset(*, dataset: str=DATASET, columns: list[str]|None=None,
                verbose: bool=False) -> list[dict[str, str]]:
    # Optionally strip to essential columns:
    # columns = ['Unnamed: 0', 'Prompt', 'Answer', 'reasoning_types']

    if verbose:
        print(f'Loading dataset {dataset}...')

    data = load_dataset(dataset)

    if columns:
        return data['test'].select_columns(columns).to_list()
    else:
        return data['test'].to_list()


if __name__ == '__main__':
    args = parse_args()
    verbose = args.verbose
    dataset = args.dataset
    limit = args.limit
    mode = args.mode
    question_model = args.question_model
    grader_model = args.grader_model
    output_file = args.output_file

    client = get_client()

    if dataset == 'default':
        qa = get_dataset(verbose=verbose)
    else:
        with open(dataset, 'r') as infile:
            qa = json.load(infile)

    if limit and (limit < 1 or limit > len(qa)):
        raise ValueError(f'Invalid limit: {limit}\nLimit must be between 1 and {len(qa)}.')

    # Measure execution time to benchmark sequential/synchronous execution
    # versus asynchronous execution:
    start = time.perf_counter()

    # Main loop:
    solutions = asyncio.run(
        main(qa, client, limit=limit, mode=mode, question_model=question_model,
             grader_model=grader_model, verbose=verbose)
    )

    # Save:
    save(solutions, output_file=output_file)

    # For testing/debugging:
    if verbose:
        display(solutions)

    report(
        solutions, results_file=Path(output_file), report_file=args.report_file,
        mode=mode, question_model=question_model, grader_model=grader_model
    )

    end = time.perf_counter()
    print(f'Program execution time: {end - start:,.3f} seconds\n')
