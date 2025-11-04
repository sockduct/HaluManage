#! /usr/bin/env python
'''
Asynchronous Proof of Concept for FRAMES dataset
* Query specified model with Open Router API Endpoint
* Grade response using Google's Gemini 1.5 Pro model (Open Router)
* Report results
'''


# Standard Library:
import asyncio
import json
import os
import time

# 3rd party libraries:
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError


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
#
NAIVE_Q_PROMPT = '''You are a helpful and accurate assistant.
Please answer the following question based on your knowledge.
Question: {question}
Answer:'''
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
GRADER_GATE = asyncio.Semaphore(20)
Q_GATE = asyncio.Semaphore(20)
MAX_RETRIES = 3
#
OUTPUT_FILE = 'results.json'


async def ask_1q_naive(question: str, *, client: AsyncOpenAI, temperature: float=0.0,
                       max_completion_tokens: int=1_000, model: str=MODEL1b) -> str:
    '''Send one question to model via client and return the model's answer.'''
    prompt = NAIVE_Q_PROMPT.format(question=question)

    async with Q_GATE:
        delay = 1.0
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    extra_headers = {},
                    extra_body = {},
                    model = model,
                    messages = [
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

    return response.choices[0].message.content.strip()


async def grade_response(*, question: str, actual_answer: str, model_answer: str,
                         client: AsyncOpenAI, temperature: float=0.3,
                         max_completion_tokens: int=1_000, model: str=MODEL7,
                         verbose: bool=False) -> dict[str, str]:
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
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
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
        print('WARNING: The model\'s response was truncated due to reaching the maximum '
              'alloted tokens.')

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

    return {'decision': decision, 'explanation': explanation}


async def process_question(*, index: int, question: dict[str, str], client: AsyncOpenAI,
                           status_queue: asyncio.Queue|None=None) -> dict[str, str]:
    if status_queue:
        await status_queue.put(('ask_start', index))
    model_answer = await ask_1q_naive(question['Prompt'], client=client)
    if status_queue:
        await status_queue.put(('ask_done', index))

    if status_queue:
        await status_queue.put(('grade_start', index))
    grade = await grade_response(question=question['Prompt'],
                                 actual_answer=question['Answer'],
                                 model_answer=model_answer,
                                 client=client)
    if status_queue:
        await status_queue.put(('grade_done', index))

    return {
        'Index': question.get('Unnamed: 0', index),
        'Prompt': question['Prompt'],
        'Answer': question['Answer'],
        'ModelAnswer': model_answer,
        'GraderDecision': grade['decision'],
        'GraderExplanation': grade['explanation'],
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
            print(f'Starting ask_1q_naive for question {question_number}...')
        elif event == 'ask_done':
            print(f'Completed ask_1q_naive for question {question_number}.')
        elif event == 'grade_start':
            print(f'Starting grade_response for question {question_number}...')
        elif event == 'grade_done':
            completed += 1
            print(f'Completed grade_response for question {question_number} '
                  f'({completed}/{total}).')
        queue.task_done()


async def main(qa: list[dict[str, str]], client: AsyncOpenAI) -> list[dict[str, str]]:
    total = min(len(qa), LIMITER) if LIMITER else len(qa)
    status_queue: asyncio.Queue = asyncio.Queue()
    monitor = asyncio.create_task(status_monitor(total, status_queue))

    tasks = []
    for index, question in enumerate(qa):
        if LIMITER and index >= LIMITER:
            break
        tasks.append(asyncio.create_task(
            process_question(index=index, question=question, client=client,
                             status_queue=status_queue)
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


def report(solutions: list[dict[str, str]]) -> None:
    '''Calculate and print summary statistics'''
    total_samples = len(solutions)
    correct_answers = sum(1 for s in solutions if s['GraderDecision'] == 'TRUE')
    accuracy = correct_answers/total_samples

    ### print(f'Model: {model}')
    print(f'Total samples: {total_samples}')
    print(f'Correct answers: {correct_answers}')
    print(f'Accuracy: {accuracy:.2%}')

    # Print accuracy by reasoning type
    reasoning_types = {s['ReasoningTypes'] for s in solutions}
    for rt in reasoning_types:
        rt_samples = [s for s in solutions if s['ReasoningTypes'] == rt]
        rt_correct = sum(1 for r in rt_samples if r['GraderDecision'] == 'TRUE')
        rt_accuracy = rt_correct / len(rt_samples)
        print(f'Accuracy for {rt}: {rt_accuracy:.2%}')


def save(solutions: list[dict[str, str]]) -> None:
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(solutions, f, indent=4)


def get_dataset(*, dataset: str=DATASET, verbose: bool=False) -> list[dict[str, str]]:
    columns = ['Unnamed: 0', 'Prompt', 'Answer', 'reasoning_types']

    if verbose:
        print(f'Loading dataset {dataset}...')

    data = load_dataset(dataset)

    if columns:
        return data['test'].select_columns(columns).to_list()
    else:
        return data['test'].to_list()


if __name__ == '__main__':
    verbose = True

    client = get_client()

    qa = get_dataset(verbose=verbose)

    # Measure execution time to benchmark sequential/synchronous execution
    # versus asynchronous execution:
    start = time.perf_counter()

    # Main loop:
    solutions = asyncio.run(main(qa, client))

    # Save:
    save(solutions)

    # Optional - for testing/debugging:
    display(solutions)

    report(solutions)

    end = time.perf_counter()
    print(f'Program execution time: {end - start:,.3f} seconds\n')
