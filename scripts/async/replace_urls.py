#! /usr/bin/env python


# Standard Library:
from ast import literal_eval
import asyncio
import json
import os
import re
import ssl
from itertools import islice
from pathlib import Path
from typing import List, Optional, Tuple, Union

# 3rd party libraries:
import aiohttp
from bs4 import BeautifulSoup
from datasets import load_dataset


__version__ = '0.3.6'

DATASET = 'google/frames-benchmark'

LIMITER: int|None = None
MAX_RETRIES = 3
MAX_CONCURRENT_REQUESTS = int(os.getenv("URL_FETCH_CONCURRENCY", "2"))
REQUEST_GATE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

MAX_LENGTH = 1_000_000
URL_KEY = 'wiki_links'
OUTFILE = Path(__file__).parent/'data'/'frames_with_context.json'


def _build_ssl_setting(*, verify_ssl: bool=True, cert_path: Optional[str]=None
                       ) -> Union[bool, ssl.SSLContext, None]:
    if not verify_ssl:
        return False

    return ssl.create_default_context(cafile=cert_path) if cert_path else None


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\[.*?\]+", "", text)


def _extract_article(soup: BeautifulSoup, max_length: int) -> str:
    text_elements: List = []
    for tag in ['article', 'main', 'div[role="main"]', '.main-content']:
        content = soup.select_one(tag)
        if content:
            text_elements.extend(content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'table']))
            break

    if not text_elements:
        text_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'table'])

    content_parts: List[str] = []
    for element in text_elements:
        if element.name == 'table':
            table_content = []
            headers = element.find_all('th')
            if headers:
                table_content.append(' | '.join(header.get_text(strip=True) for header in headers))
            for row in element.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                if cells:
                    table_content.append(' | '.join(cell.get_text(strip=True) for cell in cells))
            content_parts.append('\n' + '\n'.join(table_content) + '\n')
        else:
            content_parts.append(element.get_text(strip=False))

    text = _clean_text(' '.join(content_parts))
    if len(text) > max_length:
        text = text[:max_length] + '...'

    return text


async def fetch_url_content(session: aiohttp.ClientSession, url: str, max_length: int,
                            ssl_setting: Union[bool, ssl.SSLContext, None]) -> Tuple[str, str]:
    # Required for Wikipedia or get 403:
    headers = {'User-Agent': f'optillm/{__version__} (https://github.com/codelion/optillm)'}
    delay = 1.0
    content_bytes: Optional[bytes] = None

    async with REQUEST_GATE:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(url, headers=headers, ssl=ssl_setting) as response:
                    if response.status == 429:
                        raise aiohttp.ClientResponseError(
                            status=response.status,
                            message="Too Many Requests",
                            headers=response.headers,
                            request_info=response.request_info,
                            history=response.history,
                        )
                    response.raise_for_status()
                    content_bytes = await response.read()
                    break
            except aiohttp.ClientResponseError as exc:
                if exc.status == 429 and attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                return "Unavailable", f"Error fetching content: {exc}"
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                return "Unavailable", f"Error fetching content: {exc}"

    if not content_bytes:
        return "Unavailable", "Error fetching content: empty response"

    soup = BeautifulSoup(content_bytes, 'lxml')
    for script in soup(["script", "style"]):
        script.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url
    article = _extract_article(soup, max_length)
    return title, article


def load_json(path: Union[str, Path]):
    with open(path, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    if not isinstance(data, list):
        raise ValueError("Expected the JSON file to contain a list of dictionaries.")
    return data


def write_json(path: Union[str, Path], payload) -> None:
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2)


async def process_entry(entry: dict, url_key: str, max_length: int,
                        session: aiohttp.ClientSession,
                        ssl_setting: Union[bool, ssl.SSLContext, None],
                        entry_index: int, total_entries: int) -> None:
    print(f"Starting entry {entry_index + 1}/{total_entries}")
    urls = entry.get(url_key, [])
    if isinstance(urls, str):
        urls = literal_eval(urls)
    elif not isinstance(urls, list):
        entry["wiki_content"] = []
        print(f"Finished entry {entry_index + 1}/{total_entries}")
        return

    wiki_content = []
    tasks: List[tuple[int, str, asyncio.Task]] = []
    doc_index = 1
    for raw_url in urls:
        url = str(raw_url).strip()
        if not url:
            continue
        task = asyncio.create_task(fetch_url_content(session, url, max_length, ssl_setting))
        tasks.append((doc_index, url, task))
        doc_index += 1

    for index, url, task in tasks:
        title, article = await task
        wiki_content.append({
            "doc_index": index,
            "url": url,
            "title": title,
            "article": article,
        })

    entry["wiki_content"] = wiki_content
    print(f"Finished entry {entry_index + 1}/{total_entries}")


async def augment_entries(data: List[dict], url_key: str, max_length: int) -> None:
    ssl_setting = _build_ssl_setting()
    timeout = aiohttp.ClientTimeout(total=20)
    limit = LIMITER
    total_entries = len(data) if limit is None else min(limit, len(data))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await asyncio.gather(*[
            process_entry(data[i], url_key, max_length, session, ssl_setting, i, total_entries)
            for i in range(total_entries)
        ])


def get_dataset(*, dataset: str=DATASET, verbose: bool=False) -> list[dict[str, str]]:
    # columns = ['Unnamed: 0', 'Prompt', 'Answer', 'reasoning_types']
    columns = None

    if verbose:
        print(f'Loading dataset {dataset}...')

    data = load_dataset(dataset)

    if columns:
        return data['test'].select_columns(columns).to_list()
    else:
        return data['test'].to_list()


def main() -> None:
    data = get_dataset(verbose=True)
    asyncio.run(augment_entries(data, URL_KEY, MAX_LENGTH))

    if limit := LIMITER:
        data = data[:limit]
    write_json(OUTFILE, data)


if __name__ == "__main__":
    main()
