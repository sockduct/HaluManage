#! /usr/bin/env python


# Standard Library:
import argparse
from ast import literal_eval
import asyncio
from collections import deque
from itertools import islice
import json
import os
from pathlib import Path
import re
import ssl
from typing import List, Optional, Tuple, Union

# 3rd party libraries:
import aiohttp
from bs4 import BeautifulSoup
from yarl import URL


__version__ = '0.4.0'

DATASET = 'google/frames-benchmark'

LIMITER: int|None = None
MAX_RETRIES = 3
MAX_CONCURRENT_REQUESTS = int(os.getenv("URL_FETCH_CONCURRENCY", "3"))
REQUEST_GATE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

MAX_LENGTH = 10_000_000
URL_KEY = 'wiki_links'
OUTFILE = Path(__file__).parent/'data'/'frames_with_context.json'

BAD_RECORDS = [141, 540]  # Records with former Wikipedia articles that have been deleted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Wikipedia content for FRAMES dataset entries."
    )
    parser.add_argument("-f", "--force", action="store_true",
                        help="Overwrite the existing output file if present.")
    parser.add_argument("--validate", action="store_true",
                        help="Validate URLs with HTTP HEAD before fetching content.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit how many records are processed.")
    parser.add_argument('--start', type=int, default=0,
                        help="Starting index for processing.")
    parser.add_argument('--end', type=int, default=None,
                        help="Ending index for processing.")
    return parser.parse_args()


def _build_ssl_setting(*, verify_ssl: bool=True, cert_path: Optional[str]=None
                       ) -> Union[bool, ssl.SSLContext, None]:
    if not verify_ssl:
        return False

    return ssl.create_default_context(cafile=cert_path) if cert_path else None


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\[.*?\]+", "", text)


def _normalize_url(url: str) -> str:
    '''
    Format URL and fix common errors:
    * Missing https:// prefix
      * Add
    * https://en.wikipedia.org/wiki/2021_French_Open_–_Men%2527s_singles
      * Double encoding error:  "'" encoded twice to %2527, restore to single encoding:  %27
    * https://en.wikipedia.org/wiki/Nemanja_Marković - 404
    * https://en.wikipedia.org/wiki/Jack_Vance_(tennis) - 404
      * Deleted articles - remove record (article has been removed from Wikipedia)
    * https://en.wikipedia.org/wiki/Pokémon (NOT REQUIRED, BUT HELPFUL)
      * Superfluous comment - strip everything from space on...
    * Failed to connect to https://en.wikipedia.org/wiki/American_Family_Field, <URL2>, <URL#>...
      * For wikipedia_link_11+ - if there's more than one URL, the string is actually a list of
        URLs.  Split into separate URLs - 7 of these.
    '''
    url = url.strip()
    if not url:
        return url

    # Fix common problems:
    # * %2527 -> %27 or "'"
    if '%2527' in url:
        print(f'Informational: Found %2527 in URL: {url}')
        url = url.replace('%2527', '%27')
        print(f'Informational: Updated URL: {url}')

    if ' ' in url:
        print(f'Informational: Found space in URL: {url}')
        url = url.split(' ')[0]
        print(f'Informational: Updated URL: {url}')

    if ', ' in url:
        print(f'Informational: Found comma in URL: {url}')

    if url.startswith("https://"):
        return url
    if url.startswith("http://"):
        return "https://" + url.removeprefix("http://")
    return f"https://{url}"


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
        print(f"Warning:  Truncated article to {max_length} characters.")

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


def _format_url(url: str) -> Tuple[str, str]:
    parsed_url = URL(url)
    return parsed_url.human_repr(), str(parsed_url)


async def validate_url(session: aiohttp.ClientSession, url: str, entry_index: int,
                       ssl_setting: Union[bool, ssl.SSLContext, None]) -> None:
    # Required for Wikipedia or get 403:
    headers = {'User-Agent': f'optillm/{__version__} (https://github.com/codelion/optillm)'}
    display_url, encoded_url = _format_url(url)
    async with REQUEST_GATE:
        try:
            async with session.head(
                encoded_url, headers=headers, ssl=ssl_setting, allow_redirects=True
            ) as response:
                status = response.status
                final_display, final_encoded = _format_url(str(response.url))
                redirect_suffix = ("" if not response.history else
                                   f" -> {final_display} ({final_encoded})")
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            reason = str(exc) or type(exc).__name__
            print(f"Failed to connect to ({entry_index}) {display_url} ({encoded_url}) - {reason}")
            return

    encoded_path = encoded_url.removeprefix('https://en.wikipedia.org/wiki/')
    if status == 200:
        print(f"Successfully connected to ({entry_index}) {display_url} ({encoded_path}) "
              f"{redirect_suffix}")
    elif 200 <= status < 400:
        print(f"Warning: Received HTTP Status ({entry_index}) {status} for {display_url} "
              f"({encoded_path}) {redirect_suffix}")
    else:
        print(f"Failed to connect to ({entry_index}) {display_url} ({encoded_path}) - "
              f"HTTP Status {status} {redirect_suffix}")


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
                        ssl_setting: Union[bool, ssl.SSLContext, None], *,
                        validate: bool=False,
                        entry_index: int, total_entries: int) -> None:
    print(f"Starting entry {entry_index + 1}/{total_entries}")

    # Check if marked bad record:
    if entry_index in BAD_RECORDS:
        print(f"Informational:  Skipping bad record {entry_index + 1}/{total_entries}")
        return

    urls = entry.get(url_key, [])
    if isinstance(urls, str):
        urls = literal_eval(urls)
        if not isinstance(urls, list):
            urls = [urls]
    elif not isinstance(urls, list):
        entry["wiki_content"] = []
        print(f"Finished entry {entry_index + 1}/{total_entries}")
        return

    wiki_content = []
    tasks: List[tuple[int, str, asyncio.Task]] = []
    doc_index = 1
    queue = deque(urls)
    while queue:
        raw_url = queue.popleft()

        # Is the string a URL with a superfluous comment?
        if isinstance(raw_url, str) and re.search(r' \([^)]*\) $', raw_url):
            print(f'Informational: Found superfluous comment in URL:  {raw_url}')
            raw_url = raw_url.split(' ')[0]
            print(f'Informational: Updated URL:  {raw_url}')

        # Is the string a comma-separated list?
        if isinstance(raw_url, str) and ', ' in raw_url:
            print(f'Informational: Found comma-separated list in URL:  {raw_url}')
            parts = [part.strip() for part in raw_url.split(', ') if part.strip()]
            if not parts:
                continue
            if len(parts) > 1:
                queue.extend(parts[1:])
            raw_url = parts[0]
            print(f'Informational: Updated URL:  {raw_url} (Remaining URLs to be '
                  'subsequently processed...)')

        url = _normalize_url(str(raw_url))
        if not url:
            continue

        if validate:
            await validate_url(session, url, entry_index, ssl_setting)
        else:
            task = asyncio.create_task(fetch_url_content(session, url, max_length, ssl_setting))
            tasks.append((doc_index, url, task))
            doc_index += 1

    if not validate:
        for index, url, task in tasks:
            title, article = await task
            if title == "Unavailable" and article.startswith("Error fetching content:"):
                print(f"{article} ({entry_index + 1} - doc#{index}:  {url})")
                continue

            wiki_content.append({
                "doc_index": index,
                "url": url,
                "title": title,
                "article": article,
            })

        entry["wiki_content"] = wiki_content

    print(f"Finished entry {entry_index + 1}/{total_entries}")


async def augment_entries(data: List[dict], url_key: str, max_length: int, *,
                          validate: bool=False, limit: Optional[int]=None, start: int=0,
                          end: Optional[int]=None) -> None:
    ssl_setting = _build_ssl_setting()
    timeout = aiohttp.ClientTimeout(total=20)
    if limit and end:
        limit = min(start + limit, end + 1)
        print(f'Warning:  Both limit and end specified - using smallest ending at {limit - 1}.')
    elif end:
        limit = end + 1

    total_entries = len(data) if limit is None else min(limit, len(data))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await asyncio.gather(*[
            process_entry(data[i], url_key, max_length, session, ssl_setting,
                          validate=validate, entry_index=i, total_entries=total_entries)
            for i in range(start, total_entries)
        ])


def get_dataset(*, dataset: str=DATASET, verbose: bool=False) -> list[dict[str, str]]:
    # Load library here to improve startup time (e.g., when doing ./replace_urls.py -h)
    from datasets import load_dataset

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
    args = parse_args()
    limit = args.limit if args.limit is not None else LIMITER

    if not args.validate and OUTFILE.exists() and not args.force:
        print(f"Output file already exists: {OUTFILE}")
        return

    data = get_dataset(verbose=True)
    asyncio.run(augment_entries(data, URL_KEY, MAX_LENGTH, validate=args.validate,
                                limit=limit, start=args.start, end=args.end))

    if args.validate:
        return

    if limit and args.end:
        limit = min(args.start + limit, args.end + 1)
    elif args.end:
        limit = args.end + 1

    if limit is not None:
        data = data[args.start:limit]
    for index, record in enumerate(data):
        if record["Unnamed: 0"] in BAD_RECORDS:
            data.pop(index)
    write_json(OUTFILE, data)


if __name__ == "__main__":
    main()
