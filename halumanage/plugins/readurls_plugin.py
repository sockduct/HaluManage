import re
import json
from typing import Tuple, List, Optional
import requests
import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote, urlsplit, urlunsplit, quote
from halumanage import __version__, server_config

SLUG = "readurls"

def extract_urls(text: str) -> List[Tuple[str, str]]:
    """
    Robust URL extractor:
    - extracts URLs from text, including those within Python lists/quotes
    - matches URLs with or without scheme (e.g. en.wikipedia.org/...)
    - does NOT strip characters that can be part of Wikipedia titles (apostrophe, parentheses)
    - removes ONLY sentence-ending punctuation (.,;:!?") and then fixes unbalanced )/]
    Returns list of (original_match, normalized_url) where normalized_url has https:// prepended if needed.
    """
    if not text:
        return []
    
    # Match URLs within quotes or as standalone:
    # - https?://... (with scheme)
    # - www\.... (with www)
    # - en\.wikipedia\.org/... (bare domain without scheme)
    # Capture anything until we hit a quote, bracket, or whitespace that clearly ends the URL
    url_pattern = re.compile(
        r"(?:https?://[^\s'\"\]\)]+|www\.[^\s'\"\]\)]+|[a-z]{2,}\.wikipedia\.org/[^\s'\"\]\)]+)",
        re.UNICODE | re.IGNORECASE
    )

    matches = url_pattern.findall(text)

    cleaned = []
    for orig in matches:
        url = orig
        
        # Remove terminal double-quote and common sentence punctuation
        while url and url[-1] in '.,;:!?"':
            url = url[:-1]

        # Fix unbalanced closing parens at the end only
        open_parens = url.count('(')
        close_parens = url.count(')')
        while close_parens > open_parens and url.endswith(')'):
            url = url[:-1]
            close_parens -= 1

        # Fix unbalanced closing brackets at the end only
        open_br = url.count('[')
        close_br = url.count(']')
        while close_br > open_br and url.endswith(']'):
            url = url[:-1]
            close_br -= 1

        # Preserve trailing dots for URLs that require them
        if url.endswith('.') and not url.endswith('..'):
            # Check if it's part of a title (like F.C.) by looking at context
            trailing_dot = True
            url_without_dot = url.rstrip('.')
        else:
            trailing_dot = False
            url_without_dot = url

        if not url_without_dot:
            continue

        # Normalize: if no scheme present, prepend https://
        if not re.match(r"^https?://", url_without_dot, re.I):
            normalized = "https://" + url_without_dot
        else:
            normalized = url_without_dot

        # Re-add the trailing dot if it was present (except for double dots)
        if trailing_dot and not url.endswith('..'):
            normalized += '.'

        cleaned.append((orig, normalized))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_cleaned = []
    for orig, norm in cleaned:
        if norm not in seen:
            seen.add(norm)
            unique_cleaned.append((orig, norm))
    
    return unique_cleaned

def _try_fetch_variants(original_url: str, headers: dict, verify) -> requests.Response:
    """
    Try several normalization variants with redirect support:
      - original
      - single unquote
      - double unquote (handles double-encoded %2527)
      - re-encoded path/query (quote after unquote) to ensure special chars are percent-encoded
    Returns successful Response or raises last exception.
    """
    last_exc = None
    candidates = [original_url]

    try:
        u1 = unquote(original_url)
        if u1 != original_url:
            candidates.append(u1)
        u2 = unquote(u1)
        if u2 != u1:
            candidates.append(u2)
    except Exception:
        pass

    try:
        parts = urlsplit(original_url)
        # Keep parentheses, dots, commas, and other wiki-valid chars
        path = quote(unquote(parts.path), safe="/()'!.,")
        query = quote(unquote(parts.query), safe="=&?")
        reenc = urlunsplit((parts.scheme, parts.netloc, path, query, parts.fragment))
        if reenc not in candidates:
            candidates.append(reenc)
    except Exception:
        pass

    for u in candidates:
        try:
            resp = requests.get(u, headers=headers, timeout=15, verify=verify, allow_redirects=True)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            continue

    raise last_exc

def fetch_webpage_content(url: str, max_length: int = 100000, verify_ssl: Optional[bool] = None, cert_path: Optional[str] = None) -> str:
    try:
        # Keep the URL as-is (do not strip trailing dots)
        original_url = url

        if verify_ssl is None:
            verify_ssl = server_config.get('ssl_verify', True)
        if cert_path is None:
            cert_path = server_config.get('ssl_cert_path', '')

        if not verify_ssl:
            verify = False
        elif cert_path:
            verify = cert_path
        else:
            verify = True

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        try:
            resp = _try_fetch_variants(original_url, headers, verify)
        except Exception as e:
            return f"Error fetching content: {str(e)}"

        soup = BeautifulSoup(resp.content, 'lxml')
        for script in soup(["script", "style"]):
            script.decompose()

        text_elements = []
        for tag in ['article', 'main', 'div[role="main"]', '.mw-parser-output']:
            content = soup.select_one(tag)
            if content:
                text_elements.extend(content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'table']))
                break

        if not text_elements:
            text_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'table'])

        content_parts = []
        for element in text_elements:
            if element.name == 'table':
                table_content = []
                headers_list = element.find_all('th')
                if headers_list:
                    header_text = ' | '.join(header.get_text(strip=True) for header in headers_list)
                    table_content.append(header_text)
                for row in element.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        row_text = ' | '.join(cell.get_text(strip=True) for cell in cells)
                        table_content.append(row_text)
                content_parts.append('\n' + '\n'.join(table_content) + '\n')
            else:
                content_parts.append(element.get_text(strip=False))

        text = ' '.join(content_parts)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\[.*?\]', '', text)

        if len(text) > max_length:
            text = text[:max_length] + '...'

        return text
    except Exception as e:
        return f"Error fetching content: {str(e)}"

def run(system_prompt, initial_query: str, client=None, model=None) -> Tuple[str, int]:
    """
    Replace each raw URL substring in the original text with fetched inline content.
    Keep original raw matched URL for replacement so encoding is preserved in output.
    """
    url_tuples = extract_urls(initial_query)
    modified_query = initial_query

    for raw_match, fetch_url in url_tuples:
        content = fetch_webpage_content(fetch_url)
        domain = urlparse(fetch_url).netloc
        # replace the original matched text (which may not include scheme) with the https:// variant + content
        modified_query = modified_query.replace(raw_match, f"{fetch_url} [Content from {domain}: {content}]")
        # fallback: also replace decoded variant if present
        try:
            decoded = unquote(raw_match)
            if decoded != raw_match:
                modified_query = modified_query.replace(decoded, f"{fetch_url} [Content from {domain}: {content}]")
        except Exception:
            pass

    # Load existing data from the JSON file, or initialize an empty list
    '''output_filename = "readurls_output.json"
    if os.path.exists(output_filename):
        try:
            with open(output_filename, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            output_data = []
    else:
        output_data = []

    # Append the new modified_query and write back to the file
    output_data.append(modified_query)
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)'''

    return modified_query, 0