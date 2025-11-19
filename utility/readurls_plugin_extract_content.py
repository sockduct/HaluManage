# ...existing code...
import re
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
    - matches typical URL characters including percent-encoding and apostrophes
    - matches URLs with or without scheme (e.g. en.wikipedia.org/...)
    - does NOT strip characters that can be part of Wikipedia titles (apostrophe, parentheses)
    - removes ONLY sentence-ending punctuation (.,;:!?") and then fixes unbalanced )/]
    Returns list of (original_match, normalized_url) where normalized_url has https:// prepended if needed.
    """
    if not text:
        return []
    # allow optional scheme (http/https) or www., require a dot/TLD to avoid too many false positives
    url_pattern = re.compile(
        r"(?:https?://)?(?:www\.)?[A-Za-z0-9\-.]+\.[A-Za-z]{2,}(?:[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*)", re.UNICODE
    )

    matches = url_pattern.findall(text)

    cleaned = []
    for orig in matches:
        url = orig
        # Remove terminal double-quote and common sentence punctuation, but DO NOT strip apostrophes or closing parens here
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
        if url.endswith('.'):
            trailing_dot = True
            url = url.rstrip('.')
        else:
            trailing_dot = False

        if not url:
            continue

        # Normalize: if no scheme present, prepend https://
        if not re.match(r"^https?://", url, re.I):
            normalized = "https://" + url
        else:
            normalized = url

        # Re-add the trailing dot if it was present
        if trailing_dot:
            normalized += '.'

        cleaned.append((orig, normalized))
    return cleaned

def _try_fetch_variants(original_url: str, headers: dict, verify) -> requests.Response:
    """
    Try several normalization variants:
      - original
      - single unquote
      - double unquote (handles double-encoded %2527)
      - re-encoded path/query (quote after unquote) to ensure apostrophe and other unsafe chars are percent-encoded
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
        path = quote(unquote(parts.path), safe="/()")  # keep parentheses if present
        query = quote(unquote(parts.query), safe="=&?")
        reenc = urlunsplit((parts.scheme, parts.netloc, path, query, parts.fragment))
        if reenc not in candidates:
            candidates.append(reenc)
    except Exception:
        pass

    for u in candidates:
        try:
            resp = requests.get(u, headers=headers, timeout=12, verify=verify)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            continue

    raise last_exc

def fetch_webpage_content(url: str, max_length: int = 100000, verify_ssl: Optional[bool] = None, cert_path: Optional[str] = None) -> str:
    try:
        # Normalize the URL by removing trailing dots for processing but keep them for valid URLs
        original_url = url
        url = url.rstrip('.')

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
            'User-Agent': f'halumanage/{__version__} (https://github.com/codelion/halumanage)'
        }

        try:
            resp = _try_fetch_variants(url, headers, verify)
        except Exception as e:
            return f"Error fetching content: {str(e)}"

        soup = BeautifulSoup(resp.content, 'lxml')
        for script in soup(["script", "style"]):
            script.decompose()

        text_elements = []
        for tag in ['article', 'main', 'div[role="main"]', '.main-content']:
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

    return modified_query, 0
# ...existing code...