# FILE: tools/web_tool.py
import requests
import re
import traceback
import json
import io
import random  # Added for User-Agent rotation
from urllib.parse import urlparse  # Added for Referer generation
from typing import Dict, Any
from bs4 import BeautifulSoup, NavigableString, Tag
import fitz  # PyMuPDF

from .base import Tool
import config
from config import SEARXNG_BASE_URL, SEARXNG_TIMEOUT

import logging

log = logging.getLogger("TOOL_Web")

# List of common, relatively modern User-Agent strings
COMMON_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/119.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 Edg/118.0.2088.76",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",  # Safari
]


class WebTool(Tool):
    """
    Performs web-related actions: searching using SearXNG or browsing/parsing a specific URL.
    Requires 'action' parameter: 'search' or 'browse'.
    Attempts to mimic browser behavior to reduce blocking, but may still fail on sites with advanced bot detection.
    """

    def __init__(self):
        super().__init__(
            name="web",
            description=(
                "Performs web actions. Requires 'action' parameter. "
                "Actions: "
                "'search' (requires 'query'): Uses SearXNG to find information. Returns search results. "
                "'browse' (requires 'url'): Fetches and parses content from a URL (HTML, PDF, JSON, TXT). Returns parsed content. "
                "Note: Browsing may fail on sites with strong bot protection (e.g., Cloudflare)."
            ),
        )
        # Use a session object for connection pooling and cookie handling
        self.session = requests.Session()
        # Session-wide headers (User-Agent will be set per-request)
        self.session.headers.update(
            {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,application/pdf,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",  # Request compressed content
                "Connection": "keep-alive",  # Keep connection open for potential reuse
                "Upgrade-Insecure-Requests": "1",  # Signal preference for HTTPS
                "DNT": "1",  # Signal "Do Not Track"
            }
        )
        # For search action (checked during run)
        self.search_headers = {"Accept": "application/json"}

    def _get_browse_headers(self, url: str) -> Dict[str, str]:
        """Generates headers for a specific browse request, including a random User-Agent and Referer."""
        headers = self.session.headers.copy()  # Start with session headers
        headers["User-Agent"] = random.choice(COMMON_USER_AGENTS)

        # Add a Referer header based on the target URL's domain
        try:
            parsed_url = urlparse(url)
            # Use the scheme and netloc (domain) as the referer base
            referer = f"{parsed_url.scheme}://{parsed_url.netloc}/"
            headers["Referer"] = referer
        except Exception:
            # If URL parsing fails, don't add a referer
            log.warning(f"Could not parse URL '{url}' to generate Referer header.")
            pass

        return headers

    # --- Helper methods from WebBrowserTool (unchanged) ---
    def _clean_text_basic(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_text_from_html_node(self, node: Tag) -> str:
        if isinstance(node, NavigableString):
            return re.sub(r"\s+", " ", str(node)).strip()
        if not isinstance(node, Tag):
            return ""
        content = []
        tag_name = node.name.lower()
        prefix = ""
        suffix = "\n\n"
        if tag_name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            prefix = "#" * int(tag_name[1]) + " "
        elif tag_name == "p":
            pass
        elif tag_name in ["ul", "ol"]:
            for i, item in enumerate(node.find_all("li", recursive=False)):
                item_prefix = "* " if tag_name == "ul" else f"{i+1}. "
                item_content = self._extract_text_from_html_node(item).strip()
                if item_content:
                    content.append(item_prefix + item_content)
            suffix = "\n"
            prefix = ""
        elif tag_name in ["pre", "code"]:
            code_text = node.get_text(strip=True)
            if "\n" in code_text:
                prefix = "```\n"
                suffix = "\n```\n\n"
                content = [code_text]
            else:
                prefix = "`"
                suffix = "` "
                content = [code_text]
            return prefix + "".join(content) + suffix
        elif tag_name == "a":
            href = node.get("href")
            text = node.get_text(strip=True)
            if text and href and href.startswith(("http://", "https://")):
                return f"[{text}]({href})"
            elif text:
                return text
            else:
                return ""
        elif tag_name in ["strong", "b"]:
            prefix = "**"
            suffix = "** "
        elif tag_name in ["em", "i"]:
            prefix = "*"
            suffix = "* "
        elif tag_name in ["br"]:
            return "\n"
        elif tag_name in ["hr"]:
            return "---\n\n"
        elif tag_name in [
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "aside",
            "form",
            "button",
            "select",
            "option",
            "noscript",
        ]:
            return ""
        elif tag_name in ["div", "span", "section", "article", "main", "td", "th"]:
            suffix = "\n" if tag_name in ["span"] else "\n\n"
            pass
        elif tag_name in ["table"]:
            content.append("\n| Header 1 | Header 2 | ... |\n|---|---|---|")
            for row in node.find_all("tr"):
                row_content = [
                    self._extract_text_from_html_node(cell).strip()
                    for cell in row.find_all(["th", "td"])
                ]
                if row_content:
                    content.append("| " + " | ".join(row_content) + " |")
            return "\n".join(content) + "\n\n"
        if not content:
            child_contents = []
            for child in node.children:
                child_text = self._extract_text_from_html_node(child)
                if child_text:
                    child_contents.append(child_text)
            content = [" ".join(child_contents).strip()]
        full_content = prefix + "".join(content) + suffix
        return self._clean_text_basic(full_content)

    def _parse_html_to_markdown(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, "html.parser")
        body = soup.find("body")
        if not body:
            return self._clean_text_basic(soup.get_text())
        main_content_selectors = [
            "main",
            "article",
            ".main",
            ".content",
            "#main",
            "#content",
            "body",
        ]
        main_node = None
        for selector in main_content_selectors:
            try:
                if selector == "body":
                    main_node = body
                    break
                node = body.select_one(selector)
                if node:
                    log.debug(f"Found main content using selector: '{selector}'")
                    main_node = node
                    break
            except Exception:
                log.warning(f"Selector '{selector}' failed.", exc_info=False)
                continue
        if not main_node:
            log.warning(
                "Could not identify specific main content area, parsing entire body."
            )
            main_node = body
        extracted_text = self._extract_text_from_html_node(main_node)
        lines = extracted_text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:
                cleaned_lines.append(stripped_line)
        final_text = ""
        for i, line in enumerate(cleaned_lines):
            final_text += line
            is_block_end = not line.startswith(
                (
                    "* ",
                    "1.",
                    "2.",
                    "3.",
                    "4.",
                    "5.",
                    "6.",
                    "7.",
                    "8.",
                    "9.",
                    "#",
                    "`",
                    "|",
                    "-",
                )
            )
            next_is_block_start = False
            if i + 1 < len(cleaned_lines):
                next_is_block_start = cleaned_lines[i + 1].startswith(
                    (
                        "* ",
                        "1.",
                        "2.",
                        "3.",
                        "4.",
                        "5.",
                        "6.",
                        "7.",
                        "8.",
                        "9.",
                        "#",
                        "`",
                        "|",
                        "-",
                    )
                )
            if i < len(cleaned_lines) - 1:
                if is_block_end and not next_is_block_start:
                    final_text += "\n\n"
                else:
                    final_text += "\n"
        return final_text.strip()

    # --- End Helper methods ---

    def _run_search(self, query: str) -> Dict[str, Any]:
        """Executes a search query against the configured SearXNG instance."""
        log.info(f"Executing SearXNG Search: '{query}'")
        if not SEARXNG_BASE_URL:
            return {"error": "SearXNG base URL is not configured for search action."}

        params = {"q": query, "format": "json"}
        searxng_url = SEARXNG_BASE_URL.rstrip("/") + "/search"

        try:
            # Use a standard requests.get for search, as session/fancy headers are less critical here
            response = requests.get(
                searxng_url,
                params=params,
                headers=self.search_headers,
                timeout=SEARXNG_TIMEOUT,
            )
            response.raise_for_status()

            try:
                search_data = response.json()
            except json.JSONDecodeError:
                log.error(
                    f"Failed SearXNG JSON decode. Status: {response.status_code}. Resp: {response.text[:200]}..."
                )
                return {"error": "Failed JSON decode from SearXNG."}

            # --- (Processing logic remains the same as before) ---
            processed = []
            if "infoboxes" in search_data and isinstance(
                search_data["infoboxes"], list
            ):
                for ib in search_data["infoboxes"]:
                    if isinstance(ib, dict):
                        content = ib.get("content")
                        title = ib.get("infobox_type", "Infobox")
                        url = None
                        if (
                            ib.get("urls")
                            and isinstance(ib["urls"], list)
                            and len(ib["urls"]) > 0
                            and isinstance(ib["urls"][0], dict)
                        ):
                            url = ib["urls"][0].get("url")
                        if content:
                            processed.append(
                                {
                                    "type": "infobox",
                                    "title": title,
                                    "snippet": content,
                                    "url": url,
                                }
                            )
            if "results" in search_data and isinstance(search_data["results"], list):
                for r in search_data["results"][:7]:  # Limit results
                    if isinstance(r, dict):
                        title = r.get("title")
                        url = r.get("url")
                        snippet = r.get("content")
                        if title and url and snippet:
                            processed.append(
                                {
                                    "type": "organic",
                                    "title": title,
                                    "snippet": snippet,
                                    "url": url,
                                }
                            )
            if "answers" in search_data and isinstance(search_data["answers"], list):
                for ans in search_data["answers"]:
                    if isinstance(ans, str) and ans.strip():
                        processed.append(
                            {
                                "type": "answer",
                                "title": "Direct Answer",
                                "snippet": ans,
                                "url": None,
                            }
                        )

            if not processed:
                if "error" in search_data:
                    log.warning(f"SearXNG reported error: {search_data['error']}")
                    return {"error": f"SearXNG error: {search_data['error']}"}
                else:
                    log.info("No relevant results found by SearXNG.")
                    return {"results": [], "message": "No relevant results found."}

            log.info(f"SearXNG returned {len(processed)} processed results.")
            return {
                "status": "success",
                "action": "search",
                "results": processed,
                "message": f"Search completed successfully, returning {len(processed)} results.",
            }
        # --- (Error handling remains the same) ---
        except requests.exceptions.Timeout:
            log.error(
                f"SearXNG request timed out ({SEARXNG_TIMEOUT}s) to {searxng_url}"
            )
            return {"error": "Web search timed out."}
        except requests.exceptions.RequestException as e:
            log.error(f"SearXNG request failed: {e}")
            return {"error": f"Web search connection/query failed."}
        except Exception as e:
            log.exception(f"Unexpected error during SearXNG search: {e}")
            return {"error": f"Unexpected web search error."}

    def _run_browse(self, url: str) -> Dict[str, Any]:
        """Fetches and extracts text content from the provided URL using enhanced headers."""
        log.info(f"Attempting to browse URL: {url}")

        if not url.startswith(("http://", "https://")):
            log.error(
                f"Invalid URL format: '{url}'. Must start with http:// or https://."
            )
            return {"error": "Invalid URL format. Must start with http:// or https://."}

        extracted_text = ""
        content_source = "unknown"

        try:
            # Generate headers for this specific request
            request_headers = self._get_browse_headers(url)
            log.debug(f"Using headers for {url}: {request_headers}")

            # Use the session object to make the GET request
            response = self.session.get(
                url,
                headers=request_headers,  # Pass specific headers for this request
                timeout=25,
                allow_redirects=True,  # Ensure redirects are followed by the session
            )

            # Check for non-success status codes *after* potential redirects
            final_url = response.url  # URL after potential redirects
            log.info(
                f"Request to {url} resulted in status {response.status_code} at {final_url}"
            )
            if response.status_code == 403:
                log.warning(
                    f"Received 403 Forbidden for {final_url}. Site may be blocking automated access."
                )
                # Return a specific error for 403
                return {
                    "error": f"Access denied (403 Forbidden) when trying to browse {final_url}. The site may block automated tools."
                }
            elif response.status_code == 404:
                log.warning(f"Received 404 Not Found for {final_url}.")
                return {"error": f"Page not found (404 Not Found) at {final_url}."}

            # Raise exceptions for other bad status codes (4xx client errors, 5xx server errors)
            response.raise_for_status()

            content_type = (
                response.headers.get("Content-Type", "").lower().split(";")[0].strip()
            )
            log.info(f"URL Content-Type detected: '{content_type}' from {final_url}")

            # --- (Parsing logic remains the same as before) ---
            if content_type == "application/pdf":
                content_source = "pdf"
                log.info("Parsing PDF content...")
                pdf_texts = []
                try:
                    pdf_stream = io.BytesIO(response.content)
                    doc = fitz.open(stream=pdf_stream, filetype="pdf")
                    log.info(f"PDF has {len(doc)} pages.")
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        page_text = page.get_text("text", sort=True)
                        if page_text:
                            pdf_texts.append(f"\n--- Page {page_num + 1} ---\n")
                            pdf_texts.append(page_text.strip())
                    doc.close()
                    full_text = "\n".join(pdf_texts).strip()
                    extracted_text = self._clean_text_basic(full_text)
                    log.info(
                        f"Extracted text from PDF (Length: {len(extracted_text)})."
                    )
                except Exception as pdf_err:
                    log.error(
                        f"Error parsing PDF from {final_url}: {pdf_err}", exc_info=True
                    )
                    return {"error": f"Failed to parse PDF content. Error: {pdf_err}"}

            elif content_type == "text/html":
                content_source = "html_markdown"
                log.info("Parsing HTML content to Markdown...")
                extracted_text = self._parse_html_to_markdown(response.text)
                log.info(
                    f"Extracted text from HTML as Markdown (Length: {len(extracted_text)})."
                )

            elif content_type == "application/json":
                content_source = "json"
                log.info("Parsing JSON content...")
                try:
                    json_data = json.loads(response.text)
                    extracted_text = json.dumps(json_data, indent=2, ensure_ascii=False)
                    log.info(f"Formatted JSON content (Length: {len(extracted_text)}).")
                except json.JSONDecodeError as json_err:
                    log.error(f"Invalid JSON received from {final_url}: {json_err}")
                    return {
                        "error": f"Failed to parse JSON content. Invalid format. Error: {json_err}"
                    }

            elif content_type.startswith("text/"):
                content_source = "text"
                log.info(f"Parsing plain text content ({content_type})...")
                extracted_text = self._clean_text_basic(response.text)
                log.info(f"Extracted plain text (Length: {len(extracted_text)}).")

            else:
                log.warning(
                    f"Unsupported Content-Type '{content_type}' for URL {final_url}. Attempting fallback."
                )
                try:
                    # Use response.content and decode carefully for fallback
                    fallback_text = self._clean_text_basic(
                        response.content.decode(
                            response.encoding or "utf-8", errors="replace"
                        )
                    )
                    if fallback_text and len(fallback_text) > 50:
                        extracted_text = fallback_text
                        content_source = "text_fallback"
                        log.info(
                            f"Extracted fallback text (Length: {len(extracted_text)})."
                        )
                    else:
                        return {
                            "error": f"Cannot browse: Unsupported content type '{content_type}' and no significant fallback text found."
                        }
                except Exception as fallback_err:
                    log.error(
                        f"Error during fallback text extraction for {final_url}: {fallback_err}"
                    )
                    return {
                        "error": f"Cannot browse: Unsupported content type '{content_type}' and fallback extraction failed."
                    }
            # --- (Truncation and result formatting remains the same) ---
            if not extracted_text:
                log.warning(
                    f"Could not extract significant textual content ({content_source}) from {final_url}"
                )
                return {
                    "status": "success",
                    "action": "browse",
                    "url": final_url,  # Return the final URL after redirects
                    "content": None,
                    "message": f"No significant textual content found via {content_source} parser.",
                }

            limit = config.CONTEXT_TRUNCATION_LIMIT
            truncated = False
            message = "Content browsed and parsed successfully."
            if len(extracted_text) > limit:
                log.info(
                    f"Content from {final_url} truncated from {len(extracted_text)} to {limit} characters."
                )
                extracted_text = extracted_text[:limit] + "\n\n... [CONTENT TRUNCATED]"
                truncated = True
                message = f"Content browsed successfully but was truncated to approximately {limit} characters."

            result = {
                "status": "success",
                "action": "browse",
                "url": final_url,  # Return the final URL
                "content_source": content_source,
                "content": extracted_text,
                "truncated": truncated,
                "message": message,
            }

            log.info(
                f"Successfully browsed and extracted content from {final_url} (Source: {content_source}, Length: {len(extracted_text)} chars)."
            )
            return result

        # --- (Error handling remains mostly the same, updated URL in messages) ---
        except requests.exceptions.Timeout:
            log.error(f"Request timed out while browsing {url}")
            # Use the original requested URL in the timeout error message
            return {"error": f"Request timed out accessing URL: {url}"}
        except requests.exceptions.RequestException as e:
            # Log the error with the original URL, but the error message might contain the final URL if redirection happened before failure
            log.error(f"Request failed for URL {url}: {e}")
            status_code = (
                e.response.status_code
                if hasattr(e, "response") and e.response is not None
                else "N/A"
            )
            # Use the final URL from the response if available, otherwise the original URL
            error_url = (
                e.response.url
                if hasattr(e, "response") and e.response is not None
                else url
            )
            return {
                "error": f"Failed to retrieve URL {error_url}. Status: {status_code}. Error: {e}"
            }
        except Exception as e:
            log.exception(f"Unexpected error during web browse for {url}: {e}")
            return {"error": f"Unexpected error browsing URL {url}: {e}"}

    def run(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Runs either the search or browse action based on parameters."""
        if not isinstance(parameters, dict):
            log.error(f"Invalid params type: {type(parameters)}")
            return {
                "error": "Invalid parameters format. Expected a JSON object (dict) with an 'action' key."
            }

        action = parameters.get("action")

        if action == "search":
            query = parameters.get("query")
            if not query or not isinstance(query, str) or not query.strip():
                return {
                    "error": "Missing or invalid 'query' parameter for 'search' action."
                }
            return self._run_search(query)
        elif action == "browse":
            url = parameters.get("url")
            if not url or not isinstance(url, str) or not url.strip():
                return {
                    "error": "Missing or invalid 'url' parameter for 'browse' action."
                }
            return self._run_browse(url)
        else:
            log.error(f"Invalid action specified for web tool: '{action}'")
            return {
                "error": "Invalid 'action' specified. Must be 'search' or 'browse'."
            }
