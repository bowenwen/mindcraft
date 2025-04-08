# FILE: tools/web_tool.py
import requests
import datetime
import re
import traceback
import json
import io
import random
import uuid
import hashlib
from urllib.parse import urlparse
from typing import Dict, Any, Optional, List

from bs4 import BeautifulSoup, NavigableString, Tag
import fitz  # PyMuPDF
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import Tool
import config  # Use top-level import
from utils import setup_doc_archive_chromadb  # Use util for shared DB setup

import logging

log = logging.getLogger("TOOL_Web")

# List of common, relatively modern User-Agent strings (unchanged)
COMMON_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/119.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 Edg/118.0.2088.76",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
]


class WebTool(Tool):
    """
    Performs web-related actions: searching using SearXNG or browsing/parsing a specific URL.
    Requires 'action' parameter: 'search' or 'browse'.
    'search' (requires 'query'): Uses SearXNG to find information. Returns search results.
    'browse' (requires 'url', optional 'query'):
        - If 'query' is NOT provided: Fetches and parses content (HTML, PDF, JSON, TXT). Returns full (possibly truncated) content.
        - If 'query' IS provided: Fetches full content, chunks it, stores chunks in the document vector store, then performs a semantic search on those chunks using the 'query' string, returning only the most relevant snippets.
    """

    def __init__(self):
        super().__init__(
            name="web",
            description=(
                "Performs web actions. Requires 'action' parameter. "
                "Actions: "
                "'search' (requires 'query'): find snippets of information and associated urls. Returns search results. "
                "'browse' (requires 'url', optional 'query'): Fetches and parses content (HTML, PDF, JSON, TXT) using the url gathered from search. "
                "If 'query' is provided, fetches full content and returns only relevant snippets matching the query. "
                "Otherwise, returns the full (potentially truncated) page content. "
            ),
        )
        # Session for browsing (unchanged)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,application/pdf,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "DNT": "1",
            }
        )
        # Headers for search (unchanged)
        self.search_headers = {"Accept": "application/json"}

        # --- Document Archiving Setup (Use shared DB) ---
        self.doc_archive_collection: Optional[chromadb.Collection] = None
        try:
            # Use the utility function which now points to the shared DB path
            self.doc_archive_collection = setup_doc_archive_chromadb()
            if self.doc_archive_collection is None:
                log.error(
                    "SHARED Document archive DB collection failed to initialize. Browse-with-query will not work."
                )
            else:
                log.info(
                    "SHARED Document archive DB collection initialized successfully."
                )
        except Exception as e:
            log.critical(
                f"Failed to initialize SHARED document archive DB: {e}", exc_info=True
            )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.DOC_ARCHIVE_CHUNK_SIZE,
            chunk_overlap=config.DOC_ARCHIVE_CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )
        # --- End Document Archiving Setup ---

    # _get_browse_headers remains unchanged
    def _get_browse_headers(self, url: str) -> Dict[str, str]:
        headers = self.session.headers.copy()
        headers["User-Agent"] = random.choice(COMMON_USER_AGENTS)
        try:
            parsed_url = urlparse(url)
            referer = f"{parsed_url.scheme}://{parsed_url.netloc}/"
            headers["Referer"] = referer
        except Exception:
            log.warning(f"Could not parse URL '{url}' to generate Referer header.")
        return headers

    # --- Helper methods for HTML/Text Cleaning (unchanged) ---
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
                final_text += (
                    "\n\n" if is_block_end and not next_is_block_start else "\n"
                )
        return final_text.strip()

    # --- End Helper methods ---

    # _run_search remains unchanged
    def _run_search(self, query: str) -> Dict[str, Any]:
        log.info(f"Executing SearXNG Search: '{query}'")
        if not config.SEARXNG_BASE_URL:
            return {"error": "SearXNG base URL is not configured for search action."}
        params = {"q": query, "format": "json"}
        searxng_url = config.SEARXNG_BASE_URL.rstrip("/") + "/search"
        try:
            response = requests.get(
                searxng_url,
                params=params,
                headers=self.search_headers,
                timeout=config.SEARXNG_TIMEOUT,
            )
            response.raise_for_status()
            try:
                search_data = response.json()
            except json.JSONDecodeError:
                log.error(
                    f"Failed SearXNG JSON decode. Status: {response.status_code}. Resp: {response.text[:200]}..."
                )
                return {"error": "Failed JSON decode from SearXNG."}
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
        except requests.exceptions.Timeout:
            log.error(
                f"SearXNG request timed out ({config.SEARXNG_TIMEOUT}s) to {searxng_url}"
            )
            return {"error": "Web search timed out."}
        except requests.exceptions.RequestException as e:
            log.error(f"SearXNG request failed: {e}")
            return {"error": f"Web search connection/query failed."}
        except Exception as e:
            log.exception(f"Unexpected error during SearXNG search: {e}")
            return {"error": f"Unexpected web search error."}

    # _run_browse (Corrected)
    def _run_browse(self, url: str, query: Optional[str] = None) -> Dict[str, Any]:
        log.info(
            f"Attempting to browse URL: {url}"
            + (f" with query: '{query[:50]}...'" if query else "")
        )

        if not url.startswith(("http://", "https://")):
            log.error(
                f"Invalid URL format: '{url}'. Must start with http:// or https://."
            )
            return {"error": "Invalid URL format. Must start with http:// or https://."}

        extracted_text = ""
        content_source = "unknown"
        final_url = url
        response = None

        try:  # Main try for browse
            request_headers = self._get_browse_headers(url)
            log.debug(f"Using headers for {url}: {request_headers}")

            response = self.session.get(
                url, headers=request_headers, timeout=25, allow_redirects=True
            )
            final_url = response.url
            log.info(
                f"Request to {url} resulted in status {response.status_code} at {final_url}"
            )

            if response.status_code == 403:
                log.warning(
                    f"Received 403 Forbidden for {final_url}. Site may be blocking automated access."
                )
                return {
                    "error": f"Access denied (403 Forbidden) when trying to browse {final_url}. The site may block automated tools.",
                    "query_mode": bool(query),
                }
            elif response.status_code == 404:
                log.warning(f"Received 404 Not Found for {final_url}.")
                return {
                    "error": f"Page not found (404 Not Found) at {final_url}.",
                    "query_mode": bool(query),
                }
            response.raise_for_status()

            content_type = (
                response.headers.get("Content-Type", "").lower().split(";")[0].strip()
            )
            log.info(f"URL Content-Type detected: '{content_type}' from {final_url}")

            # --- Step 2: Parse content ---
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
                        f"Extracted text from PDF (Full Length: {len(extracted_text)})."
                    )
                except Exception as pdf_err:
                    log.error(
                        f"Error parsing PDF from {final_url}: {pdf_err}", exc_info=True
                    )
                    return {
                        "error": f"Failed to parse PDF content. Error: {pdf_err}",
                        "query_mode": bool(query),
                    }

            elif content_type == "text/html":
                content_source = "html_markdown"
                log.info("Parsing HTML content to Markdown...")
                extracted_text = self._parse_html_to_markdown(response.text)
                log.info(
                    f"Extracted text from HTML as Markdown (Full Length: {len(extracted_text)})."
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
                        "error": f"Failed to parse JSON content. Invalid format. Error: {json_err}",
                        "query_mode": bool(query),
                    }

            elif content_type.startswith("text/"):
                content_source = "text"
                log.info(f"Parsing plain text content ({content_type})...")
                extracted_text = self._clean_text_basic(response.text)
                log.info(f"Extracted plain text (Full Length: {len(extracted_text)}).")

            else:  # Fallback block
                log.warning(
                    f"Unsupported Content-Type '{content_type}' for URL {final_url}. Attempting fallback."
                )
                try:  # Inner try for fallback decoding
                    fallback_text = self._clean_text_basic(
                        response.content.decode(
                            response.encoding or "utf-8", errors="replace"
                        )
                    )
                    if fallback_text and len(fallback_text) > 50:  # Check decoded text
                        extracted_text = fallback_text  # Assign if good
                        content_source = "text_fallback"
                        log.info(
                            f"Extracted fallback text (Full Length: {len(extracted_text)})."
                        )
                    else:
                        # If no significant fallback text, return error immediately
                        log.warning(
                            f"No significant fallback text found for {final_url}."
                        )
                        return {
                            "error": f"Cannot browse: Unsupported content type '{content_type}' and no significant fallback text found.",
                            "query_mode": bool(query),
                        }
                except Exception as fallback_err:  # Inner except for fallback decoding
                    log.error(
                        f"Error during fallback text extraction for {final_url}: {fallback_err}"
                    )
                    return {
                        "error": f"Cannot browse: Unsupported content type '{content_type}' and fallback extraction failed.",
                        "query_mode": bool(query),
                    }
            # --- End Parsing Attempts ---

            # --- Check extracted text AFTER all parsing attempts ---
            # This block is now correctly positioned outside the 'else' for fallback
            if not extracted_text or not extracted_text.strip():
                log.warning(
                    f"Could not extract significant textual content ({content_source}) from {final_url}"
                )
                return {
                    "status": "success",
                    "action": "browse",
                    "url": final_url,
                    "content": None,
                    "message": f"No significant textual content found via {content_source} parser.",
                    "query_mode": bool(query),
                }

            # --- Step 3: Handle based on query ---
            if query and self.doc_archive_collection:
                log.info(
                    f"Query provided for {final_url}. Archiving and searching chunks in SHARED DB."
                )
                try:
                    doc_id = hashlib.sha256(final_url.encode("utf-8")).hexdigest()
                    log.debug(f"Generated doc_id '{doc_id}' for URL '{final_url}'")
                    existing_chunks = self.doc_archive_collection.get(
                        where={"doc_id": doc_id}, include=[]
                    )
                    if (
                        existing_chunks
                        and existing_chunks["ids"]
                        and len(existing_chunks["ids"]) > 0
                    ):
                        log.info(
                            f"Document '{doc_id}' already exists in SHARED archive ({len(existing_chunks['ids'])} chunks). Skipping add."
                        )
                    else:
                        log.info(
                            f"Document '{doc_id}' not found in SHARED archive. Chunking and adding..."
                        )
                        chunks = self.text_splitter.split_text(extracted_text)
                        if not chunks:
                            log.warning(
                                f"Text splitter returned no chunks for {final_url}."
                            )
                            return {
                                "status": "completed_with_issue",
                                "action": "browse",
                                "url": final_url,
                                "message": "Content extracted but failed to chunk for querying.",
                                "query_mode": True,
                            }
                        log.info(
                            f"Split content into {len(chunks)} chunks (Size: {config.DOC_ARCHIVE_CHUNK_SIZE}, Overlap: {config.DOC_ARCHIVE_CHUNK_OVERLAP})."
                        )
                        chunk_docs = []
                        chunk_metadatas = []
                        chunk_ids = []
                        for i, chunk_text in enumerate(chunks):
                            chunk_id = f"{doc_id}-{i}"
                            metadata = {
                                "doc_id": doc_id,
                                "url": final_url,
                                "chunk_index": i,
                                "source_type": content_source,
                                "timestamp": datetime.datetime.now(
                                    datetime.timezone.utc
                                ).isoformat(),
                            }
                            cleaned_metadata = {k: str(v) for k, v in metadata.items()}
                            chunk_docs.append(chunk_text)
                            chunk_metadatas.append(cleaned_metadata)
                            chunk_ids.append(chunk_id)
                        if chunk_ids:
                            log.info(
                                f"Adding {len(chunk_ids)} chunks to SHARED document archive..."
                            )
                            self.doc_archive_collection.add(
                                documents=chunk_docs,
                                metadatas=chunk_metadatas,
                                ids=chunk_ids,
                            )
                            log.info(
                                f"Successfully added chunks for document '{doc_id}' to SHARED archive."
                            )
                    log.info(
                        f"Querying SHARED archive for '{query[:50]}...' within doc '{doc_id}'..."
                    )
                    n_query_results = config.DOC_ARCHIVE_QUERY_RESULTS
                    query_results = self.doc_archive_collection.query(
                        query_texts=[query],
                        n_results=n_query_results,
                        where={"doc_id": doc_id},
                        include=["metadatas", "documents", "distances"],
                    )
                    snippets = []
                    if (
                        query_results
                        and query_results.get("ids")
                        and query_results["ids"][0]
                    ):
                        log.info(
                            f"Found {len(query_results['ids'][0])} relevant snippets in SHARED archive."
                        )
                        for i, chunk_id in enumerate(query_results["ids"][0]):
                            doc = query_results["documents"][0][i]
                            meta = query_results["metadatas"][0][i]
                            dist = query_results["distances"][0][i]
                            snippets.append(
                                {
                                    "chunk_id": chunk_id,
                                    "content": doc,
                                    "distance": f"{dist:.4f}",
                                    "chunk_index": meta.get("chunk_index", "N/A"),
                                }
                            )
                    else:
                        log.info(
                            "No relevant snippets found in SHARED archive for the query within this document."
                        )
                    return {
                        "status": "success",
                        "action": "browse",
                        "url": final_url,
                        "query_mode": True,
                        "query": query,
                        "retrieved_snippets": snippets,
                        "message": f"Focused retrieval performed on SHARED archived content. Found {len(snippets)} relevant snippets.",
                    }
                except Exception as archive_err:
                    log.exception(
                        f"Error during SHARED document archiving or querying for {final_url}: {archive_err}"
                    )
                    return {
                        "error": f"Error during document archiving/querying: {archive_err}",
                        "query_mode": True,
                    }

            elif query and not self.doc_archive_collection:
                log.error(
                    "Query provided for browse, but SHARED document archive DB is not available."
                )
                return {
                    "error": "Cannot perform query browse: SHARED document archive database is not configured or failed to initialize.",
                    "query_mode": True,
                }

            else:  # No query provided
                limit = config.MAX_CHARACTERS_CUMULATIVE_FINDINGS
                truncated = False
                message = "Content browsed successfully."
                if len(extracted_text) > limit:
                    log.info(
                        f"Truncated Content from {final_url} returned. {len(extracted_text)} characters truncated to {limit}."
                    )
                    extracted_text = extracted_text[:limit] + "\n\n..."
                    truncated = True
                    message = f"Content browsed successfully and returned first {limit} characters."
                else:
                    log.info(f"Full content from {final_url} returned.")
                result = {
                    "status": "success",
                    "action": "browse",
                    "url": final_url,
                    "content_source": content_source,
                    "content": extracted_text,
                    "truncated": truncated,
                    "message": message,
                    "query_mode": False,
                }
                log.info(
                    f"Successfully browsed and extracted content from {final_url} (Source: {content_source}, Length: {len(extracted_text)} chars, Truncated: {truncated})."
                )
                return result

        # --- Step 4: Error Handling (for main try) ---
        except requests.exceptions.Timeout:
            log.error(f"Request timed out while browsing {url}")
            return {
                "error": f"Request timed out accessing URL: {url}",
                "query_mode": bool(query),
            }
        except requests.exceptions.RequestException as e:
            log.error(f"Request failed for URL {url}: {e}")
            status_code = (
                e.response.status_code
                if hasattr(e, "response") and e.response is not None
                else "N/A"
            )
            error_url = (
                e.response.url
                if hasattr(e, "response") and e.response is not None
                else url
            )
            return {
                "error": f"Failed to retrieve URL {error_url}. Status: {status_code}. Error: {e}",
                "query_mode": bool(query),
            }
        except Exception as e:
            log.exception(f"Unexpected error during web browse for {url}: {e}")
            return {
                "error": f"Unexpected error browsing URL {url}: {e}",
                "query_mode": bool(query),
            }

    # run method logic remains the same
    def run(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
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
            query = parameters.get("query")
            if query is not None and (not isinstance(query, str) or not query.strip()):
                log.warning(
                    f"Received invalid 'query' parameter for browse (type: {type(query)}). Ignoring query."
                )
                query = None
            return self._run_browse(url, query)
        else:
            log.error(f"Invalid action specified for web tool: '{action}'")
            return {
                "error": "Invalid 'action' specified. Must be 'search' or 'browse'."
            }
