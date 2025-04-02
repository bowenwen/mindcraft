# FILE: tools/web_browse.py
import requests
import re
import traceback
import json
import io  # For handling PDF stream
from typing import Dict, Any, List, Tuple
from bs4 import BeautifulSoup, NavigableString, Tag
import fitz  # PyMuPDF

# Import base class and config
from .base import Tool
import config

# Use logging module
import logging

log = logging.getLogger("TOOL_WebBrowse")  # Use specific logger name


class WebBrowserTool(Tool):
    """
    Tool to fetch and parse content from a specific web page URL into a Markdown-like format.
    Supports HTML, PDF, JSON, and plain text documents.
    NOTE: HTML parsing focuses on semantic tags and structure; complex JS-rendered sites may not parse well.
          PDF/text extraction quality depends on the file's structure.
    """

    def __init__(self):
        super().__init__(
            name="web_browse",
            description=(
                "Fetches and parses content from a given URL (HTML, PDF, JSON, TXT) into a readable format (often Markdown-like for HTML). "
                "Useful for reading articles, documentation, reports, or data when you have the URL. "
                "Parameters: url (str, required)"
            ),
        )
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,application/pdf,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    def _clean_text_basic(self, text: str) -> str:
        """Removes excessive whitespace but tries to preserve single newlines."""
        if not text:
            return ""
        # Replace multiple spaces/tabs with single space, but keep newlines mostly intact
        text = re.sub(r"[ \t]+", " ", text)
        # Replace 3 or more newlines with two (like paragraphs)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_text_from_html_node(self, node: Tag) -> str:
        """Recursively extracts text from HTML node with Markdown-like formatting."""
        if isinstance(node, NavigableString):
            # Basic cleaning for NavigableString content
            return re.sub(r"\s+", " ", str(node)).strip()

        if not isinstance(node, Tag):
            return ""

        content = []
        tag_name = node.name.lower()

        # --- Formatting based on tag ---
        prefix = ""
        suffix = "\n\n"  # Default: treat as block element

        if tag_name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            level = int(tag_name[1])
            prefix = "#" * level + " "
        elif tag_name == "p":
            pass  # Standard paragraph block
        elif tag_name in ["ul", "ol"]:
            # Process list items individually
            for i, item in enumerate(node.find_all("li", recursive=False)):
                item_prefix = "* " if tag_name == "ul" else f"{i+1}. "
                item_content = self._extract_text_from_html_node(item).strip()
                if item_content:
                    content.append(item_prefix + item_content)
            suffix = "\n"  # Tighter spacing after list
            prefix = ""  # Handled by items
        elif tag_name == "li":
            # Usually handled by ul/ol logic, but process directly if needed
            pass  # Handled by parent list logic mostly
        elif tag_name in ["pre", "code"]:
            # If it's a block of code, wrap it
            code_text = node.get_text(strip=True)
            if "\n" in code_text:  # Treat as block if multiple lines
                prefix = "```\n"
                suffix = "\n```\n\n"
                content = [code_text]  # Use raw text
            else:  # Treat as inline
                prefix = "`"
                suffix = "` "  # Add space after inline code
                content = [code_text]
            # Prevent recursion into code blocks
            return prefix + "".join(content) + suffix
        elif tag_name == "a":
            href = node.get("href")
            text = node.get_text(strip=True)
            if text and href:
                # Avoid relative links or JS links for now
                if href.startswith(("http://", "https://")):
                    return f"[{text}]({href})"
                else:
                    return text  # Just return text if link isn't useful
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
            return "\n"  # Force newline
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
            return ""  # Skip noise/non-content tags
        elif tag_name in ["div", "span", "section", "article", "main", "td", "th"]:
            # These are containers, process children but don't add much formatting themselves
            # Use single newline suffix for tighter integration for spans/divs maybe?
            suffix = "\n" if tag_name in ["span"] else "\n\n"
            pass
        elif tag_name in ["table"]:
            # Basic table representation (could be improved)
            content.append(
                "\n| Header 1 | Header 2 | ... |\n|---|---|---|"
            )  # Placeholder
            for row in node.find_all("tr"):
                row_content = [
                    self._extract_text_from_html_node(cell).strip()
                    for cell in row.find_all(["th", "td"])
                ]
                if row_content:
                    content.append("| " + " | ".join(row_content) + " |")
            return "\n".join(content) + "\n\n"

        # --- Recursively process children if not already handled (like code/table) ---
        if not content:  # If content wasn't explicitly set by tag logic above
            child_contents = []
            for child in node.children:
                child_text = self._extract_text_from_html_node(child)
                if child_text:  # Only add if child produced text
                    child_contents.append(child_text)
            # Join children with minimal space, let parent suffix handle block spacing
            content = [" ".join(child_contents).strip()]

        # Combine prefix, content, suffix
        full_content = prefix + "".join(content) + suffix

        # Apply basic cleaning to the *result* of this node processing
        return self._clean_text_basic(full_content)

    def _parse_html_to_markdown(self, html_content: str) -> str:
        """Attempts to parse HTML into a Markdown-like text format."""
        soup = BeautifulSoup(html_content, "html.parser")
        # Try to find the main content area
        body = soup.find("body")
        if not body:
            return self._clean_text_basic(soup.get_text())  # Fallback if no body

        # Prioritize common main content containers
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
                if selector == "body":  # Use the already found body as last resort
                    main_node = body
                    break
                node = body.select_one(selector)
                if node:
                    log.debug(f"Found main content using selector: '{selector}'")
                    main_node = node
                    break
            except Exception:  # Handle invalid selectors if necessary
                log.warning(f"Selector '{selector}' failed.", exc_info=False)
                continue

        if not main_node:
            log.warning(
                "Could not identify specific main content area, parsing entire body."
            )
            main_node = body  # Fallback to body

        # Extract text from the selected main node
        extracted_text = self._extract_text_from_html_node(main_node)

        # Final cleanup pass on the whole markdown document
        lines = extracted_text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:  # Keep non-empty lines
                cleaned_lines.append(stripped_line)

        # Join lines, ensuring paragraphs have double newlines
        final_text = ""
        for i, line in enumerate(cleaned_lines):
            final_text += line
            # Add double newline after potential block elements unless it's the last line
            # or the next line is also a block element (like list item or heading)
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
                    final_text += (
                        "\n"  # Single newline otherwise (e.g., between list items)
                    )

        return final_text.strip()

    def run(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Fetches and extracts text content from the provided URL."""
        if not isinstance(parameters, dict):
            log.error(f"Invalid params type: {type(parameters)}")
            return {
                "error": "Invalid parameters format. Expected a JSON object (dict)."
            }

        url = parameters.get("url")
        if not url or not isinstance(url, str) or not url.strip():
            return {
                "error": "Missing or invalid 'url' parameter (must be a non-empty string)."
            }

        if not url.startswith(("http://", "https://")):
            log.error(
                f"Invalid URL format: '{url}'. Must start with http:// or https://."
            )
            return {"error": "Invalid URL format. Must start with http:// or https://."}

        log.info(f"Attempting to browse URL: {url}")

        extracted_text = ""
        content_source = "unknown"

        try:
            response = requests.get(url, headers=self.headers, timeout=25)
            response.raise_for_status()
            content_type = (
                response.headers.get("Content-Type", "").lower().split(";")[0].strip()
            )
            log.info(f"URL Content-Type detected: '{content_type}'")

            # --- PDF Handling ---
            if content_type == "application/pdf":
                content_source = "pdf"
                log.info("Parsing PDF content...")
                pdf_texts = []
                try:
                    # Use io.BytesIO to handle the content stream directly
                    pdf_stream = io.BytesIO(response.content)
                    doc = fitz.open(stream=pdf_stream, filetype="pdf")
                    log.info(f"PDF has {len(doc)} pages.")
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        page_text = page.get_text(
                            "text", sort=True
                        )  # Try sorting text blocks
                        if page_text:
                            # Add page separator
                            pdf_texts.append(f"\n--- Page {page_num + 1} ---\n")
                            pdf_texts.append(page_text.strip())
                    doc.close()
                    full_text = "\n".join(pdf_texts).strip()  # Join pages
                    extracted_text = self._clean_text_basic(
                        full_text
                    )  # Basic clean after joining
                    log.info(
                        f"Extracted text from PDF (Length: {len(extracted_text)})."
                    )
                except Exception as pdf_err:
                    log.error(f"Error parsing PDF from {url}: {pdf_err}", exc_info=True)
                    return {"error": f"Failed to parse PDF content. Error: {pdf_err}"}

            # --- HTML Handling (using new Markdown parser) ---
            elif content_type == "text/html":
                content_source = "html_markdown"
                log.info("Parsing HTML content to Markdown...")
                extracted_text = self._parse_html_to_markdown(response.text)
                log.info(
                    f"Extracted text from HTML as Markdown (Length: {len(extracted_text)})."
                )

            # --- JSON Handling ---
            elif content_type == "application/json":
                content_source = "json"
                log.info("Parsing JSON content...")
                try:
                    json_data = json.loads(response.text)
                    extracted_text = json.dumps(json_data, indent=2, ensure_ascii=False)
                    log.info(f"Formatted JSON content (Length: {len(extracted_text)}).")
                except json.JSONDecodeError as json_err:
                    log.error(f"Invalid JSON received from {url}: {json_err}")
                    return {
                        "error": f"Failed to parse JSON content. Invalid format. Error: {json_err}"
                    }

            # --- Plain Text Handling ---
            elif content_type.startswith("text/"):
                content_source = "text"
                log.info(f"Parsing plain text content ({content_type})...")
                extracted_text = self._clean_text_basic(response.text)
                log.info(f"Extracted plain text (Length: {len(extracted_text)}).")

            # --- Other Content Types (Fallback) ---
            else:
                log.warning(
                    f"Unsupported Content-Type '{content_type}' for URL {url}. Attempting fallback."
                )
                try:
                    fallback_text = self._clean_text_basic(response.text)
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
                        f"Error during fallback text extraction for {url}: {fallback_err}"
                    )
                    return {
                        "error": f"Cannot browse: Unsupported content type '{content_type}' and fallback extraction failed."
                    }

            # --- Post-processing ---
            if not extracted_text:
                log.warning(
                    f"Could not extract significant textual content ({content_source}) from {url}"
                )
                return {
                    "url": url,
                    "content": None,
                    "message": f"No significant textual content found via {content_source} parser.",
                }

            # Truncate content if it's too long
            limit = config.CONTEXT_TRUNCATION_LIMIT
            truncated = False
            if len(extracted_text) > limit:
                log.info(
                    f"Content from {url} truncated from {len(extracted_text)} to {limit} characters."
                )
                # Truncate and add note
                extracted_text = extracted_text[:limit] + "\n\n... [CONTENT TRUNCATED]"
                truncated = True

            result = {
                "url": url,
                "content_source": content_source,
                "content": extracted_text,
            }
            if truncated:
                result["message"] = (
                    f"Content was truncated to approximately {limit} characters."
                )

            log.info(
                f"Successfully browsed and extracted content from {url} (Source: {content_source}, Length: {len(extracted_text)} chars)."
            )
            return result

        except requests.exceptions.Timeout:
            log.error(f"Request timed out while browsing {url}")
            return {"error": f"Request timed out accessing URL: {url}"}
        except requests.exceptions.RequestException as e:
            log.error(f"Request failed for URL {url}: {e}")
            # Include status code if available
            status_code = (
                e.response.status_code
                if hasattr(e, "response") and e.response is not None
                else "N/A"
            )
            return {
                "error": f"Failed to retrieve URL {url}. Status: {status_code}. Error: {e}"
            }
        except Exception as e:
            log.exception(f"Unexpected error during web browse for {url}: {e}")
            return {"error": f"Unexpected error browsing URL {url}: {e}"}
