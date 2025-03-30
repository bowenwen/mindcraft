# autonomous_agent/tools/web_browse.py
import requests
import re
import traceback
import json # <<<--- ADD JSON import
from typing import Dict, Any
from bs4 import BeautifulSoup
import fitz # PyMuPDF

# Import base class and config
from .base import Tool
import config

# Use logging module
import logging
# Ensure basicConfig is called in main.py, just get logger here
log = logging.getLogger("TOOL_WebBrowse") # Use specific logger name


class WebBrowserTool(Tool):
    """
    Tool to fetch the textual content of a specific web page URL.
    Supports HTML, PDF, JSON, and plain text documents.
    NOTE: HTML parsing primarily works for static content.
          PDF/text extraction depends on the file's structure.
    """
    def __init__(self):
        super().__init__(
            name="web_browse",
            description=(
                "Fetches the primary textual content from a given URL (HTML, PDF, JSON, TXT). " # Updated description
                "Useful for reading articles, blog posts, documentation, PDF reports, JSON data, or plain text files when you have the URL. "
                "Parameters: url (str, required)"
            )
        )
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Tags for HTML extraction
        self.content_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'article', 'main', 'section', 'pre', 'code']

    def _clean_text(self, text: str) -> str:
        """Removes excessive whitespace and performs basic cleaning."""
        if not text:
            return ""
        # Replace multiple whitespace characters (newlines, tabs, spaces) with a single space
        text = re.sub(r'\s+', ' ', text)
        # You could add more specific cleaning here if needed
        return text.strip()

    def run(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Fetches and extracts text content from the provided URL."""
        if not isinstance(parameters, dict):
            log.error(f"Invalid params type: {type(parameters)}")
            return {"error": "Invalid parameters format. Expected a JSON object (dict)."}

        url = parameters.get("url")
        if not url or not isinstance(url, str) or not url.strip():
            return {"error": "Missing or invalid 'url' parameter (must be a non-empty string)."}

        if not url.startswith(("http://", "https://")):
            # Attempting without scheme can lead to errors, better to require it
            log.error(f"Invalid URL format: '{url}'. Must start with http:// or https://.")
            return {"error": "Invalid URL format. Must start with http:// or https://."}
            # Alternatively: try adding 'https://' but that's guesswork
            # log.warning(f"URL '{url}' missing scheme. Prepending https://.")
            # url = f"https://{url}"


        log.info(f"Attempting to browse URL: {url}")

        extracted_text = ""
        content_source = "unknown"

        try:
            response = requests.get(url, headers=self.headers, timeout=25)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '').lower().split(';')[0].strip() # Get main type before charset etc.
            log.info(f"URL Content-Type detected: '{content_type}'")

            # --- PDF Handling ---
            if content_type == 'application/pdf':
                content_source = "pdf"
                log.info("Parsing PDF content...")
                pdf_texts = []
                try:
                    doc = fitz.open(stream=response.content, filetype="pdf")
                    log.info(f"PDF has {len(doc)} pages.")
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        page_text = page.get_text("text")
                        if page_text: pdf_texts.append(page_text)
                    doc.close()
                    full_text = "\n".join(pdf_texts) # Join pages with newline
                    extracted_text = self._clean_text(full_text) # Clean after joining
                    log.info(f"Extracted text from PDF (Length: {len(extracted_text)}).")
                except Exception as pdf_err:
                    log.error(f"Error parsing PDF from {url}: {pdf_err}", exc_info=True)
                    return {"error": f"Failed to parse PDF content. Error: {pdf_err}"}

            # --- HTML Handling ---
            elif content_type == 'text/html':
                content_source = "html"
                log.info("Parsing HTML content...")
                soup = BeautifulSoup(response.text, 'html.parser')
                extracted_texts = []
                main_content = soup.find('main') or soup.find('article') or soup.find('body')
                if main_content:
                     elements = main_content.find_all(self.content_tags)
                     if not elements: extracted_texts.append(self._clean_text(main_content.get_text(separator=' ', strip=True)))
                     else: extracted_texts.extend(self._clean_text(el.get_text(separator=' ', strip=True)) for el in elements)
                else: extracted_texts.extend(self._clean_text(el.get_text(separator=' ', strip=True)) for el in soup.find_all(self.content_tags))
                full_text = " ".join(filter(None, extracted_texts)) # Join non-empty strings
                extracted_text = self._clean_text(full_text) # Final cleanup
                log.info(f"Extracted text from HTML (Length: {len(extracted_text)}).")

            # --- JSON Handling ---
            elif content_type == 'application/json':
                 content_source = "json"
                 log.info("Parsing JSON content...")
                 try:
                     json_data = json.loads(response.text)
                     # Pretty-print JSON back into a string for the LLM
                     extracted_text = json.dumps(json_data, indent=2, ensure_ascii=False)
                     # No extra cleaning needed for JSON string usually
                     log.info(f"Formatted JSON content (Length: {len(extracted_text)}).")
                 except json.JSONDecodeError as json_err:
                     log.error(f"Invalid JSON received from {url}: {json_err}")
                     log.debug(f"Raw JSON text received: {response.text[:500]}...") # Log snippet
                     return {"error": f"Failed to parse JSON content. Invalid format. Error: {json_err}"}

            # --- Plain Text Handling ---
            elif content_type.startswith('text/'): # Catch text/plain, text/csv, etc.
                 content_source = "text"
                 log.info(f"Parsing plain text content ({content_type})...")
                 # Use response.text directly and clean it
                 extracted_text = self._clean_text(response.text)
                 log.info(f"Extracted plain text (Length: {len(extracted_text)}).")

            # --- Other Content Types ---
            else:
                log.warning(f"Unsupported Content-Type '{content_type}' for URL {url}.")
                # Maybe try a generic text extraction as a fallback? Or just fail.
                # Option: try generic text extraction
                try:
                     log.info("Attempting generic text extraction as fallback...")
                     fallback_text = self._clean_text(response.text)
                     if fallback_text and len(fallback_text) > 50: # Heuristic: only keep if substantial text found
                          extracted_text = fallback_text
                          content_source = "text_fallback"
                          log.info(f"Extracted fallback text (Length: {len(extracted_text)}).")
                     else:
                           return {"error": f"Cannot browse unsupported content type: {content_type}"}
                except Exception:
                     return {"error": f"Cannot browse unsupported content type: {content_type}"}


            # --- Post-processing ---
            if not extracted_text:
                log.warning(f"Could not extract significant textual content ({content_source}) from {url}")
                return {"url": url, "content": None, "message": f"No significant textual content found via {content_source} parser."}

            # Truncate content if it's too long
            limit = config.CONTEXT_TRUNCATION_LIMIT
            truncated = False
            if len(extracted_text) > limit:
                log.info(f"Content from {url} truncated from {len(extracted_text)} to {limit} characters.")
                # Simple truncation, could be smarter (e.g., preserve JSON structure)
                extracted_text = extracted_text[:limit]
                truncated = True

            result = {
                "url": url,
                "content_source": content_source,
                "content": extracted_text
            }
            if truncated:
                result["message"] = f"Content truncated to {limit} characters."

            log.info(f"Successfully browsed and extracted content from {url} (Source: {content_source}, Length: {len(extracted_text)} chars).")
            return result

        except requests.exceptions.Timeout:
            log.error(f"Request timed out while browsing {url}")
            return {"error": f"Request timed out accessing URL: {url}"}
        except requests.exceptions.RequestException as e:
            # Catch connection errors, HTTP errors, etc.
            log.error(f"Request failed for URL {url}: {e}")
            return {"error": f"Failed to retrieve URL {url}. Error: {e}"}
        except Exception as e:
            log.exception(f"Unexpected error during web browse for {url}: {e}")
            return {"error": f"Unexpected error browsing URL {url}: {e}"}