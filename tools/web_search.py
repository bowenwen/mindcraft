# autonomous_agent/tools/web_search.py
import requests
import json
import traceback
from typing import Dict, Any

# Import base class and config
from .base import Tool
from config import SEARXNG_BASE_URL, SEARXNG_TIMEOUT

# Suggestion: Use logging module
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s][TOOL_WebSearch] %(message)s')
log = logging.getLogger(__name__)


class WebSearchTool(Tool):
    """Tool for performing web searches using a SearXNG instance."""
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Performs a web search using a SearXNG instance to find information. Parameters: query (str, required)"
        )
        # Config check message moved to agent init for better flow

    def run(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a search query against the configured SearXNG instance."""
        if not SEARXNG_BASE_URL: return {"error": "SearXNG base URL is not configured."}
        if not isinstance(parameters, dict): log.error(f"Invalid params type: {type(parameters)}"); return {"error": "Invalid parameters format."}
        query = parameters.get("query")
        if not query or not isinstance(query, str) or not query.strip(): return {"error": "Missing or invalid 'query' parameter."}

        log.info(f"Executing SearXNG Search: '{query}'")
        params = {'q': query, 'format': 'json'}
        searxng_url = SEARXNG_BASE_URL.rstrip('/') + "/search"
        headers = {'Accept': 'application/json'}

        try:
            response = requests.get(searxng_url, params=params, headers=headers, timeout=SEARXNG_TIMEOUT)
            response.raise_for_status() # Check for HTTP errors

            try: search_data = response.json()
            except json.JSONDecodeError: log.error(f"Failed SearXNG JSON decode. Status: {response.status_code}. Resp: {response.text[:200]}..."); return {"error": "Failed JSON decode from SearXNG."}

            processed = []
            # Process infoboxes
            if 'infoboxes' in search_data and isinstance(search_data['infoboxes'], list):
                for ib in search_data['infoboxes']:
                    if isinstance(ib, dict): content = ib.get('content'); title = ib.get('infobox_type', 'Infobox'); url = None;
                    if ib.get('urls') and isinstance(ib['urls'], list) and len(ib['urls']) > 0 and isinstance(ib['urls'][0], dict): url = ib['urls'][0].get('url');
                    if content: processed.append({"type": "infobox", "title": title, "snippet": content, "url": url})
            # Process regular results
            if 'results' in search_data and isinstance(search_data['results'], list):
                 for r in search_data['results'][:7]: # Limit results
                     if isinstance(r, dict): title = r.get('title'); url = r.get('url'); snippet = r.get('content');
                     if title and url and snippet: processed.append({"type": "organic", "title": title, "snippet": snippet, "url": url})
            # Process answers
            if 'answers' in search_data and isinstance(search_data['answers'], list):
                 for ans in search_data['answers']:
                      if isinstance(ans, str) and ans.strip(): processed.append({"type": "answer", "title": "Direct Answer", "snippet": ans, "url": None})

            if not processed:
                if 'error' in search_data: log.warning(f"SearXNG reported error: {search_data['error']}"); return {"error": f"SearXNG error: {search_data['error']}"}
                else: log.info("No relevant results found by SearXNG."); return {"results": [], "message": "No relevant results found."}

            log.info(f"SearXNG returned {len(processed)} processed results.")
            return {"results": processed}

        except requests.exceptions.Timeout: log.error(f"SearXNG request timed out ({SEARXNG_TIMEOUT}s) to {searxng_url}"); return {"error": f"Web search timed out."}
        except requests.exceptions.RequestException as e: log.error(f"SearXNG request failed: {e}"); return {"error": f"Web search connection/query failed."}
        except Exception as e: log.exception(f"Unexpected error during SearXNG search: {e}"); return {"error": f"Unexpected web search error."} # Log stack trace