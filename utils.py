# FILE: utils.py
# autonomous_agent/utils.py
import os
import requests
import time
import datetime
import json
from typing import Optional, List, Tuple

# --- ChromaDB Imports ---
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# Import specific config values needed
from config import (
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
    OLLAMA_CHAT_MODEL,
    OLLAMA_CHAT_MODEL_REPEAT_PENALTY,
    OLLAMA_CHAT_MODEL_TEMPERATURE,
    OLLAMA_CHAT_MODEL_TOP_K,
    OLLAMA_CHAT_MODEL_TOP_P,
    OLLAMA_CHAT_MODEL_MIN_P,
    MODEL_CONTEXT_LENGTH,
    SHARED_ARTIFACT_FOLDER,
    SEARXNG_BASE_URL,
    SEARXNG_TIMEOUT,
    SHARED_DOC_ARCHIVE_DB_PATH,
    DOC_ARCHIVE_COLLECTION_NAME,
    OLLAMA_EMBED_MODEL,
)

import logging

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s][UTIL] %(message)s"
)  # Changed logger name
log = logging.getLogger(__name__)


def call_ollama_api(
    prompt: str,
    model: str = OLLAMA_CHAT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = OLLAMA_TIMEOUT,
) -> Optional[str]:
    """Calls the Ollama chat API and returns the content of the response message."""
    api_url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "num_ctx": MODEL_CONTEXT_LENGTH,
            "repeat_penalty": OLLAMA_CHAT_MODEL_REPEAT_PENALTY,
            "temperature": OLLAMA_CHAT_MODEL_TEMPERATURE,
            "top_k": OLLAMA_CHAT_MODEL_TOP_K,
            "top_p": OLLAMA_CHAT_MODEL_TOP_P,
            "min_p": OLLAMA_CHAT_MODEL_MIN_P,
        },
    }
    headers = {"Content-Type": "application/json"}
    max_retries = 1
    retry_delay = 3

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                api_url, json=payload, headers=headers, timeout=timeout
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            response_data = response.json()
            if "message" in response_data and "content" in response_data["message"]:
                input_tokens = response_data["prompt_eval_count"]
                input_tps = round(
                    input_tokens / response_data["prompt_eval_duration"] * 1e9, 1
                )
                resp_tokens = response_data["eval_count"]
                resp_tps = round(resp_tokens / response_data["eval_duration"] * 1e9, 1)
                log.info(
                    f"Ollama call successful: Input token count {input_tokens} ({input_tps} tps); Generated token count {resp_tokens} ({resp_tps} tps)..."
                )
                return response_data["message"]["content"]
            elif "error" in response_data:
                log.error(f"Error from Ollama API ({model}): {response_data['error']}")
                return None
            else:
                log.error(
                    f"Unexpected Ollama response structure ({model}): {response_data}"
                )
                return None
        except requests.exceptions.Timeout:
            log.error(f"Ollama API request timed out after {timeout}s ({model}).")
            if attempt < max_retries:
                log.info(f"Retrying Ollama call in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                log.error("Max retries reached for Ollama timeout.")
                return None
        except requests.exceptions.RequestException as e:
            log.error(f"Error calling Ollama API ({model}) at {api_url}: {e}")
            if hasattr(e, "response") and e.response is not None:
                log.error(
                    f"  Response status: {e.response.status_code}, Text: {e.response.text[:200]}..."
                )
            return None
        except Exception as e:
            log.exception(f"Unexpected error calling Ollama ({model}): {e}")
            return None
    return None


# --- Document Archive ChromaDB Setup (Shared) ---
def setup_doc_archive_chromadb() -> Optional[chromadb.Collection]:
    """Initializes ChromaDB client and collection for the SHARED document archive."""
    log.info(
        f"Initializing SHARED Document Archive ChromaDB client at path: {SHARED_DOC_ARCHIVE_DB_PATH}"
    )
    try:
        # Ensure the shared directory exists before initializing client
        os.makedirs(SHARED_DOC_ARCHIVE_DB_PATH, exist_ok=True)

        settings = chromadb.Settings(anonymized_telemetry=False)
        doc_vector_db = chromadb.PersistentClient(
            path=SHARED_DOC_ARCHIVE_DB_PATH, settings=settings
        )
        log.info(
            f"Getting or creating SHARED Doc Archive ChromaDB collection '{DOC_ARCHIVE_COLLECTION_NAME}'"
        )
        embedding_function = OllamaEmbeddingFunction(
            url=f"{OLLAMA_BASE_URL}/api/embeddings", model_name=OLLAMA_EMBED_MODEL
        )
        doc_archive_collection = doc_vector_db.get_or_create_collection(
            name=DOC_ARCHIVE_COLLECTION_NAME,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("SHARED Document Archive ChromaDB collection ready.")
        return doc_archive_collection
    except Exception as e:
        log.critical(
            f"Could not initialize SHARED Document Archive ChromaDB at {SHARED_DOC_ARCHIVE_DB_PATH}.",
            exc_info=True,
        )
        return None


# --- End Document Archive ChromaDB Setup ---


# --- Status Check Functions (Unchanged) ---
def check_ollama_status(
    base_url: str = OLLAMA_BASE_URL, timeout: int = 5
) -> Tuple[bool, str]:
    """Checks if the Ollama API is reachable and responding."""
    if not base_url:
        return False, "Ollama base URL not configured."
    api_url = f"{base_url}/api/tags"
    try:
        response = requests.get(api_url, timeout=timeout)
        response.raise_for_status()
        response.json()
        return True, "OK"
    except requests.exceptions.Timeout:
        return False, f"Timeout ({timeout}s)"
    except requests.exceptions.ConnectionError:
        return False, "Connection Error"
    except requests.exceptions.RequestException as e:
        status = (
            e.response.status_code
            if hasattr(e, "response") and e.response is not None
            else "N/A"
        )
        return False, f"Request Error (Status: {status})"
    except json.JSONDecodeError:
        return False, "Invalid JSON Response (API might be up but malfunctioning)"
    except Exception as e:
        return False, f"Unexpected Error: {str(e)[:50]}"


def check_searxng_status(
    base_url: str = SEARXNG_BASE_URL, timeout: int = SEARXNG_TIMEOUT
) -> Tuple[bool, str]:
    """Checks if the SearXNG instance is reachable."""
    if not base_url:
        return False, "SearXNG base URL not configured."
    try:
        response = requests.get(
            base_url, timeout=timeout, headers={"Accept": "text/html"}
        )
        response.raise_for_status()
        if "<html" not in response.text.lower():
            return (
                True,
                "OK (Unexpected Content)",
            )
        return True, "OK"
    except requests.exceptions.Timeout:
        return False, f"Timeout ({timeout}s)"
    except requests.exceptions.ConnectionError:
        return False, "Connection Error"
    except requests.exceptions.RequestException as e:
        status = (
            e.response.status_code
            if hasattr(e, "response") and e.response is not None
            else "N/A"
        )
        return False, f"Request Error (Status: {status})"
    except Exception as e:
        return False, f"Unexpected Error: {str(e)[:50]}"


# --- End Status Check Functions ---


# --- Time Formatting (Unchanged) ---
def format_relative_time(timestamp_str: Optional[str]) -> str:
    """Converts an ISO timestamp string into a user-friendly relative time string."""
    if not timestamp_str:
        return "Time N/A"
    try:
        if timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str[:-1] + "+00:00"
        event_time = datetime.datetime.fromisoformat(timestamp_str).replace(
            tzinfo=datetime.timezone.utc
        )
        now = datetime.datetime.now(datetime.timezone.utc)
        delta = now - event_time

        seconds = delta.total_seconds()
        if seconds < 0:
            return "in future?"
        elif seconds < 60:
            return "<1 min ago"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} min{'s' if minutes > 1 else ''} ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif seconds < 172800:
            if (now.date() - event_time.date()).days == 1:
                return "yesterday"
            else:
                return "1 day ago"
        elif seconds < 604800:
            days = int(seconds / 86400)
            return f"{days} days ago"
        elif seconds < 1209600:
            return "last week"
        elif seconds < 2592000:
            weeks = int(seconds / 604800)
            return f"{weeks} week{'s' if weeks > 1 else ''} ago"
        elif seconds < 5184000:
            return "1 month ago"
        else:
            return ">1 month ago"

    except ValueError:
        log.warning(f"Could not parse timestamp for relative time: {timestamp_str}")
        return "Invalid Time"
    except Exception as e:
        log.error(f"Error formatting relative time for {timestamp_str}: {e}")
        return "Time Error"


# --- File Path Utilities (Updated to use SHARED_ARTIFACT_FOLDER) ---
def sanitize_and_validate_path(
    filename: str,
    base_artifact_path: str = SHARED_ARTIFACT_FOLDER,  # Use shared path by default
) -> Tuple[bool, str, Optional[str], Optional[str]]:
    """
    Sanitizes the filename and validates that it points to a path within the SHARED artifact workspace.
    Returns: (is_valid, message, full_path, relative_filename)
    """
    try:
        normalized_filename = os.path.normpath(filename).lstrip(os.sep)
        joined_path = os.path.join(base_artifact_path, normalized_filename)
        normalized_joined_path = os.path.normpath(joined_path)
        # Check against the absolute path of the shared folder
        abs_base_path = os.path.abspath(base_artifact_path)
        abs_joined_path = os.path.abspath(normalized_joined_path)

        if not abs_joined_path.startswith(abs_base_path):
            log.warning(
                f"Path validation failed: Attempted access outside of designated shared workspace. Target: '{abs_joined_path}', Base: '{abs_base_path}'"
            )
            return (
                False,
                "Path validation failed: Attempted access outside of designated shared workspace.",
                None,
                None,
            )
        full_path = abs_joined_path
    except Exception as e:
        log.error(f"Error resolving path for '{filename}': {e}")
        return False, f"Internal error resolving path: {e}", None, None

    max_path_len = 255
    if len(full_path) > max_path_len:
        return (
            False,
            f"Resulting file path is too long (>{max_path_len} chars).",
            None,
            None,
        )

    log.debug(f"Path validated: '{filename}' -> '{full_path}'")
    relative_path = os.path.relpath(full_path, abs_base_path)
    relative_path = "." if relative_path == os.curdir else relative_path
    return True, "Path validated successfully.", full_path, relative_path


def list_directory_contents(directory_path: str) -> Tuple[bool, List[str]]:
    """Safely lists contents of a directory within the SHARED artifact workspace."""
    try:
        base_artifact_path = os.path.abspath(SHARED_ARTIFACT_FOLDER)  # Use shared path
        abs_directory_path = os.path.abspath(directory_path)
        if not abs_directory_path.startswith(base_artifact_path):
            log.error(
                f"Security Error: Attempted to list directory outside shared workspace: {abs_directory_path}"
            )
            return False, []
        if not os.path.isdir(abs_directory_path):
            log.warning(
                f"Cannot list contents, path is not a directory: {abs_directory_path}"
            )
            return False, []
        contents = os.listdir(abs_directory_path)
        log.info(f"Listed {len(contents)} items in directory: {abs_directory_path}")
        return True, contents
    except OSError as e:
        log.error(f"OS error listing directory '{directory_path}': {e}")
        return False, []
    except Exception as e:
        log.exception(f"Unexpected error listing directory '{directory_path}': {e}")
        return False, []
