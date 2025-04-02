# FILE: utils.py
# autonomous_agent/llm_utils.py
import os
import requests
import time
import datetime
import json
from typing import Optional, List, Tuple

# Import specific config values needed
from config import (
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
    MODEL_CONTEXT_LENGTH,
    ARTIFACT_FOLDER,
    SEARXNG_BASE_URL, # Added for status check
    SEARXNG_TIMEOUT, # Added for status check
)

# Suggestion: Consider adding basic logging setup here or importing from a dedicated logging module
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s][LLM] %(message)s")
log = logging.getLogger(__name__)


def call_ollama_api(
    prompt: str,
    model: str,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = OLLAMA_TIMEOUT,
) -> Optional[str]:
    """Calls the Ollama chat API and returns the content of the response message."""
    api_url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.8, "num_ctx": MODEL_CONTEXT_LENGTH},
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

# --- NEW: Status Check Functions ---
def check_ollama_status(base_url: str = OLLAMA_BASE_URL, timeout: int = 5) -> Tuple[bool, str]:
    """Checks if the Ollama API is reachable and responding."""
    if not base_url:
        return False, "Ollama base URL not configured."
    # Try a lightweight endpoint, like listing models or just hitting the base URL
    api_url = f"{base_url}/api/tags" # Check models endpoint
    try:
        response = requests.get(api_url, timeout=timeout)
        response.raise_for_status()
        # Check if response is valid JSON (expected for /api/tags)
        response.json()
        return True, "OK"
    except requests.exceptions.Timeout:
        return False, f"Timeout ({timeout}s)"
    except requests.exceptions.ConnectionError:
        return False, "Connection Error"
    except requests.exceptions.RequestException as e:
        status = e.response.status_code if hasattr(e, 'response') and e.response is not None else 'N/A'
        return False, f"Request Error (Status: {status})"
    except json.JSONDecodeError:
        return False, "Invalid JSON Response (API might be up but malfunctioning)"
    except Exception as e:
        return False, f"Unexpected Error: {str(e)[:50]}"

def check_searxng_status(base_url: str = SEARXNG_BASE_URL, timeout: int = 5) -> Tuple[bool, str]:
    """Checks if the SearXNG instance is reachable."""
    if not base_url:
        return False, "SearXNG base URL not configured."
    try:
        # Just check if the base URL is accessible
        response = requests.get(base_url, timeout=timeout, headers={'Accept': 'text/html'})
        response.raise_for_status()
        # Basic check for expected HTML content
        if "<html" not in response.text.lower():
            return True, "OK (Unexpected Content)" # Reachable, but maybe not SearXNG homepage
        return True, "OK"
    except requests.exceptions.Timeout:
        return False, f"Timeout ({timeout}s)"
    except requests.exceptions.ConnectionError:
        return False, "Connection Error"
    except requests.exceptions.RequestException as e:
        status = e.response.status_code if hasattr(e, 'response') and e.response is not None else 'N/A'
        return False, f"Request Error (Status: {status})"
    except Exception as e:
        return False, f"Unexpected Error: {str(e)[:50]}"
# --- End NEW Status Check Functions ---


def format_relative_time(timestamp_str: Optional[str]) -> str:
    """Converts an ISO timestamp string into a user-friendly relative time string."""
    if not timestamp_str:
        return "Time N/A"
    try:
        # Handle potential 'Z' for UTC timezone
        if timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str[:-1] + "+00:00"
        event_time = datetime.datetime.fromisoformat(timestamp_str).replace(
            tzinfo=datetime.timezone.utc
        )
        now = datetime.datetime.now(datetime.timezone.utc)
        delta = now - event_time

        seconds = delta.total_seconds()
        if seconds < 0:
            return "in future?"  # Should not happen for memories
        elif seconds < 60:
            return "<1 min ago"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} min{'s' if minutes > 1 else ''} ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif seconds < 172800:  # Less than 2 days
            if (now.date() - event_time.date()).days == 1:
                return "yesterday"
            else:
                return "1 day ago"  # Could be same day if time crosses midnight UTC vs local
        elif seconds < 604800:  # Less than 7 days
            days = int(seconds / 86400)
            return f"{days} days ago"
        elif seconds < 1209600:  # Less than 14 days
            return "last week"
        elif seconds < 2592000:  # Approx 30 days
            weeks = int(seconds / 604800)
            return f"{weeks} week{'s' if weeks > 1 else ''} ago"
        elif seconds < 5184000:  # Approx 60 days
            return "1 month ago"
        else:
            return ">1 month ago"

    except ValueError:
        log.warning(f"Could not parse timestamp for relative time: {timestamp_str}")
        return "Invalid Time"
    except Exception as e:
        log.error(f"Error formatting relative time for {timestamp_str}: {e}")
        return "Time Error"


# New shared utility functions moved from artifact_reader.py and artifact_writer.py
def sanitize_and_validate_path(
    filename: str, base_artifact_path: str
) -> Tuple[bool, str, Optional[str], Optional[str]]: # Added relative_filename output
    """
    Sanitizes the filename and validates that it points to a path within the artifact workspace.
    Returns: (is_valid, message, full_path, relative_filename)
    """
    try:
        # Normalize the received filename and join with the base artifact folder path
        normalized_filename = os.path.normpath(filename).lstrip(os.sep)
        joined_path = os.path.join(base_artifact_path, normalized_filename)

        # Resolve the full absolute path to ensure no directory traversal attacks can occur
        normalized_joined_path = os.path.normpath(joined_path)
        if not normalized_joined_path.startswith(os.path.normpath(base_artifact_path)):
            return (
                False,
                "Path validation failed: Attempted access outside of designated workspace.",
                None,
                None,
            )

        full_path = os.path.abspath(normalized_joined_path)

    except Exception as e:
        log.error(f"Error resolving path for '{filename}': {e}")
        return False, f"Internal error resolving path: {e}", None, None

    # Ensure the final path is within the artifact folder
    normalized_base_path = os.path.normpath(base_artifact_path)
    if not (
        full_path.startswith(normalized_base_path + os.sep)
        or full_path == normalized_base_path
    ):
        log.warning(
            f"Path validation failed: Resolved path '{full_path}' is outside artifact folder '{normalized_base_path}'"
        )
        return (
            False,
            "Path validation failed: Attempted access outside of designated workspace.",
            None,
            None,
        )

    max_path_len = 255
    if len(full_path) > max_path_len:
        return False, f"Resulting file path is too long (>{max_path_len} chars).", None, None

    log.debug(f"Path validated: '{filename}' -> '{full_path}'")
    # Calculate relative path *after* validation
    relative_path = os.path.relpath(full_path, normalized_base_path)
    # Handle edge case where the path is the base itself
    relative_path = '.' if relative_path == os.curdir else relative_path

    return True, "Path validated successfully.", full_path, relative_path


def list_directory_contents(directory_path: str) -> Tuple[bool, List[str]]:
    """Safely lists contents of a directory within the artifact workspace."""
    try:
        base_artifact_path = os.path.abspath(ARTIFACT_FOLDER)
        abs_directory_path = os.path.abspath(directory_path)

        # Final safety check: ensure the directory itself is within the workspace
        if not abs_directory_path.startswith(base_artifact_path):
            log.error(
                f"Security Error: Attempted to list directory outside workspace: {abs_directory_path}"
            )
            return False, []  # Return empty list on security violation

        if not os.path.isdir(abs_directory_path):
            log.warning(
                f"Cannot list contents, path is not a directory: {abs_directory_path}"
            )
            return False, []  # Not a directory, can't list

        contents = os.listdir(abs_directory_path)
        log.info(f"Listed {len(contents)} items in directory: {abs_directory_path}")
        return True, contents
    except OSError as e:
        log.error(f"OS error listing directory '{directory_path}': {e}")
        return False, []
    except Exception as e:
        log.exception(f"Unexpected error listing directory '{directory_path}': {e}")
        return False, []