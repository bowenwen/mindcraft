# autonomous_agent/llm_utils.py
import requests
import time
import datetime
import json
from typing import Optional, List, Dict, Any

# Import specific config values needed
from config import OLLAMA_BASE_URL, OLLAMA_TIMEOUT, MODEL_CONTEXT_LENGTH

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
    # log.debug(f"Sending request to Ollama ({model}). Prompt snippet: {prompt[:100]}...") # Use logger

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                api_url, json=payload, headers=headers, timeout=timeout
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            response_data = response.json()
            if "message" in response_data and "content" in response_data["message"]:
                # log.debug(f"Received response from Ollama ({model}).") # Use logger
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
            log.exception(
                f"Unexpected error calling Ollama ({model}): {e}"
            )  # Logs stack trace
            return None
    return None  # Should only be reached if retries fail


# --- NEW: Relative Time Formatting Utility ---
def format_relative_time(timestamp_str: Optional[str]) -> str:
    """Converts an ISO timestamp string into a user-friendly relative time string."""
    if not timestamp_str:
        return "Time N/A"
    try:
        # Attempt to parse, handling potential 'Z' for UTC timezone
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1] + '+00:00'
        event_time = datetime.datetime.fromisoformat(timestamp_str).replace(tzinfo=datetime.timezone.utc)
        now = datetime.datetime.now(datetime.timezone.utc)
        delta = now - event_time

        seconds = delta.total_seconds()
        if seconds < 0:
            return "in future?" # Should not happen for memories
        elif seconds < 60:
            return "<1 min ago"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} min{'s' if minutes > 1 else ''} ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif seconds < 172800: # Less than 2 days
            # Check if it was actually yesterday
            if (now.date() - event_time.date()).days == 1:
                 return "yesterday"
            else:
                 return "1 day ago" # Could be same day if time crosses midnight UTC vs local
        elif seconds < 604800: # Less than 7 days
            days = int(seconds / 86400)
            return f"{days} days ago"
        elif seconds < 1209600: # Less than 14 days
            return "last week"
        elif seconds < 2592000: # Approx 30 days
             weeks = int(seconds / 604800)
             return f"{weeks} week{'s' if weeks > 1 else ''} ago"
        elif seconds < 5184000: # Approx 60 days
             return "1 month ago"
        else:
            return ">1 month ago"

    except ValueError:
        log.warning(f"Could not parse timestamp for relative time: {timestamp_str}")
        return "Invalid Time"
    except Exception as e:
        log.error(f"Error formatting relative time for {timestamp_str}: {e}")
        return "Time Error"