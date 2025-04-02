# autonomous_agent/artifact_reader.py
import os
from typing import Any, Dict, List, Tuple
import logging

from .base import Tool
from config import ARTIFACT_FOLDER, CONTEXT_TRUNCATION_LIMIT
from utils import sanitize_and_validate_path, list_directory_contents

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s][Artifact Reader] %(message)s"
)
log = logging.getLogger(__name__)


class ArtifactReaderTool(Tool):
    """
    Reads text content from a specified file within the agent's secure workspace
    (defined by ARTIFACT_FOLDER in config). Performs path validation to prevent
    reading outside the workspace. If the file is not found, it returns an error
    along with a listing of the intended directory's contents.
    """

    def __init__(self):
        super().__init__(
            name="artifact_reader",
            description=(
                "Reads text content from a specified file within the agent's secure workspace. "
                "Useful for retrieving previously saved summaries, reports, code, or other text data. "
                "Performs strict path validation. If the file doesn't exist, provides a listing of the target directory. "
                "Returns the file content or an error. "
                "Parameters: filename (str, relative path within workspace, e.g., 'report.txt' or 'project_alpha/summary.md')"
            ),
        )
        # We can assume the writer tool or config ensures the base folder exists
        self.base_artifact_path = os.path.abspath(ARTIFACT_FOLDER)

    def run(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Reads the content from the sanitized filename within the artifact folder."""
        if not isinstance(parameters, dict):
            log.error(f"Invalid params type: {type(parameters)}")
            return {
                "error": "Invalid parameters format. Expected a JSON object (dict)."
            }

        filename = parameters.get("filename")

        if not filename or not isinstance(filename, str):
            return {
                "error": "Missing or invalid 'filename' parameter (must be a non-empty string)."
            }

        # --- Path Validation ---
        is_valid, message, full_path, filename = sanitize_and_validate_path(
            filename=filename, base_artifact_path=self.base_artifact_path
        )
        if not is_valid or not full_path:
            log.error(f"Artifact read failed validation for '{filename}': {message}")
            return {"error": f"Invalid filename: {message}"}

        log.info(f"Attempting to read artifact from: {full_path}")

        try:
            # --- Check if File Exists and is a File ---
            if not os.path.exists(full_path):
                error_message = f"File not found at the specified path: {filename}"
                log.error(f"Artifact not found at resolved path: {full_path}")
                # --- Attempt to list directory contents ---
                target_directory = os.path.dirname(full_path)
                list_success, dir_contents = list_directory_contents(target_directory)
                result_dict = {"error": error_message}
                if list_success:
                    result_dict["directory_listing"] = (
                        dir_contents if dir_contents else "(empty directory)"
                    )
                    result_dict["directory_path"] = target_directory  # Provide context
                else:
                    result_dict["directory_listing_error"] = (
                        f"Could not list contents of target directory: {target_directory}"
                    )
                return result_dict

            if not os.path.isfile(full_path):
                log.error(f"Path exists but is not a file: {full_path}")
                return {
                    "error": f"Specified path exists but is not a readable file: {filename}"
                }

            # --- Read Content ---
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            log.info(
                f"Successfully read {len(content)} characters from artifact: {full_path}"
            )

            # --- Truncate if necessary ---
            limit = CONTEXT_TRUNCATION_LIMIT  # Use same limit as browse
            truncated = False
            message = "Content read successfully."
            if len(content) > limit:
                log.info(
                    f"Content from {full_path} truncated from {len(content)} to {limit} characters."
                )
                content = content[:limit] + "\n\n... [CONTENT TRUNCATED]"
                truncated = True
                message = f"Content read successfully but was truncated to approximately {limit} characters."

            return {
                "status": "success",
                "filepath": filename,  # Return the relative path file name
                "content": content,
                "message": message,
                "truncated": truncated,
            }

        except IOError as e:
            log.error(f"IOError reading artifact '{full_path}': {e}", exc_info=True)
            return {"error": f"File system error reading artifact: {e}"}
        except UnicodeDecodeError as e:
            log.error(
                f"Encoding error reading artifact '{full_path}' as UTF-8: {e}",
                exc_info=False,
            )
            return {
                "error": f"File encoding error: Could not read file as UTF-8 text. It might be binary or have a different encoding."
            }
        except Exception as e:
            log.exception(f"Unexpected error reading artifact '{full_path}': {e}")
            return {"error": f"Unexpected error reading artifact: {e}"}
