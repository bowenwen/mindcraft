# FILE: tools/artifact_writer.py
# autonomous_agent/artifact_writer.py
import os
from typing import Any, Dict, List, Tuple
import logging

from .base import Tool
from config import ARTIFACT_FOLDER
from utils import sanitize_and_validate_path, list_directory_contents

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s][Artifact Writer] %(message)s"
)
log = logging.getLogger(__name__)


class ArtifactWriterTool(Tool):
    """
    Writes text content to a specified file within the agent's secure workspace (defined by ARTIFACT_FOLDER in config).
    Creates subdirectories if they don't exist. Performs path validation to prevent writing outside the workspace.
    Checks for existing files and returns an error with a list of current files if a conflict is detected (does not overwrite).
    """

    def __init__(self):
        super().__init__(
            name="artifact_writer",
            description=(
                "Writes text content to a specified file within the agent's secure workspace. "
                "Useful for saving summaries, reports, code snippets, or final outputs. "
                "Performs strict path validation. Creates subdirectories as needed. "
                "IMPORTANT: Does NOT overwrite existing files. Returns an error with a directory listing if the file already exists. "
                "Parameters: filename (str, relative path within workspace, e.g., 'report.txt' or 'project_alpha/summary.md'), content (str, the text to write)"
            ),
        )
        # Ensure the base artifact folder exists upon tool initialization
        try:
            os.makedirs(ARTIFACT_FOLDER, exist_ok=True)
            log.info(
                f"Artifact workspace folder ensured: {os.path.abspath(ARTIFACT_FOLDER)}"
            )
        except Exception as e:
            log.error(
                f"CRITICAL: Failed to create artifact folder '{ARTIFACT_FOLDER}': {e}"
            )
            # Tool will likely fail if the folder cannot be created.

        self.base_artifact_path = os.path.abspath(ARTIFACT_FOLDER)

    def run(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Writes content to the sanitized filename within the artifact folder, creating directories and checking for existing files."""
        if not isinstance(parameters, dict):
            log.error(f"Invalid params type: {type(parameters)}")
            return {
                "error": "Invalid parameters format. Expected a JSON object (dict)."
            }

        filename = parameters.get("filename")
        content = parameters.get("content")

        if not filename or not isinstance(filename, str):
            return {
                "error": "Missing or invalid 'filename' parameter (must be a non-empty string)."
            }

        # Content can be an empty string, so check for None specifically
        if content is None:
            return {"error": "Missing 'content' parameter."}

        # Convert content to string just in case
        if not isinstance(content, str):
            try:
                content = str(content)
                log.warning(
                    f"Content parameter was not a string, converted to string (length: {len(content)})."
                )
            except Exception as e:
                return {
                    "error": f"Invalid 'content' parameter: could not convert to string. Error: {e}"
                }

        # --- Path Validation ---
        is_valid, message, full_path, filename = sanitize_and_validate_path(
            filename=filename, base_artifact_path=self.base_artifact_path
        )
        if not is_valid or not full_path:
            log.error(f"Artifact write failed validation for '{filename}': {message}")
            return {"error": f"Invalid filename: {message}"}

        log.info(f"Attempting to write artifact to: {full_path}")

        try:
            target_directory = os.path.dirname(full_path)

            # --- Create Subdirectories if they don't exist ---
            if not os.path.exists(target_directory):
                log.info(f"Creating directory: {target_directory}")
                os.makedirs(target_directory, exist_ok=True)
            elif not os.path.isdir(target_directory):
                # This should ideally not happen if sanitize works, but safety check
                log.error(
                    f"Target path's parent exists but is not a directory: {target_directory}"
                )
                return {
                    "error": f"Cannot write file, parent path '{target_directory}' is not a directory."
                }

            # --- Check if File Already Exists ---
            if os.path.exists(full_path):
                error_message = f"File already exists at the specified path: {filename}. Writing aborted to prevent overwrite."
                log.error(f"Artifact write aborted, file exists: {full_path}")
                # --- Attempt to list directory contents ---
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

            # --- Write Content (File does not exist) ---
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

            log.info(
                f"Successfully wrote {len(content)} characters to artifact: {full_path}"
            )

            return {
                "status": "success",
                "filepath": filename,  # Return the relative path file name
                "message": "Content written successfully.",
            }

        except IOError as e:
            log.error(f"IOError writing artifact '{full_path}': {e}", exc_info=True)
            return {"error": f"File system error writing artifact: {e}"}
        except OSError as e:
            log.error(
                f"OSError (e.g., creating directory) for '{full_path}': {e}",
                exc_info=True,
            )
            return {
                "error": f"Operating system error writing artifact or creating directory: {e}"
            }
        except Exception as e:
            log.exception(f"Unexpected error writing artifact '{full_path}': {e}")
            return {"error": f"Unexpected error writing artifact: {e}"}
