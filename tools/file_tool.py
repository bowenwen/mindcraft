# FILE: tools/file_tool.py
import os
import shutil
import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging

from .base import Tool

# Use SHARED paths from config
from config import (
    SHARED_ARTIFACT_FOLDER,
    CONTEXT_TRUNCATION_LIMIT,
    SHARED_ARCHIVE_FOLDER,
)
from utils import sanitize_and_validate_path, list_directory_contents

logging.basicConfig(level=logging.INFO, format="[%(levelname)s][File Tool] %(message)s")
log = logging.getLogger(__name__)


class FileTool(Tool):
    """
    Reads, writes, or lists files within the SHARED agent workspace (SHARED_ARTIFACT_FOLDER).
    Requires 'action' parameter: 'read', 'write', or 'list'.
    Organize files into logical subdirectories by project and content category.
    Writing to an existing file automatically archives the old version into the SHARED archive folder.
    """

    def __init__(self):
        super().__init__(
            name="file",
            description=(
                "Performs file operations within the SHARED agent workspace. Use subdirectories for organization, by project and content category (e.g., 'proj_xxx/summaries/report.txt', 'proj_xxx/code/script.py'). Requires 'action' parameter. "
                "Actions: "
                "'read' (requires 'filename'): Reads text content from a file (use subdirs). Returns content or error with directory listing if not found. "
                "'write' (requires 'filename', 'content'): Writes text content to a file (use subdirs). Creates directories if needed. **Automatically archives the previous version if the file exists.** Returns success status and paths. "
                "'list' (optional 'directory'): Lists files/folders in the specified directory (defaults to workspace root). Returns list or error."
            ),
        )
        # Use shared paths directly from config
        self.base_artifact_path = SHARED_ARTIFACT_FOLDER
        self.archive_base_path = SHARED_ARCHIVE_FOLDER
        log.info(
            f"FileTool initialized using SHARED workspace: {self.base_artifact_path}"
        )
        log.info(f"FileTool initialized using SHARED archive: {self.archive_base_path}")
        # Folder existence checked in config.py

    # _run_read uses self.base_artifact_path and self.archive_base_path (now shared)
    def _run_read(self, filename: str) -> Dict[str, Any]:
        log.info(f"Executing File Read: '{filename}'")
        is_valid, message, full_path, rel_filename = sanitize_and_validate_path(
            filename=filename, base_artifact_path=self.base_artifact_path
        )
        if not is_valid or not full_path:
            log.error(f"File read failed validation for '{filename}': {message}")
            return {"error": f"Invalid filename: {message}"}
        if os.path.normpath(full_path).startswith(
            os.path.normpath(self.archive_base_path)
        ):
            log.warning(f"Attempted to read from archive folder: {rel_filename}")
            return {
                "error": "Reading directly from the internal archive folder is not permitted."
            }
        log.info(f"Attempting to read artifact from: {full_path}")
        try:
            if not os.path.exists(full_path):
                error_message = f"File not found at the specified path: {rel_filename}"
                log.error(f"Artifact not found at resolved path: {full_path}")
                target_directory = os.path.dirname(full_path)
                list_success, dir_contents = list_directory_contents(target_directory)
                result_dict = {"error": error_message}
                if list_success:
                    result_dict["directory_listing"] = (
                        dir_contents if dir_contents else "(empty directory)"
                    )
                    rel_dir_path = os.path.relpath(
                        target_directory, self.base_artifact_path
                    )
                    rel_dir_path = "." if rel_dir_path == os.curdir else rel_dir_path
                    result_dict["directory_path"] = rel_dir_path
                else:
                    result_dict["directory_listing_error"] = (
                        f"Could not list contents of target directory: {target_directory}"
                    )
                return result_dict
            if not os.path.isfile(full_path):
                log.error(f"Path exists but is not a file: {full_path}")
                return {
                    "error": f"Specified path exists but is not a readable file: {rel_filename}"
                }
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            log.info(
                f"Successfully read {len(content)} characters from artifact: {full_path}"
            )
            limit = CONTEXT_TRUNCATION_LIMIT
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
                "action": "read",
                "filepath": rel_filename,
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

    # _run_write uses self.base_artifact_path and self.archive_base_path (now shared)
    def _run_write(self, filename: str, content: str) -> Dict[str, Any]:
        log.info(f"Executing File Write: '{filename}' (Content length: {len(content)})")
        if content is None:
            return {"error": "Missing 'content' parameter for 'write' action."}
        if not isinstance(content, str):
            try:
                content = str(content)
            except Exception as e:
                return {
                    "error": f"Invalid 'content' parameter: could not convert to string. Error: {e}"
                }
        is_valid, message, full_path, rel_filename = sanitize_and_validate_path(
            filename=filename, base_artifact_path=self.base_artifact_path
        )
        if not is_valid or not full_path:
            log.error(f"File write failed validation for '{filename}': {message}")
            return {"error": f"Invalid filename: {message}"}
        if os.path.normpath(full_path).startswith(
            os.path.normpath(self.archive_base_path)
        ):
            log.warning(f"Attempted to write to archive folder: {rel_filename}")
            return {
                "error": "Writing directly to the internal archive folder is not permitted."
            }
        log.info(f"Attempting to write artifact to: {full_path}")
        archived_filepath_rel = None
        archive_message = ""
        try:
            target_directory = os.path.dirname(full_path)
            if not os.path.exists(target_directory):
                log.info(f"Creating directory: {target_directory}")
                os.makedirs(target_directory, exist_ok=True)
            elif not os.path.isdir(target_directory):
                log.error(
                    f"Target path's parent exists but is not a directory: {target_directory}"
                )
                return {
                    "error": f"Cannot write file, parent path '{os.path.dirname(rel_filename)}' is not a directory."
                }

            if os.path.exists(full_path):
                if os.path.isfile(full_path):
                    log.info(f"File '{full_path}' exists. Archiving previous version.")
                    try:
                        now = datetime.datetime.now()
                        date_str = now.strftime("%Y-%m-%d")
                        time_str = now.strftime("%H%M%S_%f")[:-3]
                        original_rel_dir = os.path.dirname(rel_filename)
                        original_basename = os.path.basename(rel_filename)
                        archive_sub_dir = os.path.join(
                            self.archive_base_path,
                            date_str,
                            original_rel_dir.lstrip("." + os.sep),
                        )
                        os.makedirs(archive_sub_dir, exist_ok=True)
                        archive_filename = f"{time_str}_{original_basename}"
                        archive_full_path = os.path.join(
                            archive_sub_dir, archive_filename
                        )
                        shutil.move(full_path, archive_full_path)
                        archived_filepath_rel = os.path.relpath(
                            archive_full_path, self.base_artifact_path
                        )  # Relative to WORKSPACE
                        log.info(
                            f"Successfully archived '{rel_filename}' to '{archive_full_path}' (Relative: '{archived_filepath_rel}')"
                        )
                        archive_message = (
                            f" Previous version archived."  # Keep UI msg simpler
                        )
                    except Exception as archive_err:
                        log.exception(
                            f"Failed to archive existing file '{full_path}': {archive_err}"
                        )
                        return {
                            "error": f"Failed to archive existing file before writing. Error: {archive_err}. Write aborted."
                        }
                else:
                    log.error(
                        f"Cannot write: Path '{full_path}' exists but is not a file."
                    )
                    return {
                        "error": f"Cannot write: Path '{rel_filename}' exists but is a directory, not a file."
                    }

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            log.info(
                f"Successfully wrote {len(content)} characters to artifact: {full_path}"
            )
            return {
                "status": "success_archived" if archived_filepath_rel else "success",
                "action": "write",
                "filepath": rel_filename,
                "archived_filepath": archived_filepath_rel,
                "message": f"Content written successfully to '{rel_filename}'.{archive_message}",
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

    # _run_list uses self.base_artifact_path and self.archive_base_path (now shared)
    def _run_list(self, directory: Optional[str]) -> Dict[str, Any]:
        log.info(f"Executing File List: Directory='{directory or '(root)'}'")
        target_dir_rel = directory or "."
        is_valid, message, full_dir_path, rel_dir_path = sanitize_and_validate_path(
            filename=target_dir_rel, base_artifact_path=self.base_artifact_path
        )
        if directory is None or directory == "." or directory == "":
            full_dir_path = self.base_artifact_path
            rel_dir_path = "."
            is_valid = True
        if not is_valid or not full_dir_path:
            log.error(
                f"File list failed validation for directory '{directory}': {message}"
            )
            return {"error": f"Invalid directory path: {message}"}
        if os.path.normpath(full_dir_path).startswith(
            os.path.normpath(self.archive_base_path)
        ):
            log.warning(f"Attempted to list the archive folder: {rel_dir_path}")
            return {"error": "Listing the internal archive folder is not permitted."}
        log.info(
            f"Attempting to list directory: {full_dir_path} (Relative: {rel_dir_path})"
        )
        if not os.path.exists(full_dir_path):
            log.error(f"Directory does not exist: {full_dir_path}")
            return {"error": f"Directory not found: '{rel_dir_path}'"}
        if not os.path.isdir(full_dir_path):
            log.error(f"Path is not a directory: {full_dir_path}")
            return {"error": f"Specified path is not a directory: '{rel_dir_path}'"}
        list_success, contents = list_directory_contents(full_dir_path)
        if list_success:
            # Filter out the archive directory from the root listing
            if rel_dir_path == ".":
                archive_dir_name = os.path.basename(self.archive_base_path)
                contents = [item for item in contents if item != archive_dir_name]
            log.info(
                f"Successfully listed directory '{rel_dir_path}'. Found {len(contents)} user-visible items."
            )
            files = [
                item
                for item in contents
                if os.path.isfile(os.path.join(full_dir_path, item))
            ]
            dirs = [
                item
                for item in contents
                if os.path.isdir(os.path.join(full_dir_path, item))
            ]
            return {
                "status": "success",
                "action": "list",
                "directory_path": rel_dir_path,
                "files": files,
                "directories": dirs,
                "message": f"Successfully listed contents of '{rel_dir_path}'.",
            }
        else:
            log.error(f"Failed to list directory contents for: {full_dir_path}")
            return {"error": f"Failed to list contents of directory '{rel_dir_path}'."}

    # run method logic remains the same, calls methods that now use shared paths
    def run(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(parameters, dict):
            log.error(f"Invalid params type: {type(parameters)}")
            return {
                "error": "Invalid parameters format. Expected a JSON object (dict) with an 'action' key."
            }
        action = parameters.get("action")
        if action == "read":
            filename = parameters.get("filename")
            if not filename or not isinstance(filename, str) or not filename.strip():
                return {
                    "error": "Missing or invalid 'filename' parameter for 'read' action."
                }
            return self._run_read(filename)
        elif action == "write":
            filename = parameters.get("filename")
            content = parameters.get("content")
            if not filename or not isinstance(filename, str) or not filename.strip():
                return {
                    "error": "Missing or invalid 'filename' parameter for 'write' action."
                }
            return self._run_write(filename, content if content is not None else "")
        elif action == "list":
            directory = parameters.get("directory")
            if directory is not None and not isinstance(directory, str):
                return {
                    "error": "Invalid 'directory' parameter for 'list' action (must be a string or omitted)."
                }
            return self._run_list(directory)
        else:
            log.error(f"Invalid action specified for file tool: '{action}'")
            return {
                "error": "Invalid 'action' specified. Must be 'read', 'write', or 'list'."
            }
