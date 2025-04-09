# FILE: app_ui.py
# autonomous_agent/app_ui.py
import gradio as gr
import datetime
import json
import traceback
import time
import os
import sys
import io
import threading
from typing import List, Tuple, Optional, Dict, Any, Deque, Iterator
import pandas as pd
from collections import deque, defaultdict
import html
import logging

# Project Imports
import config
from utils import (
    call_ollama_api,
    format_relative_time,
    setup_doc_archive_chromadb,
)
from memory import AgentMemory, setup_chromadb
from task_manager import TaskQueue
from data_structures import Task
from agent import AutonomousAgent
import chromadb

# Logging Setup
console_log_buffer: deque = deque(maxlen=config.UI_LOG_MAX_LENGTH)
file_log_handlers: Dict[str, logging.FileHandler] = {}
main_log_handler: Optional[logging.FileHandler] = None
root_logger = logging.getLogger()
root_logger.setLevel(config.LOG_LEVEL)
log_formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s][%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# Store original streams before any potential redirection
original_stdout = sys.__stdout__  # Use underlying streams
original_stderr = sys.__stderr__

# Clear existing handlers FIRST
# This prevents handlers being added multiple times if the script is reloaded or run strangely
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)
    print(f"Removed existing logger handler: {handler.name}", file=original_stderr)


class ConsoleLogHandler(logging.Handler):
    """Handles logging ONLY to an in-memory deque for the UI."""

    # Removed the file handler dependency - root logger handles file logging.

    def __init__(self, buffer: deque):
        super().__init__()
        self.buffer = buffer
        self.name = "UIConsoleHandler"  # Give it a name for easier debugging

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            # Append to UI deque buffer efficiently
            self.buffer.append(msg + "\n")
        except Exception:
            # Use original stderr for handler errors
            self.handleError(record)

    def handleError(self, record: logging.LogRecord):
        """Handle errors during logging itself."""
        ei = sys.exc_info()
        try:
            traceback.print_exception(
                ei[0], ei[1], ei[2], None, original_stderr
            )  # Use original
            original_stderr.write("Logged from ConsoleLogHandler error handler\n")
            original_stderr.write(f"Original Record: {record}\n")
            original_stderr.flush()
        except Exception:
            pass  # Don't cause infinite recursion
        finally:
            del ei

    def flush(self):
        """No file handler to flush here."""
        pass


# Global State
agents: Dict[str, AutonomousAgent] = {}
active_agent_id: str = config.DEFAULT_AGENT_ID
completed_tasks_since_switch: Dict[str, int] = defaultdict(int)  # Use defaultdict
ui_update_lock = threading.Lock()
shared_doc_archive_collection: Optional[chromadb.Collection] = None
app_shutdown_requested = threading.Event()


# Initialization
initialization_success = True
try:
    # Setup Main Log File
    main_log_filename = (
        f"main_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    main_log_filepath = os.path.join(config.LOG_FOLDER, main_log_filename)
    os.makedirs(config.LOG_FOLDER, exist_ok=True)
    main_log_handler = logging.FileHandler(main_log_filepath, encoding="utf-8")
    main_log_handler.setFormatter(log_formatter)
    main_log_handler.setLevel(config.LOG_LEVEL)
    main_log_handler.name = "MainFileHandler"  # Give it a name
    root_logger.addHandler(main_log_handler)  # Add ONLY the file handler to root
    file_log_handlers["main"] = main_log_handler
    print(
        f"Multi-Agent App Start (Main logging to: {main_log_filepath})",
        file=original_stdout,
    )  # Use original stdout for initial messages

    # Setup Console Logging
    # This handler ONLY writes to the deque.
    ui_console_handler = ConsoleLogHandler(console_log_buffer)
    ui_console_handler.setFormatter(log_formatter)
    ui_console_handler.setLevel(config.LOG_LEVEL)
    root_logger.addHandler(ui_console_handler)  # Add the deque handler to root

    # REMOVED stdout/stderr redirection
    # Rely solely on the logging framework now.
    # sys.stdout = log_stream_redirector # REMOVED
    # sys.stderr = log_stream_redirector # REMOVED

    log = logging.getLogger("AppUI")  # Get logger *after* handlers are set
    log.info(
        f"Logging configured. UI console buffer active. Main log: {main_log_filepath}"
    )

    # Initialize Shared Document Archive
    log.info("Initializing SHARED Document Archive DB...")
    shared_doc_archive_collection = setup_doc_archive_chromadb()
    if shared_doc_archive_collection is None:
        log.error(
            "SHARED Document Archive DB setup failed. Browse-with-query will not work."
        )

    # Load Last Active Agent
    if os.path.exists(config.LAST_ACTIVE_AGENT_FILE):
        try:
            with open(config.LAST_ACTIVE_AGENT_FILE, "r") as f:
                last_id = f.read().strip()
            if last_id in config.AGENT_IDS:
                active_agent_id = last_id
                log.info(f"Loaded last active agent ID: {active_agent_id}")
            else:
                log.warning(
                    f"Saved agent ID '{last_id}' is invalid. Defaulting to {config.DEFAULT_AGENT_ID}."
                )
                active_agent_id = config.DEFAULT_AGENT_ID
        except Exception as e:
            log.error(
                f"Failed to load last active agent ID: {e}. Defaulting to {config.DEFAULT_AGENT_ID}."
            )
            active_agent_id = config.DEFAULT_AGENT_ID

    # Initialize Each Agent
    for agent_id, agent_config_data in config.AGENTS.items():
        log.info(
            f"Initializing Agent: {agent_id} ({agent_config_data.get('name', agent_id)})"
        )
        try:
            agent_instance = AutonomousAgent(
                agent_id=agent_id, agent_config=agent_config_data
            )
            # Test DB connection early if possible
            try:
                count = agent_instance.memory.collection.count()
                log.info(
                    f"Agent '{agent_id}' memory DB connection successful (Count: {count})."
                )
            except Exception as db_err:
                log.error(
                    f"Agent '{agent_id}' initial memory DB check failed: {db_err}",
                    exc_info=True,
                )
                # Decide if this is fatal or just a warning
                # raise db_err # Make it fatal for now

            agents[agent_id] = agent_instance
            log.info(f"Agent '{agent_id}' initialized successfully.")
        except Exception as e:
            log.critical(
                f"Fatal error during Agent '{agent_id}' initialization: {e}",
                exc_info=True,
            )
            initialization_success = False
            break  # Stop initializing agents if one fails

    if not initialization_success or not agents:
        log.critical("Agent initialization failed. Cannot continue.")
        initialization_success = False  # Ensure this is False if agents dict is empty

    if initialization_success:
        log.info("All required agent components initialized successfully.")
        if active_agent_id in agents:
            # Start loop paused - ensures thread exists but doesn't run immediately
            agents[active_agent_id].start_autonomous_loop()
            agents[active_agent_id].pause_autonomous_loop()
            log.info(
                f"Initial active agent '{active_agent_id}' loop created and paused."
            )
        else:
            log.error(
                f"Initial active agent ID '{active_agent_id}' not found in initialized agents!"
            )
            if agents:
                active_agent_id = list(agents.keys())[0]
                log.warning(
                    f"Setting active agent to first available: {active_agent_id}"
                )
                agents[active_agent_id].start_autonomous_loop()
                agents[active_agent_id].pause_autonomous_loop()
            else:
                initialization_success = (
                    False  # Should already be false if agents is empty
                )

except Exception as e:
    # Use original stderr for initialization errors before logging might be fully set up
    # or if logging setup itself failed.
    sys.stderr = original_stderr
    print(
        f"\n\nFATAL ERROR during App component initialization: {e}\n{traceback.format_exc()}",
        file=sys.stderr,
    )
    # Attempt to write to log file manually if handler exists
    if (
        main_log_handler
        and main_log_handler.stream
        and not main_log_handler.stream.closed
    ):
        try:
            main_log_handler.stream.write(
                f"\nFATAL INIT ERROR\n{e}\n{traceback.format_exc()}\nEND ERROR\n"
            )
            main_log_handler.stream.flush()
        except Exception as log_err:
            print(
                f"(Also failed to write fatal error to log file: {log_err})",
                file=sys.stderr,
            )

    print(
        f"Check log file in {config.BASE_OUTPUT_FOLDER} for details.\n", file=sys.stderr
    )
    initialization_success = False
    agents = {}


# Set Initial UI State Based on Active Agent
# Wrap this in a try-except just in case something goes wrong reading initial state
initial_status_text = "Error: Agent not initialized."
initial_thinking_text = "(Agent Initializing...)"
initial_tool_results_text = "(No tool results yet)"
initial_dependent_tasks_text = "Dependent Tasks: (None)"
initial_action_history_data: List[Dict] = []
initial_action_desc_text = "(Agent Initializing - Action)"
initial_plan_text = "(Agent Initializing - Plan)"
initial_task_status_text = "(Agent Initializing - Task Status)"
initial_console_log_text = "(Console Log Initializing...)"
initial_agent_name = "N/A"
initial_general_memory_data: pd.DataFrame = pd.DataFrame()
initial_task_filter_choices: List[Tuple[str, str]] = [("All Tasks", "ALL")]


# Formatting functions


def format_memories_for_display(
    memories: List[Dict[str, Any]], context_label: str
) -> str:
    if not memories:
        return f"No relevant memories found for {context_label}."
    output_parts = [f"🧠 **Recent/Relevant Memories ({context_label}):**\n"]
    for i, mem in enumerate(memories[:7]):
        content = mem.get("content", "N/A")
        meta = mem.get("metadata", {})
        mem_type = meta.get("type", "memory")
        dist_str = (
            f"{mem.get('distance', -1.0):.3f}"
            if mem.get("distance") is not None
            else "N/A"
        )
        timestamp = meta.get("timestamp")
        relative_time = format_relative_time(timestamp)
        safe_content = html.escape(content)
        summary_line = (
            f"**{i+1}. {relative_time} - Type:** {mem_type} (Dist: {dist_str})"
        )
        content_preview = safe_content[:100].replace("\n", " ") + (
            "..." if len(safe_content) > 100 else ""
        )
        details_block = f"""<details><summary>{summary_line}: {content_preview}</summary><pre style="background-color: #f0f0f0; border: 1px solid #ccc; padding: 5px; margin-top: 5px; white-space: pre-wrap; word-wrap: break-word;">{safe_content}</pre></details>"""
        output_parts.append(details_block)
    return "\n".join(output_parts)


def format_web_search_results(results_data: Optional[Dict[str, Any]]) -> str:
    actual_results = (
        results_data.get("result", {}) if isinstance(results_data, dict) else {}
    )
    if not actual_results or not isinstance(actual_results, dict):
        return "(Invalid web search results data)"
    results_list = actual_results.get("results", [])
    if not results_list:
        return "(No web search results found)"
    output = ["**🌐 Web Search Results:**\n"]
    for i, res in enumerate(results_list):
        title = res.get("title", "No Title")
        snippet = res.get("snippet", "...")
        url = res.get("url")
        res_type = res.get("type", "organic")
        output.append(f"{i+1}. **{title}** ({res_type})")
        if url:  # Corrected indentation
            output.append(f"   [{url}]({url})")
        output.append(f"   > {snippet}\n")  # Ensure snippet is always appended
    return "\n".join(output)


def format_memory_search_results(results_data: Optional[Dict[str, Any]]) -> str:
    actual_results = (
        results_data.get("result", {}) if isinstance(results_data, dict) else {}
    )
    if not actual_results or not isinstance(actual_results, dict):
        return "(Invalid memory search results data)"
    results_list = actual_results.get("retrieved_memories", [])
    if not results_list:
        return "(No relevant memories found in search)"
    output = ["**🧠 Memory Search Results:**\n"]
    for res in results_list:
        rank = res.get("rank", "-")
        rel_time = res.get("relative_time", "N/A")
        mem_type = res.get("type", "N/A")
        dist = res.get("distance", "N/A")
        snippet = res.get("content_snippet", "...")
        output.append(f"**Rank {rank} ({rel_time})** - Type: {mem_type}, Dist: {dist}")
        output.append(f"   > {snippet}\n")
    return "\n".join(output)


def format_web_browse_results(results_data: Optional[Dict[str, Any]]) -> str:
    actual_results = (
        results_data.get("result", {}) if isinstance(results_data, dict) else {}
    )
    if not actual_results or not isinstance(actual_results, dict):
        return "(Invalid web browse results data)"
    if actual_results.get("query_mode") is True:
        log.warning(
            "format_web_browse_results called with query_mode=True data. Routing to query formatter."
        )
        return format_web_browse_query_results(results_data)
    content = actual_results.get("content")
    url = actual_results.get("url", "N/A")
    source = actual_results.get("content_source", "N/A")
    message = actual_results.get("message", "")
    truncated = actual_results.get("truncated", False)
    output = [
        f"**📄 Web Browse Result**\n\n**URL:** {url}\n\n**Source Type:** {source}\n"
    ]
    if truncated:
        output.append("**Note:** Content was truncated.\n")
    if message:
        output.append(f"{message}\n")
    output.append("\n")  # Moved outside message check
    output.append(
        "```text\n" + (content if content else "(No content extracted)") + "\n```"
    )
    return "\n".join(output)


def format_web_browse_query_results(results_data: Optional[Dict[str, Any]]) -> str:
    actual_results = (
        results_data.get("result", {}) if isinstance(results_data, dict) else {}
    )
    if (
        not actual_results
        or not isinstance(actual_results, dict)
        or actual_results.get("query_mode") is not True
    ):
        return "(Invalid focused web browse results data)"
    url = actual_results.get("url", "N/A")
    query = actual_results.get("query", "N/A")
    snippets = actual_results.get("retrieved_snippets", [])
    message = actual_results.get("message", "")
    output = [
        f"**📄 Focused Web Browse Results**\n\n**URL:** {url}\n\n**Query:** `{query}`\n"
    ]
    if message:
        output.append(f"{message}\n")
    if not snippets:
        output.append(
            "\n,\n_(No relevant snippets found for the query in the archived content.)_"
        )
        return "\n".join(output)
    output.append("\n**Retrieved Snippets:**\n")
    for i, snip in enumerate(snippets):
        content = snip.get("content", "N/A")
        dist = snip.get("distance", "N/A")
        idx = snip.get("chunk_index", "N/A")
        safe_content = html.escape(content)
        summary_line = f"**{i+1}. Snippet (Index: {idx}, Distance: {dist})**"
        content_preview = safe_content[:120].replace("\n", " ") + (
            "..." if len(safe_content) > 120 else ""
        )
        details_block = f"""<details><summary>{summary_line}: {content_preview}</summary><pre style="background-color: #f0f0f0; border: 1px solid #ccc; padding: 5px; margin-top: 5px; white-space: pre-wrap; word-wrap: break-word;">{safe_content}</pre></details>"""
        output.append(details_block)
    return "\n".join(output)


def format_other_tool_results(results_data: Optional[Dict[str, Any]]) -> str:
    if not results_data or not isinstance(results_data, dict):
        return "(Invalid tool results data)"
    tool_name = results_data.get("tool_name", "Unknown Tool")
    action = results_data.get("action", "Unknown Action")
    status = results_data.get("status", "Unknown Status")
    error = results_data.get("error")
    actual_results = results_data.get("result", {})
    if status == "failed" and error:
        return f"**🛠️ Tool Execution Failed**\n\n**Tool:** {tool_name}\n**Action:** {action}\n**Status:** {status}\n\n**Error:** {error}"
    result_error = (
        actual_results.get("error") if isinstance(actual_results, dict) else None
    )
    if (
        tool_name == "status"
        and action == "status_report"
        and isinstance(actual_results, dict)
        and "report_content" in actual_results
    ):
        report_content = actual_results.get(
            "report_content", "(Status report content missing)"
        )
        message = actual_results.get("message", "")
        return f"**📊 Status Report**\n\n**Status:** {status}\n\n{message}\n\n\n```markdown\n{report_content}\n```"
    output = [f"**🛠️ Tool Result: {tool_name} ({action})**\n\n**Status:** {status}\n"]
    if result_error:
        output.append(f"\n**Error:** {result_error}\n")
    content_to_display = {}
    message = None
    if isinstance(actual_results, dict):
        message = actual_results.get("message")
        if message:
            output.append(f"\n{message}\n")
        content_to_display = {
            k: v
            for k, v in actual_results.items()
            if k not in ["status", "action", "error", "message", "report_content"]
        }
    if content_to_display:
        try:
            result_json = json.dumps(content_to_display, indent=2, ensure_ascii=False)
            output.append("\n**Details:**\n```json\n" + result_json + "\n```")
        except Exception:
            output.append("\n**Details:**\n" + str(content_to_display))
    elif (
        not result_error and not message and not isinstance(actual_results, dict)
    ):  # Avoid adding raw result if message or error was already there
        output.append("\n**Result:**\n```text\n" + str(actual_results) + "\n```")
    elif (
        not result_error
        and not message
        and not content_to_display
        and not actual_results
    ):
        output.append("\n_(Tool returned empty result)_")

    return "\n".join(output)


def format_display_df(data, columns):
    if not isinstance(data, list) or (
        data and not all(isinstance(item, dict) for item in data)
    ):
        log.warning(
            f"Invalid data type passed to format_display_df: {type(data)}. Expected list of dicts."
        )
        return pd.DataFrame(columns=columns)
    if not data:
        return pd.DataFrame(columns=columns)
    try:
        df = pd.DataFrame(data)
        # Ensure all expected columns exist, fill with NA if missing
        for col in columns:
            if col not in df.columns:
                df[col] = pd.NA  # Use pandas NA
        # Reindex to ensure correct column order and fill missing values
        df = df.reindex(columns=columns, fill_value=pd.NA)
        # Convert Timestamp if it exists and needs it
        if "Timestamp" in df.columns:
            # Convert to datetime, coercing errors, make timezone-aware (UTC) then remove tz info for display consistency
            df["Timestamp_dt"] = pd.to_datetime(
                df["Timestamp"], errors="coerce", utc=True
            )
            # Format for display, handling NaT
            df["Timestamp"] = (
                df["Timestamp_dt"].dt.strftime("%Y-%m-%d %H:%M:%S UTC").fillna("N/A")
            )
            df = df.drop(columns=["Timestamp_dt"])  # Remove temporary column

        return df
    except Exception as df_err:
        log.error(f"Error creating DataFrame: {df_err}. Data sample: {str(data)[:200]}")
        return pd.DataFrame(columns=columns)


try:
    if initialization_success and active_agent_id in agents:
        initial_state = agents[active_agent_id].get_ui_update_state()
        task_status = initial_state.get("status", "paused")
        initial_agent_name = initial_state.get("agent_name", active_agent_id)
        initial_status_text = f"Status: {task_status}"
        initial_thinking_text = initial_state.get("thinking", "(Agent Paused/Idle)")
        initial_action_history_data = initial_state.get("action_history", [])
        initial_action_desc_text = initial_state.get("current_action_desc", "N/A")
        initial_plan_text = initial_state.get("current_plan", "(None)")
        initial_task_status_text = f"ID: {initial_state.get('current_task_id', 'None')} ({task_status})\n\nDescription: {initial_state.get('current_task_desc', 'N/A')}"
        initial_console_log_text = "".join(console_log_buffer) or "(Console Log Empty)"
        # Initial general memory formatting
        general_memories_list = initial_state.get("recent_general_memories", [])
        memory_cols = ["Relative Time", "Timestamp", "Type", "Content Snippet", "ID"]
        initial_general_memory_data = format_display_df(
            general_memories_list, memory_cols
        )

        # Initial Task filter choices
        all_tasks_for_filter = (
            agents[active_agent_id]
            .get_agent_dashboard_state()
            .get("all_tasks_for_filter", [])
        )
        initial_task_filter_choices.extend(
            [
                (
                    f"{t['id'][:8]}... ({t['status']}) - {t['description'][:30]}...",
                    t["id"],
                )
                for t in all_tasks_for_filter
            ]
        )

        log.info(f"Calculated initial UI state for agent '{active_agent_id}'.")
    elif not initialization_success:
        # If init failed, ensure console log reflects it
        initial_console_log_text = (
            "FATAL: Agent initialization failed. Check startup logs.\n"
            + "".join(console_log_buffer)
        )
        initial_general_memory_data = pd.DataFrame([{"Error": "Initialization Failed"}])
    # else: agent ID not found but init was success? Should not happen based on logic above.

except Exception as initial_state_err:
    log.exception(
        f"CRITICAL: Error getting initial UI state for agent '{active_agent_id}'."
    )
    initial_status_text = f"Error loading initial state for {active_agent_id}."
    initial_general_memory_data = pd.DataFrame(
        [{"Error": f"Error loading state: {initial_state_err}"}]
    )
    initial_console_log_text = (
        f"CRITICAL Error during initial UI state calculation:\n{traceback.format_exc()}\n"
        + "".join(console_log_buffer)
    )
    # Potentially halt here if initial state is crucial and fails?
    # For now, we let it proceed to define the UI, which might show errors.


# Agent Control Functions (Unchanged: save_last_active_agent, switch_active_agent)
def save_last_active_agent():
    """Saves the current active_agent_id to a file."""
    if not initialization_success:
        return  # Don't save if init failed
    try:
        with open(config.LAST_ACTIVE_AGENT_FILE, "w") as f:
            f.write(active_agent_id)
        log.debug(f"Saved last active agent ID: {active_agent_id}")
    except Exception as e:
        log.error(f"Error saving last active agent ID '{active_agent_id}': {e}")


def switch_active_agent(new_agent_id: str) -> Tuple[str, str]:
    """Handles pausing the current agent and activating the new one."""
    global active_agent_id, completed_tasks_since_switch
    if not initialization_success:
        return "Error: Agents not initialized.", "Error: Cannot switch agents."

    with ui_update_lock:
        current_agent_id = active_agent_id
        if new_agent_id == current_agent_id:
            log.info(f"Agent {new_agent_id} is already active.")
            agent = agents.get(current_agent_id)
            status = (
                agent.get_ui_update_state().get("status", "unknown")
                if agent
                else "unknown"
            )
            name = agent.agent_name if agent else "N/A"
            return (
                f"Agent '{name}' Status: {status}",
                f"Agent {new_agent_id} already active.",
            )

        if new_agent_id not in agents:
            log.error(f"Cannot switch: Agent ID '{new_agent_id}' not found.")
            agent = agents.get(current_agent_id)
            status = (
                agent.get_ui_update_state().get("status", "unknown")
                if agent
                else "unknown"
            )
            name = agent.agent_name if agent else "N/A"
            return (
                f"Agent '{name}' Status: {status}",
                f"Error: Agent {new_agent_id} not found.",
            )

        log.info(
            f"Switching active agent from '{current_agent_id}' to '{new_agent_id}'..."
        )

        current_agent = agents.get(current_agent_id)
        if current_agent:  # Check if current agent exists before pausing
            is_currently_running = (
                current_agent._is_running.is_set()
            )  # Check before pausing
            log.info(
                f"Pausing current agent '{current_agent_id}' (was running: {is_currently_running})..."
            )
            current_agent.pause_autonomous_loop()
            # Add a small delay ONLY if it was running to allow loop cycle to potentially finish gracefully
            if is_currently_running:
                time.sleep(0.5)  # Increased delay slightly

        # Update the global active agent ID
        active_agent_id = new_agent_id
        log.info(f"Active agent is now '{active_agent_id}'.")

        # Reset the task completion counter for the *newly activated* agent
        # We want it to run AUTO_SWITCH_TASK_COUNT tasks *from now*
        completed_tasks_since_switch[active_agent_id] = 0
        log.info(
            f"Reset task completion counter for new active agent '{active_agent_id}'."
        )

        # Ensure the new agent's loop is created and paused (it might already exist)
        new_agent = agents[active_agent_id]
        log.info(f"Ensuring agent '{active_agent_id}' loop is created and paused...")
        new_agent.start_autonomous_loop()  # Ensures thread exists if it doesn't
        new_agent.pause_autonomous_loop()  # Ensure it ends up paused

        # Save the new active agent ID
        save_last_active_agent()

        new_state = new_agent.get_ui_update_state()
        status_msg = f"Agent '{new_agent.agent_name}' Status: {new_state.get('status', 'paused')}"
        feedback_msg = f"Switched to Agent: {new_agent.agent_name} ({active_agent_id})"

    # Force UI refresh outside the lock via return values
    log.info(f"Agent switch complete. New status: {status_msg}")
    return status_msg, feedback_msg


# Monitor Tab functions (Unchanged: start_agent_processing, pause_agent_processing, suggest_task_change)
def start_agent_processing():
    if not initialization_success or active_agent_id not in agents:
        log.error("UI start failed: Active agent not initialized or init failed.")
        return "ERROR: Agent not initialized."

    agent = agents[active_agent_id]
    log.info(f"UI: Start/Resume Agent '{active_agent_id}'")
    agent.start_autonomous_loop()
    # Brief pause to allow the agent's status update to potentially register
    time.sleep(config.UI_UPDATE_INTERVAL / 2.0)  # Wait half update interval
    state = agent.get_ui_update_state()
    return f"Status: {state.get('status', 'running')}"


def pause_agent_processing():
    if not initialization_success or active_agent_id not in agents:
        log.error("UI pause failed: Active agent not initialized or init failed.")
        return "ERROR: Agent not initialized."

    agent = agents[active_agent_id]
    log.info(f"UI: Pause Agent '{active_agent_id}'")
    agent.pause_autonomous_loop()
    # Wait slightly longer to ensure the loop sees the pause signal
    time.sleep(config.UI_UPDATE_INTERVAL)
    state = agent.get_ui_update_state()
    return f"Status: {state.get('status', 'paused')}"


def suggest_task_change():
    feedback = "Suggestion ignored: Agent not initialized."
    if initialization_success and active_agent_id in agents:
        agent = agents[active_agent_id]
        log.info(
            f"UI: User clicked 'Suggest Task Change' for agent '{active_agent_id}'"
        )
        result = agent.handle_user_suggestion_move_on()
        feedback = result
    else:
        log.error(
            f"UI suggest_task_change failed: Active agent '{active_agent_id}' not initialized or init failed."
        )
    return feedback


# update_monitor_ui <<< MODIFIED >>>
def update_monitor_ui() -> Tuple[
    str,  # current_task_display
    str,  # current_action_display
    str,  # current_plan_display
    str,  # dependent_tasks_display
    str,  # thinking_display
    str,  # console_log_output
    str,  # memory_display (task-specific)
    pd.DataFrame,  # general_memory_df
    str,  # tool_results_display
    str,  # status_bar_text
    str,  # agent_selection_value
    List[Dict],  # current_action_history_data (for history tab)
]:
    """Fetches the latest active agent UI state, console log, handles auto-switch, and returns UI updates."""
    global active_agent_id, completed_tasks_since_switch
    current_task_display = "(Error)"
    current_action_display = "(Error)"
    current_plan_display = "(Error)"
    dependent_tasks_display = "Dependent Tasks: (Error)"
    thinking_display = "(Error)"
    console_log_output = "".join(list(console_log_buffer)) or "(Console Log Empty)"
    memory_display = "(Error)"
    general_memory_df = pd.DataFrame()  # Initialize empty
    tool_results_display = "(Error)"
    status_bar_text = "Error"
    current_action_history_data = []
    agent_selection_value = active_agent_id
    memory_cols = [
        "Relative Time",
        "Timestamp",
        "Type",
        "Content Snippet",
        "ID",
    ]  # Define cols for general mem

    if not initialization_success or not agents or active_agent_id not in agents:
        status_bar_text = "Error: Active agent not initialized or init failed."
        current_task_display = "Not Initialized"
        current_action_display = "Not Initialized"
        console_log_output = "Agent Not Initialized / Init Failed\n" + "".join(
            list(console_log_buffer)
        )
        general_memory_df = format_display_df([], memory_cols)  # Ensure empty df

        return (
            current_task_display,
            current_action_display,
            current_plan_display,
            dependent_tasks_display,
            thinking_display,
            console_log_output,
            memory_display,
            general_memory_df,  # Return empty df
            tool_results_display,
            status_bar_text,
            agent_selection_value,
            current_action_history_data,
        )

    try:
        with ui_update_lock:
            agent = agents[active_agent_id]
            ui_state = agent.get_ui_update_state()

            # Extract state for UI elements
            current_task_id = ui_state.get("current_task_id")
            task_status = ui_state.get("status", "unknown")
            current_task_desc = ui_state.get("current_task_desc", "N/A")
            agent_name = ui_state.get("agent_name", active_agent_id)
            current_task_display = f"ID: {current_task_id or 'None'} ({task_status})\n\nDescription: {current_task_desc}"
            current_action_display = f"{ui_state.get('current_action_desc', 'N/A')}"
            current_plan_display = ui_state.get("current_plan", "(No plan active)")
            deps = ui_state.get("dependent_tasks", [])
            if deps:
                dep_lines = [
                    f"- {d['id'][:8]}...: {d['description'][:60]}..." for d in deps
                ]
                dependent_tasks_display = "Dependent Tasks:\n" + "\n".join(dep_lines)
            else:
                dependent_tasks_display = "Dependent Tasks: (None)"
            thinking_display = ui_state.get("thinking", "(No thinking recorded)")

            # Format task-specific memories
            recent_memories = ui_state.get("recent_memories", [])
            memory_display = format_memories_for_display(
                recent_memories, context_label=f"Monitor ({agent_name})"
            )

            # Format general memories
            recent_general_memories = ui_state.get("recent_general_memories", [])
            # Add relative time if missing (should be done by agent ideally)
            for mem in recent_general_memories:
                if "Relative Time" not in mem:
                    mem["Relative Time"] = format_relative_time(mem.get("Timestamp"))
            general_memory_df = format_display_df(recent_general_memories, memory_cols)
            if not general_memory_df.empty and "Timestamp" in general_memory_df.columns:
                # Sort descending by original timestamp string
                general_memory_df = general_memory_df.sort_values(
                    by="Timestamp",
                    ascending=False,
                    na_position="last",
                    key=lambda col: pd.to_datetime(col, errors="coerce"),
                )

            last_action = ui_state.get("last_action_type")
            last_results = ui_state.get("last_tool_results")

            # Format tool results
            if last_action == "use_tool" and last_results:
                tool_name = last_results.get("tool_name")
                action_name = last_results.get("action")
                tool_status = last_results.get("status")
                if tool_status == "failed":
                    tool_results_display = format_other_tool_results(last_results)
                elif tool_name == "web" and action_name == "search":
                    tool_results_display = format_web_search_results(last_results)
                elif tool_name == "web" and action_name == "browse":
                    result_payload = last_results.get("result", {})
                    if result_payload.get("query_mode") is True:
                        tool_results_display = format_web_browse_query_results(
                            last_results
                        )
                    else:
                        tool_results_display = format_web_browse_results(last_results)
                elif tool_name == "memory" and action_name == "search":
                    tool_results_display = format_memory_search_results(last_results)
                elif tool_name in ["status", "file"] or (
                    tool_name == "memory" and action_name == "write"
                ):
                    tool_results_display = format_other_tool_results(last_results)
                else:
                    log.warning(
                        f"No specific formatter for tool '{tool_name}' action '{action_name}'. Using general formatter."
                    )
                    tool_results_display = format_other_tool_results(last_results)
            elif last_action == "final_answer":
                tool_results_display = "(Provided Final Answer - See History Tab)"
            elif last_action == "error":
                tool_results_display = (
                    "(Agent action resulted in error - see Console Log or History Tab)"
                )
            else:
                # Display last known web content if no recent tool action
                last_web = ui_state.get("last_web_content", "(No recent web browse)")
                if (
                    last_web
                    and last_web != "(None)"
                    and last_web != "(No recent web browse)"
                ):
                    tool_results_display = f"**📄 Last Browsed Content (Query Mode Off):**\n\n```text\n{last_web[:1000]}...\n```"
                else:
                    tool_results_display = "(No recent tool execution or web content)"

            status_bar_text = f"Status: {task_status}"
            current_action_history_data = ui_state.get("action_history", [])

            # Auto-Switch Logic (remains the same)
            agent_is_idle_or_paused = task_status in [
                "idle",
                "paused",
                "completed",
                "failed",
            ]
            if (
                task_status in ["completed", "failed"]
                and current_task_id is not None
                and agent_is_idle_or_paused
            ):
                log.info(
                    f"Agent '{active_agent_id}' finished task {current_task_id} with status {task_status}. Incrementing switch counter."
                )
                completed_tasks_since_switch[active_agent_id] += 1
                log.info(
                    f"Agent '{active_agent_id}' completed tasks since last switch: {completed_tasks_since_switch[active_agent_id]}"
                )

                if (
                    len(agents) > 1
                    and completed_tasks_since_switch[active_agent_id]
                    >= config.AUTO_SWITCH_TASK_COUNT
                ):
                    log.info(
                        f"Agent '{active_agent_id}' reached task limit ({config.AUTO_SWITCH_TASK_COUNT}). Triggering auto-switch."
                    )
                    current_index = config.AGENT_IDS.index(active_agent_id)
                    next_index = (current_index + 1) % len(config.AGENT_IDS)
                    next_agent_id = config.AGENT_IDS[next_index]

                    log.warning(
                        f"AUTO-SWITCHING from {active_agent_id} to {next_agent_id}"
                    )

                    _new_status_msg, _feedback_msg = switch_active_agent(next_agent_id)
                    status_bar_text = _new_status_msg
                    agent_selection_value = next_agent_id

                    # Get state of the *new* agent
                    agent = agents[active_agent_id]
                    ui_state = agent.get_ui_update_state()
                    # Re-extract state variables for the new agent
                    current_task_id = ui_state.get("current_task_id")
                    task_status = ui_state.get("status", "unknown")
                    current_task_desc = ui_state.get("current_task_desc", "N/A")
                    agent_name = ui_state.get("agent_name", active_agent_id)
                    current_task_display = f"ID: {current_task_id or 'None'} ({task_status})\n\nDescription: {current_task_desc}"
                    current_action_display = (
                        f"{ui_state.get('current_action_desc', 'N/A')}"
                    )
                    current_plan_display = ui_state.get(
                        "current_plan", "(No plan active)"
                    )
                    deps = ui_state.get("dependent_tasks", [])
                    dependent_tasks_display = (
                        "Dependent Tasks:\n"
                        + "\n".join(
                            [
                                f"- {d['id'][:8]}...: {d['description'][:60]}..."
                                for d in deps
                            ]
                        )
                        if deps
                        else "Dependent Tasks: (None)"
                    )
                    thinking_display = ui_state.get(
                        "thinking", "(No thinking recorded)"
                    )
                    recent_memories = ui_state.get("recent_memories", [])
                    memory_display = format_memories_for_display(
                        recent_memories, context_label=f"Monitor ({agent_name})"
                    )
                    # Refresh general memories for new agent
                    recent_general_memories = ui_state.get(
                        "recent_general_memories", []
                    )
                    # Add relative time if missing
                    for mem in recent_general_memories:
                        if "Relative Time" not in mem:
                            mem["Relative Time"] = format_relative_time(
                                mem.get("Timestamp")
                            )
                    general_memory_df = format_display_df(
                        recent_general_memories, memory_cols
                    )
                    if (
                        not general_memory_df.empty
                        and "Timestamp" in general_memory_df.columns
                    ):
                        general_memory_df = general_memory_df.sort_values(
                            by="Timestamp",
                            ascending=False,
                            na_position="last",
                            key=lambda col: pd.to_datetime(col, errors="coerce"),
                        )

                    tool_results_display = "(No recent tool execution for this agent)"
                    current_action_history_data = ui_state.get("action_history", [])
                    status_bar_text = f"Status: {task_status}"

    except Exception as e:
        log.exception("Error in update_monitor_ui")
        error_msg = f"ERROR updating UI:\n{traceback.format_exc()}"
        console_log_output = error_msg + "\n" + "".join(list(console_log_buffer))
        status_bar_text = "Error updating UI"
        current_task_display = "Error"
        current_action_display = "Error"
        current_plan_display = "Error"
        dependent_tasks_display = "Dependent Tasks: (Error)"
        thinking_display = "Error"
        memory_display = "Error"
        general_memory_df = format_display_df(
            [{"Error": str(e)}], memory_cols
        )  # Show error in df
        tool_results_display = "Error"
        current_action_history_data = [{"error": f"Update failed: {e}"}]

    return (
        current_task_display,
        current_action_display,
        current_plan_display,
        dependent_tasks_display,
        thinking_display,
        console_log_output,
        memory_display,
        general_memory_df,  # Return updated general memories
        tool_results_display,
        status_bar_text,
        agent_selection_value,
        current_action_history_data,
    )


# End MODIFIED update_monitor_ui


# Functions for Action History Navigation <<< MODIFIED >>>
def format_action_details_for_display(
    action_data: Optional[Dict],
) -> Tuple[str, str, str, str, str, str]:  # Added llm_prompt
    if not action_data or not isinstance(action_data, dict):
        return (
            "(No Action Data)",
            "(N/A)",
            "(N/A)",
            "(N/A)",  # llm_prompt
            "(N/A)",
            "(N/A)",
        )
    cycle_num = action_data.get("action_cycle", "N/A")
    task_id = action_data.get("task_id", "N/A")
    timestamp = action_data.get("timestamp", "N/A")
    rel_time = format_relative_time(timestamp)
    title = f"**Cycle {cycle_num}** (Task: {task_id[:8]}... | Attempt: {action_data.get('task_attempt','?')}) - {rel_time}"
    objective = action_data.get("action_objective", "(Objective not recorded)")
    thinking = action_data.get("thinking", "(No thinking recorded)")
    llm_prompt = action_data.get(
        "llm_prompt", "(LLM prompt not recorded)"
    )  # <<< GET PROMPT >>>
    log_snippet = action_data.get("log_snippet", "(No log snippet)")
    action_type = action_data.get("action_type")
    action_params = action_data.get("action_params")
    result_status = action_data.get("result_status")
    result_summary = action_data.get("result_summary", "(No result summary)")

    results_display = f"**Action Type:** {action_type}\n\n"
    if action_type == "use_tool":
        results_display += f"**Tool:** {action_data.get('tool', '?')}\n"
        if action_params:
            try:
                results_display += f"**Params:**\n```json\n{json.dumps(action_params, indent=2)}\n```\n"
            except:
                results_display += f"**Params:** {action_params}\n\n"
        results_display += f"**Result Status:** `{result_status}`\n\n"
        results_display += f"**Result Summary:**\n```markdown\n{result_summary}\n```\n"
    elif action_type == "final_answer":
        results_display += f"**Result Status:** `{result_status}`\n\n"
        results_display += f"{result_summary}\n"  # Result summary contains the formatted answer/reflections
    elif action_type == "error":
        results_display += f"**Result Status:** `{result_status}`\n\n"
        results_display += f"**Error Details:**\n```text\n{result_summary}\n```\n"  # Result summary contains error message
    else:  # Internal / Unknown
        results_display += f"**Result Status:** `{result_status}`\n\n"
        results_display += f"{result_summary}\n"

    return title, objective, thinking, llm_prompt, log_snippet, results_display


def filter_and_get_action_details(
    selected_task_id: str,
    current_index_str: str,
    delta: int,
    full_history_data: List[Dict],
) -> Tuple[str, str, str, str, str, str, str, List[Dict]]:
    """Filters history by task_id and navigates steps."""
    try:
        current_index = int(current_index_str)
    except (ValueError, TypeError):
        current_index = 0

    if not full_history_data:
        title, objective, thinking, prompt, log_s, result_s = (
            format_action_details_for_display(None)
        )
        return str(0), title, objective, thinking, prompt, log_s, result_s, []

    # Filter history based on selected task ID
    if selected_task_id == "ALL" or not selected_task_id:
        filtered_history = full_history_data
    else:
        filtered_history = [
            item
            for item in full_history_data
            if item.get("task_id") == selected_task_id
        ]

    if not filtered_history:
        title, objective, thinking, prompt, log_s, result_s = (
            format_action_details_for_display(None)
        )
        return str(0), title, objective, thinking, prompt, log_s, result_s, []

    # Navigate within the filtered history
    new_index = current_index + delta
    new_index = max(0, min(len(filtered_history) - 1, new_index))

    selected_data = filtered_history[new_index]
    title, objective, thinking, prompt, log_s, result_s = (
        format_action_details_for_display(selected_data)
    )

    return (
        str(new_index),
        title,
        objective,
        thinking,
        prompt,
        log_s,
        result_s,
        filtered_history,
    )


def update_history_task_filter_choices() -> gr.Dropdown:
    """Updates the choices for the task filter dropdown in the history tab."""
    if not initialization_success or not agents or active_agent_id not in agents:
        return gr.Dropdown(
            choices=[("Error loading tasks", None)], value=None, interactive=False
        )

    try:
        agent = agents[active_agent_id]
        # Fetch the latest task list directly
        state_data = agent.get_agent_dashboard_state()
        all_tasks = state_data.get("all_tasks_for_filter", [])
        log.info(
            f"Updating history filter with {len(all_tasks)} tasks for agent {active_agent_id}"
        )
        task_choices = [("All Tasks", "ALL")] + [
            (f"{t['id'][:8]}... ({t['status']}) - {t['description'][:30]}...", t["id"])
            for t in all_tasks
        ]
        # Preserve current selection if possible, otherwise default to ALL
        # current_value = # How to get current value easily? Pass it in?
        # For now, just reset to ALL on refresh.
        return gr.Dropdown(
            choices=task_choices, value="ALL", label="Filter by Task", interactive=True
        )
    except Exception as e:
        log.exception(
            f"Error updating history task filter choices for agent {active_agent_id}"
        )
        return gr.Dropdown(
            choices=[("Error loading tasks", None)],
            value=None,
            label="Filter by Task (Error)",
            interactive=False,
        )


# End Action History Functions


# Chat Tab functions (Unchanged: _should_generate_task, prioritize_generated_task, create_priority_task, chat_response)
def _should_generate_task(user_msg: str, assistant_response: str) -> bool:
    if not agents or active_agent_id not in agents:
        return False
    agent = agents[active_agent_id]
    log.info(f"[{agent.log_prefix}] Evaluating if task gen warranted...")
    eval_model = os.environ.get("OLLAMA_EVAL_MODEL", config.OLLAMA_CHAT_MODEL)
    prompt = f"""Analyze chat. Does user OR assistant response imply need for complex background task (e.g., research, multi-step analysis)? Answer ONLY "YES" or "NO".\n\nUser: {user_msg}\nAssistant: {assistant_response}"""
    try:
        log.info(f"[{agent.log_prefix}] Calling task evaluation model {eval_model}...")
        response = call_ollama_api(
            prompt=prompt, model=eval_model, base_url=config.OLLAMA_BASE_URL, timeout=30
        )
        log.info(f"[{agent.log_prefix}] Task evaluation response: '{response}'")
        decision = response and "yes" in response.strip().lower()
        log.info(
            f"[{agent.log_prefix}] Task gen decision: {'YES' if decision else 'NO'}"
        )
        return decision
    except Exception as e:
        log.error(f"[{agent.log_prefix}] Error during task generation evaluation: {e}")
        return False


def prioritize_generated_task(last_generated_task_id: Optional[str]):
    feedback = "Prioritization failed: Agent not initialized."
    if initialization_success and agents and active_agent_id in agents:
        agent = agents[active_agent_id]
        if last_generated_task_id:
            log.info(
                f"UI: User requested prioritization for task {last_generated_task_id} on agent {active_agent_id}"
            )
            success = agent.task_queue.set_priority(last_generated_task_id, 9)
            if success:
                task = agent.task_queue.get_task(last_generated_task_id)
                feedback = (
                    f"✅ Task '{task.description[:30]}...' priority set to 9 for Agent {agent.agent_name}."
                    if task
                    else f"✅ Task {last_generated_task_id} priority set to 9 for Agent {agent.agent_name} (task details not found)."
                )
            else:
                feedback = f"⚠️ Failed to set priority for task {last_generated_task_id} (may already be completed/failed or ID invalid)."
        else:
            feedback = "⚠️ Prioritization failed: No task ID provided (was a task generated this turn?)."
    return feedback


def create_priority_task(message_text: str):
    feedback = "Create Task failed: Agent not initialized."
    if initialization_success and agents and active_agent_id in agents:
        agent = agents[active_agent_id]
        if message_text and message_text.strip():
            log.info(
                f"UI: Creating priority task for Agent {active_agent_id}: '{message_text[:50]}...'"
            )
            new_task = Task(description=message_text.strip(), priority=9)
            task_id = agent.task_queue.add_task(new_task)
            if task_id:
                feedback = f"✅ Priority Task Created for Agent {agent.agent_name} (ID: {task_id[:8]}...): '{new_task.description[:60]}...'"
                log.info(f"Priority task {task_id} added for agent {active_agent_id}.")
            else:
                feedback = "❌ Error adding task (likely duplicate description or internal error)."
                log.error(
                    f"Failed to add priority task via UI for agent {active_agent_id}."
                )
        else:
            feedback = "⚠️ Create Task failed: No message provided in the input box."
    else:
        log.error("UI create_priority_task failed: Active agent not initialized.")
    return feedback


# Refactored Chat Response using yield (Unchanged)
def chat_response(
    message: str, history: List[Dict[str, str]]
) -> Iterator[Tuple[List[Dict[str, str]], str, str, str, Optional[str]]]:
    """
    Handles chat interaction, yielding the main response first, then performing
    secondary actions like memory logging and task generation.
    """
    memory_display_text = "Processing..."
    task_display_text = "(No task generated this turn)"
    last_gen_id = None
    assistant_response_text = ""  # Initialize

    if not message:
        yield history, "No input provided.", task_display_text, "", last_gen_id
        return

    if not initialization_success or not agents or active_agent_id not in agents:
        history.append({"role": "user", "content": message})
        history.append(
            {"role": "assistant", "content": "**ERROR:** Active agent backend failed."}
        )
        yield history, "Error: Backend failed.", task_display_text, "", last_gen_id
        return

    agent = agents[active_agent_id]
    log.info(f"[{agent.log_prefix}] User message: '{message}'")
    history.append({"role": "user", "content": message})

    # Phase 1: Get Primary Chat Response
    try:
        agent_state = agent.get_ui_update_state()
        agent_identity = agent.identity_statement
        agent_task_id = agent_state.get("current_task_id")
        agent_task_desc = agent_state.get("current_task_desc", "N/A")
        agent_action_desc = agent_state.get("current_action_desc", "N/A")
        agent_status = agent_state.get("status", "unknown")

        # Build context string (same logic as before)
        agent_activity_context = "Currently idle."
        if agent_task_id:
            agent_task_for_context = agent.task_queue.get_task(agent_task_id)
            agent_task_desc = (
                agent_task_for_context.description
                if agent_task_for_context
                else agent_task_desc + " (Missing?)"
            )
        if agent_status == "running" and agent_task_id:
            agent_activity_context = f"Working on Task {agent_task_id[:8]}: '{agent_task_desc}'. Status: {agent_status}. Current Activity: '{agent_action_desc}'."
        elif agent_status == "planning" and agent_task_id:
            agent_activity_context = f"Planning Task {agent_task_id[:8]}: '{agent_task_desc}'. Status: {agent_status}."
        elif agent_status == "paused":
            agent_activity_context = (
                f"Paused. Active task {agent_task_id[:8]}: '{agent_task_desc}' (Activity: {agent_action_desc})."
                if agent_task_id
                else "Paused and idle."
            )
        elif agent_status in ["shutdown", "critical_error"]:
            agent_activity_context = f"Agent state: '{agent_status}'."
        log.debug(
            f"[{agent.log_prefix}] Agent activity context: {agent_activity_context}"
        )

        history_context_list = [
            f"{t.get('role','?').capitalize()}: {t.get('content','')}"
            for t in history[-4:-1]
        ]
        history_context_str = "\n".join(history_context_list)

        # Retrieve memories (can still block, but maybe faster than LLM)
        log.info(f"[{agent.log_prefix}] Retrieving memories for chat context...")
        memory_query = f"Chat context for: '{message}'\nHistory:\n{history_context_str}\nAgent: {agent_activity_context}\nIdentity: {agent_identity}"
        relevant_memories, _ = agent.memory.retrieve_and_rerank_memories(
            query=memory_query,
            task_description="Responding in chat, considering identity/activity.",
            context=f"{history_context_str}\nActivity: {agent_activity_context}\nIdentity: {agent_identity}",
            identity_statement=agent.identity_statement,
            n_results=config.MEMORY_COUNT_CHAT_RESPONSE * 2,
            n_final=config.MEMORY_COUNT_CHAT_RESPONSE,
        )
        memory_display_text = format_memories_for_display(
            relevant_memories, context_label=f"Chat ({agent.agent_name})"
        )
        log.info(
            f"[{agent.log_prefix}] Retrieved {len(relevant_memories)} memories for chat."
        )

        # Build prompt (same logic)
        system_prompt = f"""You are helpful AI assistant ({agent_identity}). Answer user conversationally. Consider identity, chat history, memories, agent's background activity. Be aware of your capabilities and limitations."""
        memories_for_prompt_list = []
        snippet_length = 250
        for mem in relevant_memories:
            relative_time = format_relative_time(
                mem.get("metadata", {}).get("timestamp")
            )
            mem_type = mem.get("metadata", {}).get("type", "N/A")
            content_snippet = mem.get("content", "")
            content_snippet = (
                content_snippet[:snippet_length].strip() + "..."
                if len(content_snippet) > snippet_length
                else content_snippet.strip()
            )
            memories_for_prompt_list.append(
                f"- [Memory - {relative_time}] (Type: {mem_type}): {content_snippet}"
            )
        memories_for_prompt = "\n".join(memories_for_prompt_list) or "None provided."
        history_for_prompt = "\n".join(
            [
                f"{t.get('role','?').capitalize()}: {t.get('content','')}"
                for t in history
            ]
        )
        prompt = f"{system_prompt}\n\n## Agent Background Activity:\n{agent_activity_context}\n\n## Relevant Memory Snippets:\n{memories_for_prompt}\n\n## Chat History:\n{history_for_prompt}\n\n## Current Query:\nUser: {message}\nAssistant:"

        log.debug(
            f"[{agent.log_prefix}] Approximate prompt length for chat response: {len(prompt)} chars"
        )
        log.info(
            f"[{agent.log_prefix}] Asking {config.OLLAMA_CHAT_MODEL} for chat response..."
        )

        # Call LLM with timeout and error handling
        response_text = call_ollama_api(
            prompt=prompt,
            model=config.OLLAMA_CHAT_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            timeout=config.OLLAMA_TIMEOUT,  # Use configured timeout
        )

        if not response_text:
            assistant_response_text = "Sorry, an error occurred while generating the response. Please check the logs or try again."
            log.error(
                f"[{agent.log_prefix}] LLM call failed or returned empty for chat response."
            )
        else:
            assistant_response_text = response_text

        history.append({"role": "assistant", "content": assistant_response_text})

        # Yield the initial response
        yield history, memory_display_text, task_display_text, "", last_gen_id

    except Exception as e:
        log.exception(
            f"[{agent.log_prefix}] Error during initial chat processing phase: {e}"
        )
        error_message = f"Internal error processing message: {e}"
        # Ensure history has user message before adding error assistant message
        if not history or history[-1]["role"] != "assistant":
            history.append({"role": "assistant", "content": error_message})
        else:  # If error happened after adding assistant message, update it
            history[-1]["content"] = error_message
        yield history, f"Error:\n```\n{traceback.format_exc()}\n```", task_display_text, "", None
        return  # Stop processing on error

    # Phase 2: Background Actions (Memory Logging, Task Generation)
    # These happen *after* the initial response is yielded to the UI
    try:
        # Refresh agent status before logging memory or generating task
        agent_state = agent.get_ui_update_state()
        agent_status = agent_state.get("status", "unknown")
        agent_is_running = (
            agent_status == "running"
        )  # Consider running state based on UI status now

        # Log chat to memory if agent is running or paused (allow logging when paused)
        if agent_status in [
            "running",
            "paused",
            "planning",
            "idle",
        ]:  # More permissive logging
            log.info(
                f"[{agent.log_prefix}] Agent status '{agent_status}', adding chat to memory (background)..."
            )
            try:
                # Re-fetch context for logging, as it might have changed slightly
                agent_activity_context_log = "Currently idle."
                if agent_task_id:  # Use task ID from Phase 1 if still valid
                    agent_task_for_context_log = agent.task_queue.get_task(
                        agent_task_id
                    )
                    agent_task_desc_log = (
                        agent_task_for_context_log.description
                        if agent_task_for_context_log
                        else agent_state.get("current_task_desc", "N/A") + " (Missing?)"
                    )
                    if agent_status == "running":
                        agent_activity_context_log = f"Working on Task {agent_task_id[:8]}: '{agent_task_desc_log}'. Status: {agent_status}. Current Activity: '{agent_state.get('current_action_desc', 'N/A')}'."
                    elif agent_status == "planning":
                        agent_activity_context_log = f"Planning Task {agent_task_id[:8]}: '{agent_task_desc_log}'. Status: {agent_status}."
                    elif agent_status == "paused":
                        agent_activity_context_log = f"Paused. Active task {agent_task_id[:8]}: '{agent_task_desc_log}' (Activity: {agent_state.get('current_action_desc', 'N/A')})."

                agent.memory.add_memory(
                    content=f"User query: {message}",
                    metadata={"type": "chat_user_query"},
                )
                agent.memory.add_memory(
                    content=f"Assistant response: {assistant_response_text}",
                    metadata={
                        "type": "chat_assistant_response",
                        "agent_activity_at_time": agent_activity_context_log,  # Log current context
                        "agent_identity_at_time": agent.identity_statement,  # Log current identity
                    },
                )
            except Exception as mem_err:
                log.error(
                    f"[{agent.log_prefix}] Error adding chat to memory (background): {mem_err}"
                )
        else:
            log.info(
                f"[{agent.log_prefix}] Agent status '{agent_status}', skipping adding chat history to memory (background)."
            )

        # Check if task generation is needed
        should_gen = _should_generate_task(message, assistant_response_text)
        if should_gen:
            log.info(
                f"[{agent.log_prefix}] Interaction warrants task generation (background)..."
            )
            first_task_generated = agent.generate_new_tasks(
                max_new_tasks=1,
                last_user_message=message,
                last_assistant_response=assistant_response_text,
                trigger_context="chat",
            )
            if first_task_generated:
                # Try to find the ID of the generated task (best effort)
                newly_added_task_obj = None
                desc_match = first_task_generated.strip().lower()
                cutoff_time = (
                    datetime.datetime.now(datetime.timezone.utc)
                    - datetime.timedelta(seconds=15)
                ).isoformat()
                potential_matches = [
                    t
                    for t in agent.task_queue.tasks.values()
                    if t.description.strip().lower() == desc_match
                    and t.created_at >= cutoff_time
                    and t.status == "pending"
                ]
                if potential_matches:
                    newly_added_task_obj = sorted(
                        potential_matches, key=lambda t: t.created_at, reverse=True
                    )[0]

                if newly_added_task_obj:
                    last_gen_id = newly_added_task_obj.id
                    task_display_text = f'✅ Task Generated for {agent.agent_name} (ID: {last_gen_id[:8]}...):\n"{first_task_generated}"'
                    log.info(
                        f"[{agent.log_prefix}] Task generated (ID: {last_gen_id}): {first_task_generated[:60]}..."
                    )

                    # Add notification to the *last* message in history
                    notification = f"\n\n*(Okay, based on our chat, I've created task {last_gen_id[:8]}... for Agent {agent.agent_name}: \"{first_task_generated}\". I'll work on it when possible. You can prioritize it if needed.)*"
                    if history and history[-1]["role"] == "assistant":
                        history[-1]["content"] += notification

                    # Save QLoRA datapoint (can also error but less critical)
                    try:
                        agent._save_qlora_datapoint(
                            source_type="chat_task_generation",
                            instruction="User interaction led to this task. Respond confirming task creation.",
                            input_context=f"Identity: {agent.identity_statement}\nUser: {message}\nActivity: {agent_activity_context}",  # Use context from phase 1
                            output=f"{assistant_response_text}{notification}",
                        )
                    except Exception as qlora_err:
                        log.error(
                            f"[{agent.log_prefix}] Failed to save QLoRA datapoint (background): {qlora_err}"
                        )

                else:
                    task_display_text = f'✅ Task Generated for {agent.agent_name} (ID unknown - check state tab):\n"{first_task_generated}"'
                    log.warning(
                        f"[{agent.log_prefix}] Task generated but could not find its ID immediately: {first_task_generated}"
                    )
                    # Add simpler notification if ID unknown
                    notification = f"\n\n*(Okay, based on our chat, I've created a task for Agent {agent.agent_name}: \"{first_task_generated}\". I'll work on it when possible.)*"
                    if history and history[-1]["role"] == "assistant":
                        history[-1]["content"] += notification

            else:
                task_display_text = (
                    "(Eval suggested task, but none generated/added by agent)"
                )
                log.info(
                    f"[{agent.log_prefix}] Task gen warranted but no task added (background)."
                )
        else:
            log.info(
                f"[{agent.log_prefix}] Task generation not warranted (background)."
            )
            task_display_text = "(No new task warranted based on conversation)"

        # Yield the final state after background actions
        # This updates the UI panels after background processing is done
        yield history, memory_display_text, task_display_text, "", last_gen_id

    except Exception as e:
        log.exception(
            f"[{agent.log_prefix}] Error during background chat processing phase: {e}"
        )
        # Update task display text with error
        task_display_text = f"Error in background processing: {e}"
        # Yield final state with error message
        yield history, memory_display_text, task_display_text, "", last_gen_id


# End Chat Response Refactor


# Agent State Tab functions <<< MODIFIED >>>
def refresh_agent_state_display():
    log.info(
        f"State Tab: Refresh button clicked. Fetching latest state for agent '{active_agent_id}'..."
    )
    if not initialization_success or not agents or active_agent_id not in agents:
        error_df = pd.DataFrame(columns=["Error"])
        return (
            "Agent not initialized.",
            error_df,  # pending
            error_df,  # inprogress
            error_df,  # completed
            error_df,  # failed
            "Error: Agent not initialized.",  # memory summary
            gr.Dropdown(choices=["Error"], value="Error"),  # task memory select
            error_df,  # task memory df
            [],  # pending data state
            [],  # inprogress data state
            [],  # completed data state
            [],  # failed data state
            "(No details to show)",  # task details markdown
        )
    try:
        agent = agents[active_agent_id]
        state_data = agent.get_agent_dashboard_state()

        pending_cols = ["ID", "Priority", "Description", "Depends On", "Created"]
        inprogress_cols = ["ID", "Priority", "Description", "Status", "Created"]
        completed_cols = ["ID", "Description", "Completed At", "Result Snippet"]
        failed_cols = ["ID", "Description", "Failed At", "Reason", "Reattempts"]
        memory_cols = ["Relative Time", "Timestamp", "Type", "Content Snippet", "ID"]

        pending_raw = state_data.get("pending_tasks", [])
        inprogress_raw = state_data.get("in_progress_tasks", [])
        completed_raw = state_data.get("completed_tasks", [])
        failed_raw = state_data.get("failed_tasks", [])

        pending_df = format_display_df(pending_raw, pending_cols)
        inprogress_df = format_display_df(inprogress_raw, inprogress_cols)
        completed_df = format_display_df(completed_raw, completed_cols)
        failed_df = format_display_df(failed_raw, failed_cols)

        completed_failed_tasks_data = state_data.get("completed_failed_tasks_data", [])
        task_id_choices_tuples = [
            (f"{t['ID'][:8]} - {t['Description'][:50]}...", t["ID"])
            for t in completed_failed_tasks_data
            if "ID" in t and "Description" in t
        ]
        dropdown_choices = [("Select Task ID...", None)] + task_id_choices_tuples
        initial_dropdown_value = None

        return (
            state_data.get("identity_statement", "Error loading identity"),
            pending_df,
            inprogress_df,
            completed_df,
            failed_df,
            state_data.get("memory_summary", "Error loading summary"),
            gr.Dropdown(
                choices=dropdown_choices,
                value=initial_dropdown_value,
                label="Select Task ID (Completed/Failed for Memory View)",  # Modified label
                interactive=True,
            ),
            pd.DataFrame(columns=memory_cols),  # Placeholder for task-specific memories
            pending_raw,
            inprogress_raw,
            completed_raw,
            failed_raw,
            "(Select a row from a task table above to see details)",
        )
    except Exception as e:
        log.exception(
            f"Error refreshing agent state display for agent {active_agent_id}"
        )
        error_df = pd.DataFrame([{"Error": str(e)}])
        return (
            f"Error: {e}",
            error_df,
            error_df,
            error_df,
            error_df,
            f"Memory Summary Error: {e}",
            gr.Dropdown(choices=["Error"], value="Error"),
            error_df,
            [],
            [],
            [],
            [],
            f"Error loading state details: {e}",
        )


# update_task_memory_display remains largely the same, but fetches relative time
def update_task_memory_display(selected_task_id: str):
    log.debug(
        f"State Tab: Task ID selected: {selected_task_id} for agent {active_agent_id}"
    )
    columns = ["Relative Time", "Timestamp", "Type", "Content Snippet", "ID"]
    if (
        not initialization_success
        or not agents
        or active_agent_id not in agents
        or not selected_task_id
    ):
        return pd.DataFrame(columns=columns)
    try:
        agent = agents[active_agent_id]
        memories = agent.get_formatted_memories_for_task(selected_task_id)
        # Add relative time here if not added by the agent function
        for mem in memories:
            if "Relative Time" not in mem:
                mem["Relative Time"] = format_relative_time(mem.get("Timestamp"))

        mem_df = format_display_df(memories, columns)
        if not mem_df.empty and "Timestamp" in mem_df.columns:
            # Sort ascending by original timestamp string
            mem_df = mem_df.sort_values(
                by="Timestamp",
                ascending=True,
                na_position="first",
                key=lambda col: pd.to_datetime(col, errors="coerce"),
            )
        return mem_df

    except Exception as e:
        log.exception(
            f"Error fetching memories for task {selected_task_id} on agent {active_agent_id}"
        )
        return pd.DataFrame([{"Error": str(e)}])


# show_task_details remains unchanged (operates on data passed via state)
def show_task_details(evt: gr.SelectData, task_list: list) -> str:
    if not task_list or not isinstance(task_list, list):
        return "(Internal Error: Invalid task list provided for details view)"
    try:
        row_index = evt.index[0]
        if 0 <= row_index < len(task_list):
            task_data = task_list[row_index]
            if isinstance(task_data, dict):
                details = [f"### Task Details (Row {row_index+1})"]
                # Sort keys for consistent display
                for key in sorted(task_data.keys()):
                    value = task_data[key]
                    display_value = str(value) if value is not None else "N/A"
                    # Check for multi-line strings or long strings for code block formatting
                    if isinstance(value, str) and ("\n" in value or len(value) > 100):
                        details.append(f"**{key}:**\n```text\n{value}\n```")
                    else:
                        details.append(f"**{key}:** `{display_value}`")
                return "\n\n".join(details)
            else:
                return f"(Internal Error: Task data at index {row_index} is not a dictionary)"
        else:
            # This happens when selection is cleared
            return "(Selection cleared. Please re-select a row.)"
    except IndexError:
        # This might happen if the dataframe updates while a cell is selected
        return "(Error: Could not get selected row index. Table might have changed.)"
    except Exception as e:
        log.exception(f"Error formatting task details: {e}")
        return f"(Error displaying task details: {e})"


# Gradio UI Definition <<< MODIFIED >>>
log.info("Defining Gradio UI...")
if not initialization_success or not agents:
    log.critical("Agent initialization failed. Cannot define UI.")
    # Define a minimal error UI if initialization failed *before* this point
    with gr.Blocks() as demo:
        gr.Markdown("# Fatal Error\nAgent backend failed to initialize. Check logs.")
        gr.Textbox(
            label="Startup Log",
            value=initial_console_log_text,
            lines=20,
            interactive=False,
        )

else:
    # Agent Dropdown Choices
    agent_choices = [
        (f"{agent_config['name']} ({agent_id})", agent_id)
        for agent_id, agent_config in config.AGENTS.items()
    ]

    with gr.Blocks(
        theme=gr.themes.Glass(), title="Multi-Agent Control Center", fill_width=True
    ) as demo:
        gr.Markdown("# Multi-Agent Control Center & Chat")

        # UI State Variables
        # Monitor Tab uses _ui_update_state directly via update_monitor_ui
        # Chat Tab uses chatbot history and other states defined in the tab
        # State Tab uses states defined below
        pending_tasks_data_state = gr.State([])
        inprogress_tasks_data_state = gr.State([])
        completed_tasks_data_state = gr.State([])
        failed_tasks_data_state = gr.State([])
        # Action History Tab states
        full_action_history_state = gr.State(
            initial_action_history_data
        )  # Holds ALL history for the agent
        hist_filtered_action_data_state = gr.State([])  # Holds history filtered by task
        hist_current_index_state = gr.State(str(0))  # Index within the *filtered* list

        # Agent Selection Row (appears above tabs)
        with gr.Accordion("Agent Control", open=True):
            with gr.Row():
                agent_selector_dropdown = gr.Dropdown(
                    label="Active Agent",
                    choices=agent_choices,
                    value=active_agent_id,  # Initial value
                    interactive=True,
                    scale=1,
                )
                monitor_status_bar = gr.Textbox(
                    label="Active Agent Status",
                    show_label=True,
                    value=initial_status_text,
                    interactive=False,
                    lines=1,
                    scale=1,
                )
                agent_switch_feedback = gr.Textbox(
                    label="Agent Switch Status",
                    value=f"Initial agent: {initial_agent_name}",
                    interactive=False,
                    lines=1,
                    scale=1,
                )
                suggestion_feedback_box = gr.Textbox(
                    label="User Suggestion",
                    show_label=True,
                    value="N/A",
                    interactive=False,
                    lines=1,
                    scale=1,
                )
            with gr.Row():
                start_resume_btn = gr.Button("▶️ Start / Resume", variant="primary")
                pause_btn = gr.Button("⏸️ Pause")
                suggest_change_btn = gr.Button("⏭️ Suggest Task Change")

        with gr.Tabs():
            # Monitor Tab <<< MODIFIED >>>
            with gr.TabItem("Agent Monitor"):
                with gr.Accordion("Console Log", open=True):
                    monitor_log = gr.Textbox(
                        label="🪵 Console Log",
                        value=initial_console_log_text,
                        lines=20,
                        max_lines=20,
                        autoscroll=True,
                        interactive=False,
                        show_copy_button=True,
                    )
                with gr.Accordion("Current Task & State", open=True):
                    with gr.Row():
                        with gr.Column(scale=1):
                            monitor_current_task = gr.Textbox(
                                label="❤️ Task Status",
                                value=initial_task_status_text,
                                lines=4,
                                interactive=False,
                            )
                            monitor_dependent_tasks = gr.Textbox(
                                value=initial_dependent_tasks_text,
                                show_label=False,
                                lines=2,
                                interactive=False,
                            )
                            monitor_plan = gr.Textbox(
                                label="📋 Intended Plan (Guidance)",
                                value=initial_plan_text,
                                lines=7,
                                interactive=False,
                            )
                        with gr.Column(scale=1):
                            monitor_current_action = gr.Textbox(
                                label="⚡ Current Action Status",
                                value=initial_action_desc_text,
                                show_label=True,
                                lines=2,
                                interactive=False,
                            )
                            monitor_thinking = gr.Textbox(
                                label="🤖 Agent Thinking (Current Cycle)",
                                value=initial_thinking_text,
                                lines=8,
                                interactive=False,
                                show_copy_button=True,
                            )

                with gr.Accordion("Tools & Memory", open=True):
                    with gr.Row():
                        with gr.Column(scale=1):
                            monitor_tool_results = gr.Markdown(
                                value=initial_tool_results_text,
                                label="🛠️ Tool Output / Browse Content",
                            )
                        with gr.Column(scale=1):
                            monitor_memory = gr.Markdown(
                                value=format_memories_for_display(
                                    [], f"Monitor ({initial_agent_name})"
                                ),
                                label="🧠 Relevant Memories (Task-Specific)",
                            )
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("##### 🕰️ General Memories (Recent)")
                            monitor_general_memory_display = gr.DataFrame(
                                value=initial_general_memory_data,
                                headers=[
                                    "Relative Time",
                                    "Timestamp",
                                    "Type",
                                    "Content Snippet",
                                    "ID",
                                ],
                                interactive=False,
                                wrap=True,
                            )

                # Monitor Button Clicks
                start_resume_btn.click(
                    fn=start_agent_processing, inputs=None, outputs=monitor_status_bar
                )
                pause_btn.click(
                    fn=pause_agent_processing, inputs=None, outputs=monitor_status_bar
                )
                suggest_change_btn.click(
                    fn=suggest_task_change, inputs=None, outputs=suggestion_feedback_box
                )

                # Timer updates the UI based on the *active* agent and handles auto-switch
                timer = gr.Timer(config.UI_UPDATE_INTERVAL)
                monitor_timer_outputs = [
                    monitor_current_task,
                    monitor_current_action,
                    monitor_plan,
                    monitor_dependent_tasks,
                    monitor_thinking,
                    monitor_log,
                    monitor_memory,
                    monitor_general_memory_display,
                    monitor_tool_results,
                    monitor_status_bar,
                    agent_selector_dropdown,
                    full_action_history_state,  # Update full history state for history tab
                ]
                timer.tick(
                    fn=update_monitor_ui, inputs=None, outputs=monitor_timer_outputs
                )

            # Chat Tab (Unchanged)
            with gr.TabItem("Chat"):
                gr.Markdown(
                    "Interact with the **currently selected agent**. It considers its identity, activity, and your input. Task generation happens *after* the response appears."
                )
                last_generated_task_id_state = gr.State(None)
                with gr.Row():
                    with gr.Column(scale=3):
                        chat_chatbot = gr.Chatbot(
                            label="Conversation",
                            type="messages",
                            height=400,
                            show_copy_button=True,
                            render=False,
                        )
                        chat_chatbot.render()
                        with gr.Row():
                            chat_task_panel = gr.Textbox(
                                label="💡 Last Generated Task (Chat)",
                                value="(No task generated yet)",
                                lines=3,
                                interactive=False,
                                show_copy_button=True,
                                scale=4,
                            )
                            prioritize_task_btn = gr.Button("Prioritize Task", scale=1)
                        chat_interaction_feedback = gr.Textbox(
                            label="Chat/Task Interaction Feedback",
                            value="",
                            interactive=False,
                            lines=1,
                        )
                        with gr.Row():
                            chat_msg_input = gr.Textbox(
                                label="Your Message / Task Description",
                                placeholder="Type message for chat with active agent, or text for new priority task...",
                                lines=3,
                                scale=4,
                                container=False,
                            )
                            chat_send_button = gr.Button(
                                "Send Chat", variant="primary", scale=1
                            )
                            create_task_btn = gr.Button("Create Priority Task", scale=1)
                    with gr.Column(scale=1):
                        chat_memory_panel = gr.Markdown(
                            value="Relevant Memories (Chat)\n...",
                            label="Memory Context",
                        )
                # Chat functions operate on the active agent
                chat_outputs = [
                    chat_chatbot,
                    chat_memory_panel,
                    chat_task_panel,
                    chat_msg_input,
                    last_generated_task_id_state,
                ]
                chat_event = chat_send_button.click(
                    fn=chat_response,
                    inputs=[chat_msg_input, chat_chatbot],
                    outputs=chat_outputs,
                    queue=True,
                )
                chat_msg_input.submit(
                    fn=chat_response,
                    inputs=[chat_msg_input, chat_chatbot],
                    outputs=chat_outputs,
                    queue=True,
                )
                prioritize_task_btn.click(
                    fn=prioritize_generated_task,
                    inputs=[last_generated_task_id_state],
                    outputs=[chat_interaction_feedback],
                )
                create_task_btn.click(
                    fn=create_priority_task,
                    inputs=[chat_msg_input],
                    outputs=[chat_interaction_feedback],
                ).then(lambda: "", outputs=[chat_msg_input])

            # Agent State Tab (Unchanged from previous step)
            with gr.TabItem("Agent State"):
                gr.Markdown(
                    "View the **currently selected agent's** identity, task queues, and memory summary. **Use button to load/refresh data.** Click rows in task tables to see full details. Task-specific memory view available here."
                )
                with gr.Row():
                    state_identity = gr.Textbox(
                        label="Agent Identity Statement",
                        lines=3,
                        interactive=False,
                        value="(Press Load State)",
                        show_copy_button=True,
                    )
                    load_state_button = gr.Button(
                        "🔄 Load/Refresh Agent State", variant="primary"
                    )
                with gr.Accordion("Task Status", open=True):
                    with gr.Column():
                        gr.Markdown("#### Pending Tasks (Highest Priority First)")
                        state_pending_tasks = gr.DataFrame(
                            headers=[
                                "ID",
                                "Priority",
                                "Description",
                                "Depends On",
                                "Created",
                            ],
                            interactive=True,
                            wrap=True,
                        )
                        gr.Markdown("#### In Progress / Planning Tasks")
                        state_inprogress_tasks = gr.DataFrame(
                            headers=[
                                "ID",
                                "Priority",
                                "Description",
                                "Status",
                                "Created",
                            ],
                            interactive=True,
                            wrap=True,
                        )
                        gr.Markdown("#### Completed Tasks (Most Recent First)")
                        state_completed_tasks = gr.DataFrame(
                            headers=[
                                "ID",
                                "Description",
                                "Completed At",
                                "Result Snippet",
                            ],
                            interactive=True,
                            wrap=True,
                        )
                        gr.Markdown("#### Failed Tasks (Most Recent First)")
                        state_failed_tasks = gr.DataFrame(
                            headers=[
                                "ID",
                                "Description",
                                "Failed At",
                                "Reason",
                                "Reattempts",
                            ],
                            interactive=True,
                            wrap=True,
                        )
                        gr.Markdown("\n ### Selected Task Details:")
                        state_details_display = gr.Markdown(
                            "(Select a row from a task table above to see details)"
                        )
                with gr.Accordion("Memory Explorer", open=True):
                    state_memory_summary = gr.Markdown(
                        "Memory Summary\n(Press Load State)"
                    )
                    with gr.Column():
                        gr.Markdown("##### Task-Specific Memories")
                        with gr.Row():
                            state_task_memory_select = gr.Dropdown(
                                label="Select Task ID (Completed/Failed for Memory View)",
                                choices=[("Select Task ID...", None)],
                                value=None,
                                interactive=True,
                            )
                        state_task_memory_display = gr.DataFrame(
                            headers=[
                                "Relative Time",
                                "Timestamp",
                                "Type",
                                "Content Snippet",
                                "ID",
                            ],
                            interactive=False,
                            wrap=True,
                        )

                # State tab functions operate on the active agent
                state_tab_outputs = [
                    state_identity,
                    state_pending_tasks,
                    state_inprogress_tasks,
                    state_completed_tasks,
                    state_failed_tasks,
                    state_memory_summary,
                    state_task_memory_select,
                    state_task_memory_display,
                    pending_tasks_data_state,
                    inprogress_tasks_data_state,
                    completed_tasks_data_state,
                    failed_tasks_data_state,
                    state_details_display,
                ]
                load_state_button.click(
                    fn=refresh_agent_state_display,
                    inputs=None,
                    outputs=state_tab_outputs,
                    queue=True,
                )
                state_task_memory_select.change(
                    fn=update_task_memory_display,
                    inputs=[state_task_memory_select],
                    outputs=[state_task_memory_display],
                    queue=True,
                )
                # Connect select events for each task dataframe (unchanged)
                state_pending_tasks.select(
                    fn=show_task_details,
                    inputs=[pending_tasks_data_state],
                    outputs=[state_details_display],
                )
                state_inprogress_tasks.select(
                    fn=show_task_details,
                    inputs=[inprogress_tasks_data_state],
                    outputs=[state_details_display],
                )
                state_completed_tasks.select(
                    fn=show_task_details,
                    inputs=[completed_tasks_data_state],
                    outputs=[state_details_display],
                )
                state_failed_tasks.select(
                    fn=show_task_details,
                    inputs=[failed_tasks_data_state],
                    outputs=[state_details_display],
                )

            # Action History Tab <<< MODIFIED >>>
            with gr.TabItem("Action History"):
                gr.Markdown(
                    "Explore the step-by-step action history for the **currently selected agent**. "
                    "Use the dropdown to filter by Task ID."
                )
                with gr.Row():
                    hist_task_filter_dropdown = gr.Dropdown(
                        label="Filter by Task",
                        choices=initial_task_filter_choices,
                        value="ALL",
                        interactive=True,
                        scale=3,  # Give dropdown more space
                    )
                    # <<< ADDED Refresh Button >>>
                    hist_refresh_task_list_btn = gr.Button(
                        "🔄 Refresh Task List", scale=1, min_width=50
                    )

                with gr.Row():  # Navigation Buttons Row
                    hist_prev_step_btn = gr.Button("◀ Previous Step", scale=1)
                    hist_next_step_btn = gr.Button("Next Step ▶", scale=1)

                hist_title = gr.Markdown(
                    "**Action Details** (Filter and navigate to view)"
                )
                hist_objective = gr.Markdown(
                    label="Action Objective", value="(No history selected)"
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("##### Thinking Process")
                        hist_thinking = gr.Textbox(
                            label="Thinking",
                            lines=15,
                            interactive=False,
                            show_copy_button=True,
                        )
                        gr.Markdown("##### Log Snippet")
                        hist_log = gr.Textbox(
                            label="Log Snippet", lines=5, interactive=False
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("##### LLM Prompt")
                        hist_llm_prompt = gr.Textbox(
                            label="LLM Prompt Used",
                            lines=15,
                            interactive=False,
                            show_copy_button=True,
                        )
                        gr.Markdown("##### Action & Result")
                        hist_result = gr.Markdown(
                            label="Action/Result Summary",
                            value="(No history selected)",
                        )

                # Action History Navigation and Filtering
                hist_view_outputs = [
                    hist_current_index_state,
                    hist_title,
                    hist_objective,
                    hist_thinking,
                    hist_llm_prompt,  # Added prompt output
                    hist_log,
                    hist_result,
                    hist_filtered_action_data_state,  # Update filtered data state
                ]

                # Refresh Task Filter Button Click <<< NEW >>>
                hist_refresh_task_list_btn.click(
                    fn=update_history_task_filter_choices,
                    inputs=None,
                    outputs=[hist_task_filter_dropdown],
                    queue=True,  # Refreshing might involve agent calls
                )

                # Filter Dropdown Change
                hist_task_filter_dropdown.change(
                    fn=filter_and_get_action_details,
                    inputs=[
                        hist_task_filter_dropdown,
                        gr.State(str(0)),  # Reset index to 0 on filter change
                        gr.State(0),  # Delta 0 to just get the first item
                        full_action_history_state,
                    ],
                    outputs=hist_view_outputs,
                    queue=True,
                )

                # Navigation Button Clicks
                hist_prev_step_btn.click(
                    fn=filter_and_get_action_details,
                    inputs=[
                        hist_task_filter_dropdown,
                        hist_current_index_state,
                        gr.State(-1),
                        full_action_history_state,
                    ],
                    outputs=hist_view_outputs,
                )
                hist_next_step_btn.click(
                    fn=filter_and_get_action_details,
                    inputs=[
                        hist_task_filter_dropdown,
                        hist_current_index_state,
                        gr.State(1),
                        full_action_history_state,
                    ],
                    outputs=hist_view_outputs,
                )

                # Function to load the initial/latest history view for the selected task
                def load_initial_history_view(task_id, full_history):
                    # Find the latest index for the filtered history
                    if task_id == "ALL" or not task_id:
                        filtered_hist = full_history
                    else:
                        filtered_hist = [
                            item
                            for item in full_history
                            if item.get("task_id") == task_id
                        ]

                    latest_index = len(filtered_hist) - 1 if filtered_hist else 0

                    return filter_and_get_action_details(
                        selected_task_id=task_id,
                        current_index_str=str(
                            latest_index
                        ),  # Go to latest index for this filter
                        delta=0,
                        full_history_data=full_history,
                    )

                # Load initial view when the tab becomes visible or agent changes (chained later)
                # Connect the dropdown update and refresh button to trigger initial load
                hist_refresh_task_list_btn.click(
                    fn=load_initial_history_view,
                    inputs=[hist_task_filter_dropdown, full_action_history_state],
                    outputs=hist_view_outputs,
                )
                # Also trigger on initial demo load
                demo.load(
                    fn=load_initial_history_view,
                    inputs=[gr.State("ALL"), full_action_history_state],
                    outputs=hist_view_outputs,
                )

        # Agent Switching Handler <<< MODIFIED >>>
        agent_selector_dropdown.change(
            fn=switch_active_agent,
            inputs=[agent_selector_dropdown],
            outputs=[
                monitor_status_bar,
                agent_switch_feedback,
            ],
        ).then(  # Refresh Monitor Tab
            fn=update_monitor_ui,
            inputs=None,
            outputs=monitor_timer_outputs,
        ).then(  # Refresh State Tab
            fn=refresh_agent_state_display,
            inputs=None,
            outputs=state_tab_outputs,
            queue=True,
        ).then(  # Refresh History Tab Filter Choices
            fn=update_history_task_filter_choices,
            inputs=None,
            outputs=[hist_task_filter_dropdown],
        ).then(  # Reset History View after filter update & switch
            fn=lambda: filter_and_get_action_details(
                "ALL", "0", 0, []
            ),  # Reset to show 'no data' for 'ALL' tasks
            inputs=None,
            outputs=hist_view_outputs,
        ).then(  # Clear Chat
            fn=lambda: (
                [],
                "(Memory cleared on agent switch)",
                "(Task status cleared)",
                None,
            ),
            inputs=None,
            outputs=[
                chat_chatbot,
                chat_memory_panel,
                chat_task_panel,
                last_generated_task_id_state,
            ],
        )

# Add print statement right after defining UI
print("DEBUG: Gradio UI definition complete.", file=original_stderr)

# Launch App (Unchanged)
if __name__ == "__main__":
    if initialization_success and agents:
        print(
            "DEBUG: Initialization successful, preparing to launch Gradio...",
            file=original_stderr,
        )
        log.info("Launching Gradio App Interface...")
        try:
            log.info("Starting Gradio server...")
            demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
            print(
                "DEBUG: demo.launch() returned (Gradio server stopped).",
                file=original_stderr,
            )
            log.info("Gradio App Interface exited normally.")

        except KeyboardInterrupt:
            log.info("Gradio launch interrupted by KeyboardInterrupt (Ctrl+C).")
        except Exception as e:
            log.critical(f"Gradio launch failed: {e}", exc_info=True)
            print(f"FATAL: Gradio launch failed: {e}", file=original_stderr)
        finally:
            # Shutdown Sequence
            log.info("Initiating shutdown sequence...")
            app_shutdown_requested.set()

            # Ensure agents are shut down cleanly
            shutdown_threads = []
            for agent_id, agent in agents.items():
                if agent:
                    log.info(f"Requesting shutdown for agent '{agent_id}'...")
                    thread = threading.Thread(
                        target=agent.shutdown, name=f"Shutdown-{agent_id}"
                    )
                    thread.start()
                    shutdown_threads.append((agent_id, thread))

            # Wait for agent shutdown threads to complete with timeout
            shutdown_timeout = 15
            start_time = time.time()
            for agent_id, thread in shutdown_threads:
                remaining_time = shutdown_timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    log.warning(
                        f"Timeout exceeded waiting for agent '{agent_id}' shutdown."
                    )
                    break
                log.info(
                    f"Waiting for agent '{agent_id}' shutdown (timeout: {remaining_time:.1f}s)..."
                )
                thread.join(timeout=remaining_time)
                if thread.is_alive():
                    log.warning(
                        f"Agent '{agent_id}' shutdown thread did not join cleanly after timeout."
                    )
                else:
                    log.info(f"Agent '{agent_id}' shutdown thread joined.")

            save_last_active_agent()

            # Close and remove main log handler
            if main_log_handler:
                log.info("Closing main log file handler.")
                try:
                    main_log_handler.close()
                    root_logger.removeHandler(main_log_handler)
                except Exception as log_close_err:
                    print(
                        f"Error closing main log handler: {log_close_err}",
                        file=original_stderr,
                    )

            # Close and remove UI console handler
            if "ui_console_handler" in locals() and ui_console_handler:
                log.info("Closing UI console log handler.")
                try:
                    ui_console_handler.close()
                    root_logger.removeHandler(ui_console_handler)
                except Exception as log_close_err:
                    print(
                        f"Error closing UI console handler: {log_close_err}",
                        file=original_stderr,
                    )

            log.info("Shutdown sequence complete.")
            print("\nMulti-Agent App End", file=original_stdout)

    else:
        # Minimal error UI if initialization failed
        sys.stderr = original_stderr
        print(
            "Agent initialization failed. Launching minimal error UI.", file=sys.stderr
        )
        log.critical("Agent initialization failed. Launching minimal error UI.")
        try:
            if "demo" in locals() and isinstance(demo, gr.Blocks):
                log.info("Launching pre-defined error UI.")
                demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
            else:
                log.info("Defining and launching minimal error UI.")
                with gr.Blocks() as error_demo:
                    gr.Markdown(
                        "# Fatal Error\nAgent backend failed to initialize. Check logs."
                    )
                    gr.Textbox(
                        label="Startup Log Snippet",
                        value=initial_console_log_text,
                        lines=20,
                        interactive=False,
                    )
                error_demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
        except Exception as e:
            log.critical(f"Gradio error UI launch failed: {e}", exc_info=True)
            print(f"\nFATAL: Gradio error UI launch also failed: {e}", file=sys.stderr)
            if main_log_handler:
                try:
                    main_log_handler.close()
                except:
                    pass
            if "ui_console_handler" in locals() and ui_console_handler:
                try:
                    ui_console_handler.close()
                except:
                    pass
            sys.exit("Gradio error UI launch failed.")
