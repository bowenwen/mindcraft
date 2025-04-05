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
from typing import List, Tuple, Optional, Dict, Any, Deque
import pandas as pd
from collections import deque
import html

# --- Project Imports ---
import config
from utils import call_ollama_api, format_relative_time
from memory import AgentMemory, setup_chromadb
from task_manager import TaskQueue
from data_structures import Task # <<< ADDED Task import
from agent import AutonomousAgent
import chromadb

# --- Logging Setup ---
import logging

# --- NEW: Console Log Capture Setup ---
# Global buffer to hold the last N characters of console output
console_log_buffer: deque = deque(maxlen=config.UI_LOG_MAX_LENGTH)


class ConsoleLogHandler(io.StringIO):
    """A custom stream handler that writes to a deque."""

    def __init__(self, buffer: deque):
        super().__init__()
        self.buffer = buffer

    def write(self, msg):
        # Append characters to the deque. Deque handles maxlen automatically.
        for char in msg:
            self.buffer.append(char)
        # Optional: If you still want the original behavior (e.g., writing to actual console)
        # sys.__stdout__.write(msg)

    def flush(self):
        # Optional: Implement if needed, e.g., for compatibility
        pass


# Setup logging *before* redirection if you want setup messages on the console
logging.basicConfig(
    level=config.LOG_LEVEL,
    format="[%(asctime)s] [%(levelname)s][%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # handlers=[logging.StreamHandler(sys.__stdout__)] # Optionally log basicConfig to original stdout
)
log = logging.getLogger("AppUI")

# --- End Console Log Capture Setup ---


# --- Global Variables / Setup ---
agent_instance: Optional[AutonomousAgent] = None
try:
    mem_collection = setup_chromadb()
    if mem_collection is None:
        raise RuntimeError("Memory Database setup failed.")
    # Document archive DB setup happens within the WebTool now
    agent_instance = AutonomousAgent(memory_collection=mem_collection)
    log.info("App components initialized successfully.")
except Exception as e:
    log.critical(f"Fatal error during App component initialization: {e}", exc_info=True)
    print(  # Use print here as logging might be redirected later
        f"\n\nFATAL ERROR: Could not initialize agent components: {e}\n",
        file=sys.stderr,
    )
    agent_instance = None

initial_status_text = "Error: Agent not initialized."
initial_thinking_text = "(Agent Initializing...)"
initial_tool_results_text = "(No tool results yet)"
initial_dependent_tasks_text = "Dependent Tasks: (None)"
initial_step_history_data: List[Dict] = []
initial_step_desc_text = "(Agent Initializing - Step)"
initial_plan_text = "(Agent Initializing - Plan)"
initial_task_status_text = "(Agent Initializing - Task Status)"
initial_console_log_text = "(Console Log Initializing...)"  # <<< NEW initial value

if agent_instance:
    log.info("Setting initial agent state to paused for UI...")
    agent_instance.start_autonomous_loop()
    agent_instance.pause_autonomous_loop()
    initial_state = agent_instance.get_ui_update_state()
    task_status = initial_state.get("status", "paused")
    initial_status_text = (
        f"Agent Status: {task_status} @ {initial_state.get('timestamp', 'N/A')}"
    )
    initial_thinking_text = initial_state.get("thinking", "(Agent Paused/Idle)")
    initial_step_history_data = initial_state.get("step_history", [])
    initial_step_desc_text = initial_state.get("current_step_desc", "N/A")
    initial_plan_text = initial_state.get("current_plan", "(None)")
    initial_task_status_text = f"ID: {initial_state.get('current_task_id', 'None')} ({task_status})\n\nDescription: {initial_state.get('current_task_desc', 'N/A')}"
    # Don't read console buffer here, it might not be fully populated yet
    initial_console_log_text = "".join(console_log_buffer) or initial_console_log_text

    log.info(f"Calculated initial status text for UI: {initial_status_text}")


# --- Global variable to store the last known Monitor UI state ---
# Updated tuple size to 10
last_monitor_state: Optional[
    Tuple[str, str, str, str, str, str, str, str, str, str]
] = None

# --- Formatting Functions ---


def format_memories_for_display(
    memories: List[Dict[str, Any]], context_label: str
) -> str:
    """Formats a list of memories for markdown display using expandable sections."""
    if not memories:
        return f"No relevant memories found for {context_label}."

    output_parts = [f"🧠 **Recent/Relevant Memories ({context_label}):**\n"]
    for i, mem in enumerate(memories[:7]):  # Limit display count
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

        safe_content = html.escape(content)  # Use html escape

        summary_line = (
            f"**{i+1}. {relative_time} - Type:** {mem_type} (Dist: {dist_str})"
        )
        content_preview = safe_content[:100].replace("\n", " ") + (
            "..." if len(safe_content) > 100 else ""
        )

        details_block = f"""
<details>
  <summary>{summary_line}: {content_preview}</summary>
  <pre style="background-color: #f0f0f0; border: 1px solid #ccc; padding: 5px; margin-top: 5px; white-space: pre-wrap; word-wrap: break-word;">{safe_content}</pre>
</details>
"""
        output_parts.append(details_block)

    return "\n".join(output_parts)


# --- Tool Result Formatting Functions ---


def format_web_search_results(results_data: Optional[Dict[str, Any]]) -> str:
    """Formats web search results for display."""
    # Use 'result' key which now contains the actual tool output
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
        if url:
            output.append(f"   [{url}]({url})")
        output.append(f"   > {snippet}\n")
    return "\n".join(output)


def format_memory_search_results(results_data: Optional[Dict[str, Any]]) -> str:
    """Formats memory search results for display."""
    # Use 'result' key
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
        # mem_id = res.get("memory_id", "N/A") # Optional
        output.append(f"**Rank {rank} ({rel_time})** - Type: {mem_type}, Dist: {dist}")
        output.append(f"   > {snippet}\n")
    return "\n".join(output)


def format_web_browse_results(results_data: Optional[Dict[str, Any]]) -> str:
    """Formats standard web browse results (full page, potentially truncated)."""
    # Use 'result' key
    actual_results = (
        results_data.get("result", {}) if isinstance(results_data, dict) else {}
    )
    if not actual_results or not isinstance(actual_results, dict):
        return "(Invalid web browse results data)"

    # Check if this was actually a query result mistakenly sent here
    if actual_results.get("query_mode") is True:
        log.warning(
            "format_web_browse_results called with query_mode=True data. Routing to query formatter."
        )
        return format_web_browse_query_results(results_data)  # Reroute

    content = actual_results.get("content")
    url = actual_results.get("url", "N/A")
    source = actual_results.get("content_source", "N/A")
    message = actual_results.get("message", "")
    truncated = actual_results.get("truncated", False)

    output = [
        f"**📄 Web Browse Result (Full Page)**\n\n**URL:** {url}\n\n**Source Type:** {source}\n"
    ]
    if truncated:
        output.append("**Note:** Content was truncated.\n")
    if message:
        output.append(f"{message}\n")

    output.append("---\n")
    # Use Markdown code block for potentially long text content
    output.append(
        "```text\n" + (content if content else "(No content extracted)") + "\n```"
    )
    return "\n".join(output)


def format_web_browse_query_results(results_data: Optional[Dict[str, Any]]) -> str:
    """Formats focused web browse results (retrieved snippets)."""
    # Use 'result' key which now contains the actual tool output
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
            "\n---\n_(No relevant snippets found for the query in the archived content.)_"
        )
        return "\n".join(output)

    output.append("\n**Retrieved Snippets:**\n")
    for i, snip in enumerate(snippets):
        content = snip.get("content", "N/A")
        dist = snip.get("distance", "N/A")
        idx = snip.get("chunk_index", "N/A")

        safe_content = html.escape(content)  # Escape content

        summary_line = f"**{i+1}. Snippet (Index: {idx}, Distance: {dist})**"
        # Preview first part of the content in summary
        content_preview = safe_content[:120].replace("\n", " ") + (
            "..." if len(safe_content) > 120 else ""
        )

        details_block = f"""
<details>
  <summary>{summary_line}: {content_preview}</summary>
  <pre style="background-color: #f0f0f0; border: 1px solid #ccc; padding: 5px; margin-top: 5px; white-space: pre-wrap; word-wrap: break-word;">{safe_content}</pre>
</details>
"""
        output.append(details_block)

    return "\n".join(output)


def format_other_tool_results(results_data: Optional[Dict[str, Any]]) -> str:
    """Formats results from other tools (like file, status) or errors as JSON/Text."""
    if not results_data or not isinstance(results_data, dict):
        return "(Invalid tool results data)"

    # Top-level info
    tool_name = results_data.get("tool_name", "Unknown Tool")
    action = results_data.get("action", "Unknown Action")
    status = results_data.get("status", "Unknown Status")
    error = results_data.get("error")  # Check for top-level error

    # Actual tool output is nested under 'result' key now
    actual_results = results_data.get("result", {})

    # Handle cases where the tool failed completely (error at top level)
    if status == "failed" and error:
        return f"**🛠️ Tool Execution Failed**\n\n**Tool:** {tool_name}\n**Action:** {action}\n**Status:** {status}\n\n**Error:** {error}"

    # Handle potential errors reported *within* the 'result' payload
    result_error = (
        actual_results.get("error") if isinstance(actual_results, dict) else None
    )

    # Special handling for 'status' tool report
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
        # Use Markdown code block for the report
        return f"**📊 Status Report**\n\n**Status:** {status}\n\n{message}\n\n---\n```markdown\n{report_content}\n```"

    # General handling
    output = [f"**🛠️ Tool Result: {tool_name} ({action})**\n\n**Status:** {status}\n"]
    if result_error:  # Display error from within the result payload
        output.append(f"\n**Error:** {result_error}\n")

    # Display the rest of the 'result' content prettily
    # Exclude known keys already displayed or handled (like error within result)
    content_to_display = {}
    message = None # Re-initialize message here for scope
    if isinstance(actual_results, dict):
        message = actual_results.get("message")  # Get message from result dict
        if message:
            output.append(f"\n{message}\n")

        content_to_display = {
            k: v
            for k, v in actual_results.items()
            if k
            not in [
                "status",
                "action",
                "error",
                "message",
                "report_content",
            ]  # Already handled
        }

    if content_to_display:
        try:
            # Pretty print the result content as JSON
            result_json = json.dumps(content_to_display, indent=2, ensure_ascii=False)
            output.append("\n**Details:**\n```json\n" + result_json + "\n```")
        except Exception:
            # Fallback if JSON fails
            output.append("\n**Details:**\n" + str(content_to_display))
    elif not result_error and not isinstance(actual_results, dict):
        # If actual_results wasn't a dict and no error, display its string representation
        output.append("\n**Result:**\n```text\n" + str(actual_results) + "\n```")

    return "\n".join(output)


# --- Functions for Monitor Tab ---
def start_agent_processing():
    if agent_instance:
        log.info("UI: Start/Resume Agent")
        agent_instance.start_autonomous_loop()
        state = agent_instance.get_ui_update_state()
        return f"Agent Status: {state.get('status', 'running')} @ {state.get('timestamp', 'N/A')}"
    else:
        log.error("UI failed: Agent not initialized.")
        return "ERROR: Agent not initialized."


def pause_agent_processing():
    if agent_instance:
        log.info("UI: Pause Agent")
        agent_instance.pause_autonomous_loop()
        time.sleep(0.2) # Give time for status to potentially update internally
        state = agent_instance.get_ui_update_state()
        return f"Agent Status: {state.get('status', 'paused')} @ {state.get('timestamp', 'N/A')}"
    else:
        log.error("UI failed: Agent not initialized.")
        return "ERROR: Agent not initialized."


def suggest_task_change():
    feedback = "Suggestion ignored: Agent not initialized."
    if agent_instance:
        log.info("UI: User clicked 'Suggest Task Change'")
        result = agent_instance.handle_user_suggestion_move_on()
        feedback = result
    else:
        log.error("UI suggest_task_change failed: Agent not initialized.")
    return feedback


# --- MODIFIED: Update function for Monitor Tab ---
def update_monitor_ui() -> Tuple[str, str, str, str, str, str, str, str, str, str]:
    """Fetches the latest agent UI state and console log for display."""
    # Initialize with error/default values (10 fields now)
    current_task_display = "(Error)"
    current_step_display = "(Error)"
    current_plan_display = "(Error)"
    dependent_tasks_display = "Dependent Tasks: (Error)"
    thinking_display = "(Error)"
    # <<< MODIFIED >>> Use global buffer for console log
    console_log_output = "".join(console_log_buffer) or "(Console Log Empty)"
    memory_display = "(Error)"
    tool_results_display = "(Error)"
    status_bar_text = "Error"
    final_answer_display = "(None)"

    if not agent_instance:
        status_bar_text = "Error: Agent not initialized."
        current_task_display = "Not Initialized"
        current_step_display = "Not Initialized"
        current_plan_display = "Not Initialized"
        dependent_tasks_display = "Dependent Tasks: (Not Initialized)"
        thinking_display = "Not Initialized"
        console_log_output = "Agent Not Initialized"  # <<< MODIFIED >>>
        memory_display = "Not Initialized"
        tool_results_display = "Not Initialized"
        final_answer_display = "Not Initialized"
        return (
            current_task_display,
            current_step_display,
            current_plan_display,
            dependent_tasks_display,
            thinking_display,
            console_log_output,  # <<< MODIFIED >>> Return console log
            memory_display,
            tool_results_display,
            status_bar_text,
            final_answer_display,
        )

    try:
        # Get latest agent state (doesn't include console log directly)
        ui_state = agent_instance.get_ui_update_state()

        # Core Task Info
        current_task_id = ui_state.get("current_task_id", "None")
        task_status = ui_state.get("status", "unknown")
        current_task_desc = ui_state.get("current_task_desc", "N/A")
        current_task_display = (
            f"ID: {current_task_id} ({task_status})\n\nDescription: {current_task_desc}"
        )

        # Current Step & Plan
        current_step_display = f"{ui_state.get('current_step_desc', 'N/A')}"
        current_plan_display = ui_state.get("current_plan", "(No plan active)")

        # Dependent Tasks
        deps = ui_state.get("dependent_tasks", [])
        if deps:
            dep_lines = [
                f"- {d['id'][:8]}...: {d['description'][:60]}..." for d in deps
            ]
            dependent_tasks_display = "Dependent Tasks:\n" + "\n".join(dep_lines)
        else:
            dependent_tasks_display = "Dependent Tasks: (None)"

        # Thinking Process
        thinking_display = ui_state.get("thinking", "(No thinking recorded)")

        # Step Log Output is now handled by the console buffer

        # Recent Memories
        recent_memories = ui_state.get("recent_memories", [])
        memory_display = format_memories_for_display(
            recent_memories, context_label="Monitor"
        )

        # Tool Results Display Formatting
        last_action = ui_state.get("last_action_type")
        last_results = ui_state.get("last_tool_results")

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
                    tool_results_display = format_web_browse_query_results(last_results)
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
            tool_results_display = "(Provided Final Answer)"
        elif last_action == "error":
            # Use console log for detailed errors now
            tool_results_display = "(Agent action resulted in error - see Console Log)"
        else:
            tool_results_display = "(No recent tool execution or action)"

        # Status Bar
        status_bar_text = (
            f"Agent Status: {task_status} @ {ui_state.get('timestamp', 'N/A')}"
        )

        # Final Answer
        final_answer = ui_state.get("final_answer")
        final_answer_display = (
            final_answer if final_answer else "(No recent final answer)"
        )

    except Exception as e:
        log.exception("Error in update_monitor_ui")
        error_msg = f"ERROR updating UI:\n{traceback.format_exc()}"
        console_log_output = error_msg  # <<< MODIFIED >>> Show error in console log box
        status_bar_text = "Error"
        current_task_display = "Error"
        current_step_display = "Error"
        current_plan_display = "Error"
        dependent_tasks_display = "Dependent Tasks: (Error)"
        thinking_display = "Error"
        memory_display = "Error"
        tool_results_display = "Error"
        final_answer_display = "Error"

    # <<< MODIFIED >>> Return console_log_output instead of step_log_output
    return (
        current_task_display,
        current_step_display,
        current_plan_display,
        dependent_tasks_display,
        thinking_display,
        console_log_output,  # Return the current console log buffer content
        memory_display,
        tool_results_display,
        status_bar_text,
        final_answer_display,
    )


# --- Functions for Step History Navigation ---
def format_step_details_for_display(
    step_data: Optional[Dict],
) -> Tuple[str, str, str, str, str]:
    if not step_data or not isinstance(step_data, dict):
        return "(No Step Data)", "(N/A)", "(N/A)", "(N/A)", "(N/A)"

    step_num = step_data.get("step", "N/A")
    task_id = step_data.get("task_id", "N/A")
    timestamp = step_data.get("timestamp", "N/A")
    rel_time = format_relative_time(timestamp)

    title = f"**Step {step_num}** (Task: {task_id[:8]}...) - {rel_time}"
    objective = step_data.get("step_objective", "(Objective not recorded)")
    thinking = step_data.get("thinking", "(No thinking recorded)")
    log_snippet = step_data.get(
        "log_snippet", "(No log snippet)"
    )  # Still keep this for historical context

    action_type = step_data.get("action_type")
    action_params = step_data.get("action_params")
    result_status = step_data.get("result_status")
    result_summary = step_data.get("result_summary", "(No result summary)")

    results_display = f"**Action:** {action_type}\n\n"
    if action_params:
        try:
            results_display += (
                f"**Params:**\n```json\n{json.dumps(action_params, indent=2)}\n```\n"
            )
        except:
            results_display += f"**Params:** {action_params}\n\n"
    results_display += f"**Result Status:** `{result_status}`\n\n"
    results_display += (
        f"**Result/Answer Summary:**\n```markdown\n{result_summary}\n```\n"
    )

    # Return title, objective, thinking, log, result summary
    return title, objective, thinking, log_snippet, results_display


def view_step_relative(
    current_index_str: str, step_delta: int, history_data: List[Dict]
) -> Tuple[str, str, str, str, str, str]:
    try:
        current_index = int(current_index_str)
    except (ValueError, TypeError):
        current_index = 0

    if not history_data:
        title, objective, thinking, log_s, result_s = format_step_details_for_display(
            None
        )
        return str(0), title, objective, thinking, log_s, result_s

    new_index = current_index + step_delta
    new_index = max(0, min(len(history_data) - 1, new_index))
    selected_step_data = history_data[new_index]
    title, objective, thinking, log_s, result_s = format_step_details_for_display(
        selected_step_data
    )

    return str(new_index), title, objective, thinking, log_s, result_s


def view_latest_step(
    history_data: List[Dict],
) -> Tuple[str, str, str, str, str, str]:
    if not history_data:
        title, objective, thinking, log_s, result_s = format_step_details_for_display(
            None
        )
        return str(0), title, objective, thinking, log_s, result_s

    latest_index = len(history_data) - 1
    selected_step_data = history_data[latest_index]
    title, objective, thinking, log_s, result_s = format_step_details_for_display(
        selected_step_data
    )

    return str(latest_index), title, objective, thinking, log_s, result_s


# --- Functions for Chat Tab ---
def _should_generate_task(user_msg: str, assistant_response: str) -> bool:
    if not agent_instance:
        return False
    log.info("Evaluating if task gen warranted...")
    eval_model = os.environ.get("OLLAMA_EVAL_MODEL", config.OLLAMA_CHAT_MODEL)
    prompt = f"""Analyze chat. Does user OR assistant response imply need for complex background task (e.g., research, multi-step analysis)? Answer ONLY "YES" or "NO".\n\nUser: {user_msg}\nAssistant: {assistant_response}"""
    response = call_ollama_api(
        prompt=prompt, model=eval_model, base_url=config.OLLAMA_BASE_URL, timeout=30
    )
    decision = response and "yes" in response.strip().lower()
    log.info(f"Task eval: {'YES' if decision else 'NO'} (LLM: '{response}')")
    return decision


def prioritize_generated_task(last_generated_task_id: Optional[str]):
    feedback = "Prioritization failed: Agent not initialized."
    if agent_instance:
        if last_generated_task_id:
            log.info(
                f"UI: User requested prioritization for task {last_generated_task_id}"
            )
            success = agent_instance.task_queue.set_priority(last_generated_task_id, 9) # Set high priority
            if success:
                task = agent_instance.task_queue.get_task(last_generated_task_id)
                feedback = (
                    f"✅ Task '{task.description[:30]}...' priority set to 9."
                    if task
                    else f"✅ Task {last_generated_task_id} priority set to 9 (task details not found)."
                )
            else:
                feedback = f"⚠️ Failed to set priority for task {last_generated_task_id} (may already be completed/failed or ID invalid)."
        else:
            feedback = "⚠️ Prioritization failed: No task ID provided (was a task generated this session?)."
    return feedback


# --- NEW: Function to create a priority task ---
def create_priority_task(message_text: str):
    """Adds the message text as a new task with priority 1."""
    feedback = "Create Task failed: Agent not initialized."
    if agent_instance:
        if message_text and message_text.strip():
            log.info(f"UI: Creating priority task: '{message_text[:50]}...'")
            # Use priority 9 for highest user priority
            new_task = Task(description=message_text.strip(), priority=9)
            task_id = agent_instance.task_queue.add_task(new_task)
            if task_id:
                feedback = f"✅ Priority Task Created (ID: {task_id[:8]}...): '{new_task.description[:60]}...'"
                log.info(f"Priority task {task_id} added.")
            else:
                feedback = "❌ Error adding task (likely duplicate description or internal error)."
                log.error("Failed to add priority task via UI.")
        else:
            feedback = "⚠️ Create Task failed: No message provided in the input box."
    else:
        log.error("UI create_priority_task failed: Agent not initialized.")
    return feedback

# --- REMOVED: inject_chat_info function ---

# --- MODIFIED: chat_response (remains mostly the same logic, calls _should_generate_task etc) ---
def chat_response(
    message: str, history: List[Dict[str, str]]
) -> Tuple[List[Dict[str, str]], str, str, str, Optional[str]]:
    memory_display_text = "Processing..."
    task_display_text = "(No task generated this turn)"
    last_gen_id = None
    if not message:
        return history, "No input provided.", task_display_text, "", last_gen_id
    if not agent_instance:
        history.append({"role": "user", "content": message})
        history.append(
            {"role": "assistant", "content": "**ERROR:** Backend agent failed."}
        )
        return history, "Error: Backend failed.", task_display_text, "", last_gen_id
    log.info(f"User message: '{message}'")
    history.append({"role": "user", "content": message})
    try:
        agent_is_running = agent_instance._is_running.is_set()
        agent_state = agent_instance.get_ui_update_state()
        agent_identity = agent_instance.identity_statement
        agent_task_id = agent_state.get("current_task_id")
        agent_task_desc = agent_state.get("current_task_desc", "N/A")
        agent_step_desc = agent_state.get(
            "current_step_desc", "N/A"
        )  # Get current step
        agent_status = agent_state.get("status", "unknown")
        agent_activity_context = "Currently idle."
        if agent_task_id:
            agent_task_for_context = agent_instance.task_queue.get_task(agent_task_id)
            agent_task_desc = (
                agent_task_for_context.description
                if agent_task_for_context
                else agent_task_desc + " (Missing?)"
            )
        if agent_status == "running" and agent_task_id:
            agent_activity_context = f"Working on Task {agent_task_id}: '{agent_task_desc}'. Status: {agent_status}. Current Step: '{agent_step_desc}'."
        elif agent_status == "planning" and agent_task_id:
            agent_activity_context = f"Planning Task {agent_task_id}: '{agent_task_desc}'. Status: {agent_status}."
        elif agent_status == "paused":
            agent_activity_context = (
                f"Paused. Active task {agent_task_id}: '{agent_task_desc}' (Step: {agent_step_desc})."
                if agent_task_id
                else "Paused and idle."
            )
        elif agent_status in ["shutdown", "critical_error"]:
            agent_activity_context = f"Agent state: '{agent_status}'."
        log.debug(f"Agent activity context: {agent_activity_context}")
        history_context_list = [
            f"{t.get('role','?').capitalize()}: {t.get('content','')}"
            for t in history[-4:-1]
        ]
        history_context_str = "\n".join(history_context_list)
        memory_query = f"Chat context for: '{message}'\nHistory:\n{history_context_str}\nAgent: {agent_activity_context}\nIdentity: {agent_identity}"
        relevant_memories, _ = agent_instance.memory.retrieve_and_rerank_memories(
            query=memory_query,
            task_description="Responding in chat, considering identity/activity.",
            context=f"{history_context_str}\nActivity: {agent_activity_context}\nIdentity: {agent_identity}",
            identity_statement=agent_instance.identity_statement,
            n_results=config.MEMORY_COUNT_CHAT_RESPONSE * 2,
            n_final=config.MEMORY_COUNT_CHAT_RESPONSE,
        )
        memory_display_text = format_memories_for_display(
            relevant_memories, context_label="Chat"
        )
        log.info(f"Retrieved {len(relevant_memories)} memories for chat.")
        system_prompt = f"""You are helpful AI assistant ({agent_identity}). Answer user conversationally. Consider identity, chat history, memories, agent's background activity. Be aware of your capabilities and limitations."""
        memories_for_prompt_list = []
        snippet_length = 250
        for mem in relevant_memories:
            relative_time = format_relative_time(
                mem.get("metadata", {}).get("timestamp")
            )
            mem_type = mem.get("metadata", {}).get("type", "N/A")
            content_snippet = mem.get("content", "")
            if len(content_snippet) > snippet_length:
                content_snippet = content_snippet[:snippet_length].strip() + "..."
            else:
                content_snippet = content_snippet.strip()
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
        log.debug(f"Approximate prompt length for chat response: {len(prompt)} chars")
        log.info(f"Asking {config.OLLAMA_CHAT_MODEL} for chat response...")
        response_text = call_ollama_api(
            prompt=prompt,
            model=config.OLLAMA_CHAT_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            timeout=config.OLLAMA_TIMEOUT,
        )
        if not response_text:
            response_text = "Sorry, error generating response."
            log.error(
                "LLM call failed for chat response. Prompt might be too long or Ollama error occurred."
            )
        history.append({"role": "assistant", "content": response_text})
        if agent_is_running:
            log.info("Agent running, adding chat to memory...")
            agent_instance.memory.add_memory(
                content=f"User query: {message}", metadata={"type": "chat_user_query"}
            )
            agent_instance.memory.add_memory(
                content=f"Assistant response: {response_text}",
                metadata={
                    "type": "chat_assistant_response",
                    "agent_activity_at_time": agent_activity_context,
                    "agent_identity_at_time": agent_identity,
                },
            )
        else:
            log.info("Agent paused, skipping adding chat history to memory.")
        if _should_generate_task(message, response_text):
            log.info("Interaction warrants task generation...")
            first_task_generated = agent_instance.generate_new_tasks(
                max_new_tasks=1,
                last_user_message=message,
                last_assistant_response=response_text,
                trigger_context="chat",
            )
            if first_task_generated:
                newly_added_task_obj = None
                desc_match = first_task_generated.strip().lower()
                now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
                cutoff_time = (
                    datetime.datetime.now(datetime.timezone.utc)
                    - datetime.timedelta(seconds=10)
                ).isoformat()
                potential_matches = [
                    t
                    for t in agent_instance.task_queue.tasks.values()
                    if t.description.strip().lower() == desc_match
                    and t.created_at >= cutoff_time
                ]
                if potential_matches:
                    newly_added_task_obj = sorted(
                        potential_matches, key=lambda t: t.created_at, reverse=True
                    )[0]
                if newly_added_task_obj:
                    last_gen_id = newly_added_task_obj.id
                    task_display_text = f'✅ Task Generated (ID: {last_gen_id}):\n"{first_task_generated}"'
                    log.info(
                        f"Task generated (ID: {last_gen_id}): {first_task_generated[:60]}..."
                    )
                    if response_text != "Sorry, error generating response.":
                        notification = f"\n\n*(Okay, based on our chat, I've created task {last_gen_id}: \"{first_task_generated}\". I'll work on it when possible. You can prioritize it if needed.)*"
                        history[-1]["content"] += notification
                    agent_instance._save_qlora_datapoint(
                        source_type="chat_task_generation",
                        instruction="User interaction led to this task. Respond confirming task creation.",
                        input_context=f"Identity: {agent_identity}\nUser: {message}\nActivity: {agent_activity_context}",
                        output=f"{response_text}{notification if response_text != 'Sorry, error generating response.' else ''}",
                    )
                else:
                    task_display_text = (
                        f'✅ Task Generated (ID unknown):\n"{first_task_generated}"'
                    )
                    log.warning(
                        f"Task generated but could not find its ID immediately: {first_task_generated}"
                    )
            else:
                task_display_text = "(Eval suggested task, none generated/added)"
                log.info("Task gen warranted but no task added.")
        else:
            log.info("Task generation not warranted.")
            task_display_text = "(No new task warranted)"
        return history, memory_display_text, task_display_text, "", last_gen_id
    except Exception as e:
        log.exception(f"Error during chat processing: {e}")
        error_message = f"Internal error processing message: {e}"
        history.append({"role": "assistant", "content": error_message})
        return (
            history,
            f"Error:\n```\n{traceback.format_exc()}\n```",
            task_display_text,
            "",
            None,
        )


# --- Functions for Agent State Tab ---
# --- MODIFIED: refresh_agent_state_display ---
def refresh_agent_state_display():
    log.info("State Tab: Refresh button clicked. Fetching latest state...")
    if not agent_instance:
        error_df = pd.DataFrame(columns=["Error"])
        # Match output structure: identity, 4 DFs, mem summary, dropdown, 2 memory DFs, + 4 raw data lists + details display
        return (
            "Agent not initialized.",
            error_df, error_df, error_df, error_df, # Task DFs
            "Error: Agent not initialized.", # Mem summary
            gr.Dropdown(choices=["Error"], value="Error"), # Dropdown
            error_df, error_df, # Memory DFs
            [], [], [], [], # Raw task data lists
            "(No details to show)", # Details display
        )
    try:
        state_data = agent_instance.get_agent_dashboard_state()

        def create_or_default_df(data, columns):
            # Ensure data is a list of dicts
            if not isinstance(data, list) or (data and not all(isinstance(item, dict) for item in data)):
                 log.warning(f"Invalid data type passed to create_or_default_df: {type(data)}. Expected list of dicts.")
                 return pd.DataFrame(columns=columns)

            if not data: return pd.DataFrame(columns=columns)
            df = pd.DataFrame(data)
            for col in columns:
                if col not in df.columns: df[col] = pd.NA
            return df.reindex(columns=columns, fill_value=pd.NA)

        # Define columns explicitly
        pending_cols = ["ID", "Priority", "Description", "Depends On", "Created"]
        inprogress_cols = ["ID", "Priority", "Description", "Status", "Created"]
        completed_cols = ["ID", "Description", "Completed At", "Result Snippet"]
        # Added Reattempts here
        failed_cols = ["ID", "Description", "Failed At", "Reason", "Reattempts"]
        memory_cols = ["Relative Time", "Timestamp", "Type", "Content Snippet", "ID"]

        # Get raw data lists
        pending_raw = state_data.get("pending_tasks", [])
        inprogress_raw = state_data.get("in_progress_tasks", []) # Already combined planning/in_progress
        completed_raw = state_data.get("completed_tasks", [])
        failed_raw = state_data.get("failed_tasks", [])

        # Create DataFrames
        pending_df = create_or_default_df(pending_raw, pending_cols)
        inprogress_df = create_or_default_df(inprogress_raw, inprogress_cols)
        completed_df = create_or_default_df(completed_raw, completed_cols)
        failed_df = create_or_default_df(failed_raw, failed_cols)

        # Dropdown data (no change needed here)
        completed_failed_tasks_data = state_data.get("completed_failed_tasks_data", [])
        task_id_choices_tuples = [(f"{t['ID']} - {t['Description'][:50]}...", t["ID"]) for t in completed_failed_tasks_data]
        dropdown_choices = [("Select Task ID...", None)] + task_id_choices_tuples
        initial_dropdown_value = None

        # General memories (no change needed here)
        general_memories = agent_instance.get_formatted_general_memories()
        for mem in general_memories: mem["Relative Time"] = format_relative_time(mem.get("Timestamp"))
        general_mem_df = create_or_default_df(general_memories, memory_cols)

        # Return includes the raw lists now, and an initial value for the details display
        return (
            state_data.get("identity_statement", "Error loading identity"),
            pending_df,
            inprogress_df,
            completed_df,
            failed_df,
            state_data.get("memory_summary", "Error loading summary"),
            gr.Dropdown(choices=dropdown_choices, value=initial_dropdown_value, label="Select Task ID (Completed/Failed)", interactive=True),
            pd.DataFrame(columns=memory_cols), # Empty task memory df initially
            general_mem_df,
            pending_raw, # Return raw data for state
            inprogress_raw,
            completed_raw,
            failed_raw,
            "(Select a row from a task table above to see details)", # Initial details text
        )
    except Exception as e:
        log.exception("Error refreshing agent state display")
        error_df = pd.DataFrame([{"Error": str(e)}])
        # Match output structure on error
        return (
            f"Error: {e}",
            error_df, error_df, error_df, error_df,
            f"Memory Summary Error: {e}",
            gr.Dropdown(choices=["Error"], value="Error"),
            error_df, error_df,
            [], [], [], [],
            f"Error loading state details: {e}",
        )


# --- MODIFIED: update_task_memory_display (No change needed) ---
def update_task_memory_display(selected_task_id: str):
    log.debug(f"State Tab: Task ID selected: {selected_task_id}")
    columns = ["Relative Time", "Timestamp", "Type", "Content Snippet", "ID"]
    if not agent_instance or not selected_task_id:
        return pd.DataFrame(columns=columns)
    try:
        memories = agent_instance.get_formatted_memories_for_task(selected_task_id)
        for mem in memories:
            mem["Relative Time"] = format_relative_time(mem.get("Timestamp"))
        mem_df = pd.DataFrame(memories)
        if mem_df.empty:
            return pd.DataFrame(columns=columns)
        else:
            # Sort by Timestamp ascending before returning
            mem_df["Timestamp"] = pd.to_datetime(mem_df["Timestamp"], errors="coerce")
            mem_df = mem_df.sort_values(by="Timestamp", ascending=True)
            # Ensure columns are correct before returning
            return mem_df.reindex(columns=columns, fill_value=pd.NA)
    except Exception as e:
        log.exception(f"Error fetching memories for task {selected_task_id}")
        return pd.DataFrame([{"Error": str(e)}])


# --- MODIFIED: update_general_memory_display (No change needed) ---
def update_general_memory_display():
    log.debug("State Tab: Refreshing general memories display data...")
    columns = ["Relative Time", "Timestamp", "Type", "Content Snippet", "ID"]
    if not agent_instance:
        return pd.DataFrame(columns=columns)
    try:
        memories = agent_instance.get_formatted_general_memories()
        for mem in memories:
            mem["Relative Time"] = format_relative_time(mem.get("Timestamp"))
        mem_df = pd.DataFrame(memories)
        if mem_df.empty:
            return pd.DataFrame(columns=columns)
        else:
            # Sort by Timestamp descending before returning
            mem_df["Timestamp"] = pd.to_datetime(mem_df["Timestamp"], errors="coerce")
            mem_df = mem_df.sort_values(by="Timestamp", ascending=False)
            return mem_df.reindex(columns=columns, fill_value=pd.NA)
    except Exception as e:
        log.exception("Error fetching general memories")
        return pd.DataFrame([{"Error": str(e)}])


# --- NEW: Handler for DataFrame row selection ---
def show_task_details(evt: gr.SelectData, task_list: list) -> str:
    """Formats the details of a selected task row for display."""
    if not task_list or not isinstance(task_list, list):
        return "(Internal Error: Invalid task list provided for details view)"
    try:
        row_index = evt.index[0]
        if 0 <= row_index < len(task_list):
            task_data = task_list[row_index]
            if isinstance(task_data, dict):
                # Format the dictionary into a readable Markdown string
                details = [f"### Task Details (Row {row_index+1})"]
                for key, value in task_data.items():
                    # Handle potential long strings like description or result/reason
                    display_value = str(value) if value is not None else "N/A" # Ensure it's a string

                    # Use code block for potentially multi-line fields or long text
                    if isinstance(value, str) and ('\n' in value or len(value) > 100):
                        details.append(f"**{key}:**\n```text\n{value}\n```")
                    else:
                         details.append(f"**{key}:** `{display_value}`") # Use inline code for single line values

                return "\n\n".join(details)
            else:
                return f"(Internal Error: Task data at index {row_index} is not a dictionary)"
        else:
            # This case might happen if the table is refreshed while a row is selected
            return "(Selection cleared or invalid. Please re-select a row.)"
    except IndexError:
         return "(Error: Could not get selected row index. Table might have changed.)"
    except Exception as e:
        log.exception(f"Error formatting task details: {e}")
        return f"(Error displaying task details: {e})"


# --- Gradio UI Definition ---
log.info("Defining Gradio UI...")
if agent_instance is None:
    log.critical("Agent failed init. Cannot launch UI.")
    with gr.Blocks() as demo:
        gr.Markdown("# Fatal Error\nAgent backend failed to initialize. Check logs.")
else:
    with gr.Blocks(theme=gr.themes.Glass(), title="Autonomous Agent Interface") as demo:
        gr.Markdown("# Autonomous Agent Control Center & Chat")

        step_history_state = gr.State(initial_step_history_data)
        step_history_index_state = gr.State(
            str(len(initial_step_history_data) - 1 if initial_step_history_data else 0)
        )

        # --- NEW: State variables for raw task data ---
        pending_tasks_data_state = gr.State([])
        inprogress_tasks_data_state = gr.State([])
        completed_tasks_data_state = gr.State([])
        failed_tasks_data_state = gr.State([])
        # --- End NEW State ---

        with gr.Tabs():
            with gr.TabItem("Agent Monitor"):
                with gr.Row():
                    start_resume_btn = gr.Button("Start / Resume", variant="primary")
                    pause_btn = gr.Button("Pause")
                    suggest_change_btn = gr.Button("Suggest Task Change")
                with gr.Accordion("Agent", open=True):
                    monitor_status_bar = gr.Textbox(
                        label="",
                        show_label=False,
                        value=initial_status_text,
                        interactive=False,
                        lines=1,
                    )
                    suggestion_feedback_box = gr.Textbox(
                        show_label=False,
                        value="No Suggestion Feedback",
                        interactive=False,
                        lines=1,
                    )
                with gr.Accordion("Task", open=True):
                    with gr.Row():
                        with gr.Column(scale=1):
                            monitor_current_task = gr.Textbox(
                                label="❤️ Task Status",
                                value=initial_task_status_text,
                                lines=4,
                                interactive=False,
                            )
                            monitor_dependent_tasks = gr.Textbox(
                                value=initial_dependent_tasks_text, show_label=False
                            )
                            monitor_plan = gr.Textbox(
                                label="📋 Execution Plan",
                                value=initial_plan_text,
                                lines=7,
                                interactive=False,
                            )
                            monitor_current_step = gr.Textbox(
                                value=initial_step_desc_text, show_label=False
                            )
                            monitor_thinking = gr.Textbox(
                                label="🤖 Agent Thinking (Current Step)",
                                value=initial_thinking_text,
                                lines=8,
                                interactive=False,
                            )
                            monitor_final_answer = gr.Textbox(
                                label="🏆 Last Final Answer",
                                lines=5,
                                interactive=False,
                                show_copy_button=True,
                            )
                        with gr.Column(scale=1):
                            # <<< MODIFIED: Log Box Definition >>>
                            monitor_log = gr.Textbox(
                                label="🪵 Console Log",  # Renamed label
                                value=initial_console_log_text,  # Use new initial value
                                max_lines=7,
                                autoscroll=True,
                                interactive=False,
                                show_copy_button=True,
                            )
                            monitor_tool_results = gr.Markdown(
                                value=initial_tool_results_text,
                                label="🛠️ Tool Output / Browse Content / Snippets",
                            )
                            monitor_memory = gr.Markdown(
                                value="🧠 Recent Memories (Monitor)\n\n(Agent Initializing)"
                            )
                with gr.Accordion("Step History", open=False):
                    # ... (Step history layout unchanged) ...
                    with gr.Row():
                        step_prev_btn = gr.Button("◀ Previous Step")
                        step_latest_btn = gr.Button("Latest Step", variant="secondary")
                        step_next_btn = gr.Button("Next Step ▶")
                    step_hist_title = gr.Markdown(
                        "**Step History** (Load latest to view)"
                    )
                    step_hist_objective = gr.Markdown(
                        label="Step Objective", value="(No history selected)"
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            step_hist_thinking = gr.Textbox(
                                label="Thinking",
                                lines=10,
                                interactive=False,
                                show_copy_button=True,
                            )
                            step_hist_log = gr.Textbox(
                                label="Log Snippet", lines=5, interactive=False
                            )  # Keep historical log snippet here
                        with gr.Column(scale=1):
                            step_hist_result = gr.Markdown(
                                label="Action/Result Summary",
                                value="(No history selected)",
                            )

                # --- Button Clicks (unchanged) ---
                start_resume_btn.click(
                    fn=start_agent_processing, inputs=None, outputs=monitor_status_bar
                )
                pause_btn.click(
                    fn=pause_agent_processing, inputs=None, outputs=monitor_status_bar
                )
                suggest_change_btn.click(
                    fn=suggest_task_change, inputs=None, outputs=suggestion_feedback_box
                )

                # --- Step History Clicks (unchanged) ---
                step_hist_view_outputs = [
                    step_history_index_state,
                    step_hist_title,
                    step_hist_objective,
                    step_hist_thinking,
                    step_hist_log,  # Keep outputting historical log snippet here
                    step_hist_result,
                ]
                step_prev_btn.click(
                    fn=view_step_relative,
                    inputs=[step_history_index_state, gr.State(-1), step_history_state],
                    outputs=step_hist_view_outputs,
                )
                step_next_btn.click(
                    fn=view_step_relative,
                    inputs=[step_history_index_state, gr.State(1), step_history_state],
                    outputs=step_hist_view_outputs,
                )
                step_latest_btn.click(
                    fn=view_latest_step,
                    inputs=[step_history_state],
                    outputs=step_hist_view_outputs,
                )

                # --- Timer Setup ---
                timer = gr.Timer(config.UI_UPDATE_INTERVAL)

                # --- MODIFIED: Timer Update Function Wrapper ---
                def update_monitor_and_history():
                    """Handles periodic UI updates for the Monitor tab, including the console log."""
                    global last_monitor_state
                    num_monitor_outputs = 10  # Keep 10 outputs
                    # Initialize with placeholder values matching the 10 outputs
                    monitor_updates_to_return = (
                        "(Initializing)",
                    ) * num_monitor_outputs
                    current_history_data = []

                    # <<< NEW: Get current console log content >>>
                    current_console_log = "".join(console_log_buffer)

                    if agent_instance:
                        try:
                            agent_ui_state = agent_instance.get_ui_update_state()
                            current_history_data = agent_ui_state.get(
                                "step_history", []
                            )

                            # Always get fresh monitor state if agent running or no previous state
                            if (
                                agent_instance._is_running.is_set()
                                or last_monitor_state is None
                            ):
                                log.debug(
                                    "Agent running or first update, getting fresh monitor state..."
                                )
                                current_monitor_state_tuple = update_monitor_ui()
                            else:
                                # Agent paused/stopped, use cached state but update status bar and log
                                log.debug(
                                    "Agent paused/stopped, using cached monitor state + live console log."
                                )
                                # Use cached tuple IF it exists and has the right size
                                if last_monitor_state is not None and len(last_monitor_state) == num_monitor_outputs:
                                    current_monitor_state_tuple = last_monitor_state
                                else:
                                    log.warning("Cached monitor state missing or invalid size, fetching fresh.")
                                    current_monitor_state_tuple = update_monitor_ui() # Fallback

                            # Ensure the tuple has the correct number of elements *before* modifying
                            if (
                                len(current_monitor_state_tuple)
                                != num_monitor_outputs
                            ):
                                raise ValueError(
                                    f"Monitor update state tuple returned {len(current_monitor_state_tuple)} elements, expected {num_monitor_outputs}"
                                )

                            last_monitor_state = current_monitor_state_tuple # Cache the latest state

                            # Now modify the tuple *if* paused
                            if not agent_instance._is_running.is_set():
                                temp_list = list(last_monitor_state)
                                # Index 8 is status_bar_text (0-based)
                                temp_list[8] = (
                                    f"Agent Status: {agent_ui_state.get('status','paused')} @ {agent_ui_state.get('timestamp', 'N/A')}"
                                )
                                # Index 5 is console_log_output (0-based)
                                temp_list[5] = (
                                    current_console_log
                                    or "(Console Log Empty - Paused)"
                                )
                                monitor_updates_to_return = tuple(temp_list)
                            else:
                                # If running, use the state as is (log should be correct from update_monitor_ui)
                                monitor_updates_to_return = last_monitor_state


                        except Exception as e:
                            log.exception("Error during monitor update wrapper")
                            # Construct error tuple matching expected output size
                            error_tuple = list(["Error: Update Failed"] * num_monitor_outputs)
                            error_tuple[5] = (
                                f"ERROR updating UI:\n{traceback.format_exc()}"  # Put error in console log
                            )
                            monitor_updates_to_return = tuple(error_tuple)
                            current_history_data = [{"error": f"Update failed: {e}"}]
                    else:
                        # Construct offline tuple matching expected output size
                        offline_tuple = list(["Error: Agent Offline"] * num_monitor_outputs)
                        offline_tuple[5] = "Agent Offline"  # Put status in console log
                        monitor_updates_to_return = tuple(offline_tuple)
                        current_history_data = [{"error": "Agent Offline"}]

                    # Validate final tuple structure before returning
                    if (
                        not isinstance(monitor_updates_to_return, tuple)
                        or len(monitor_updates_to_return) != num_monitor_outputs
                    ):
                        log.error(
                            f"Monitor update tuple structure mismatch. Expected {num_monitor_outputs}. Got {len(monitor_updates_to_return) if isinstance(monitor_updates_to_return, tuple) else type(monitor_updates_to_return)}"
                        )
                        # Construct mismatch tuple matching expected output size
                        mismatch_tuple = list([
                            "Error: Struct Mismatch"
                        ] * num_monitor_outputs)
                        mismatch_tuple[5] = (
                            "UI Update Error: Structure Mismatch"  # Put error in console log
                        )
                        monitor_updates_to_return = tuple(mismatch_tuple)

                    # Return the monitor updates and the current history data
                    # Unpack the tuple for individual component outputs, plus the history state
                    return *monitor_updates_to_return, current_history_data

                # --- MODIFIED: Timer Outputs ---
                # Index 5 now corresponds to the monitor_log Textbox
                monitor_timer_outputs = [
                    monitor_current_task,  # 0
                    monitor_current_step,  # 1
                    monitor_plan,  # 2
                    monitor_dependent_tasks,  # 3
                    monitor_thinking,  # 4
                    monitor_log,  # 5 <<< Was step log snippet, now console log
                    monitor_memory,  # 6
                    monitor_tool_results,  # 7
                    monitor_status_bar,  # 8
                    monitor_final_answer,  # 9
                    step_history_state,  # Output for the history state variable
                ]
                timer.tick(
                    fn=update_monitor_and_history,
                    inputs=None,
                    outputs=monitor_timer_outputs,
                )

                # --- Initial History Load (unchanged) ---
                (
                    initial_hist_index_str,
                    initial_hist_title,
                    initial_hist_objective,
                    initial_hist_think,
                    initial_hist_log_snippet,  # Keep using historical log snippet here
                    initial_hist_result,
                ) = view_latest_step(initial_step_history_data)
                demo.load(
                    fn=lambda: (
                        initial_hist_index_str,
                        initial_hist_title,
                        initial_hist_objective,
                        initial_hist_think,
                        initial_hist_log_snippet,  # Load historical snippet
                        initial_hist_result,
                    ),
                    inputs=[],
                    outputs=[
                        step_history_index_state,
                        step_hist_title,
                        step_hist_objective,
                        step_hist_thinking,
                        step_hist_log,  # Update the historical log snippet box
                        step_hist_result,
                    ],
                )

            with gr.TabItem("Chat"):
                # --- MODIFIED: Chat layout ---
                gr.Markdown(
                    "Interact with the agent. It considers its identity, activity, and your input."
                )
                last_generated_task_id_state = gr.State(None)
                with gr.Row():
                    with gr.Column(scale=3):
                        chat_chatbot = gr.Chatbot(
                            label="Conversation",
                            height=400,
                            show_copy_button=True,
                            type="messages",
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
                            label="Chat Interaction Feedback",
                            value="",
                            interactive=False,
                            lines=1,
                        )
                        with gr.Row():
                            chat_msg_input = gr.Textbox(
                                label="Your Message / Task Description", # Updated label
                                placeholder="Type message for chat, or text for new priority task...", # Updated placeholder
                                lines=3,
                                scale=4,
                                container=False,
                            )
                            chat_send_button = gr.Button(
                                "Send Chat", variant="primary", scale=1 # Renamed button
                            )
                            # --- Renamed Button ---
                            create_task_btn = gr.Button("Create Priority Task", scale=1) # Was inject_info_btn
                    with gr.Column(scale=1):
                        chat_memory_panel = gr.Markdown(
                            value="Relevant Memories (Chat)\n...",
                            label="Memory Context",
                        )

                chat_outputs = [
                    chat_chatbot,
                    chat_memory_panel,
                    chat_task_panel,
                    chat_msg_input, # Output to clear input
                    last_generated_task_id_state,
                ]
                # --- Send button unchanged ---
                chat_send_button.click(
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
                # --- Prioritize button unchanged ---
                prioritize_task_btn.click(
                    fn=prioritize_generated_task,
                    inputs=[last_generated_task_id_state],
                    outputs=[chat_interaction_feedback],
                )
                # --- NEW: Connect create_priority_task ---
                create_task_btn.click(
                    fn=create_priority_task,
                    inputs=[chat_msg_input], # Use the same input box
                    outputs=[chat_interaction_feedback], # Output feedback
                )


            with gr.TabItem("Agent State"):
                # --- MODIFIED: Agent State layout ---
                gr.Markdown(
                    "View the agent's current identity, task queues, and memory. **Use button to load/refresh data.** Click rows in task tables to see full details."
                )
                with gr.Row():
                    state_identity = gr.Textbox(
                        label="Agent Identity Statement",
                        lines=3,
                        interactive=False,
                        value="(Press Load State)",
                    )
                    load_state_button = gr.Button("Load Agent State", variant="primary")
                with gr.Accordion("Task Status", open=True):
                    with gr.Column():
                        gr.Markdown("#### Pending Tasks (Highest Priority First)")
                        with gr.Row():
                            # Set interactive=False, rely on select event
                            state_pending_tasks = gr.DataFrame(
                                headers=[
                                    "ID",
                                    "Priority",
                                    "Description",
                                    "Depends On",
                                    "Created",
                                ],
                                interactive=False, # Changed
                                wrap=True
                            )
                        gr.Markdown("#### In Progress / Planning Tasks")
                        with gr.Row():
                            state_inprogress_tasks = gr.DataFrame(
                                headers=[
                                    "ID",
                                    "Priority",
                                    "Description",
                                    "Status",
                                    "Created",
                                ],
                                interactive=False, # Changed
                                wrap=True
                            )
                        gr.Markdown("#### Completed Tasks (Most Recent First)")
                        with gr.Row():
                            state_completed_tasks = gr.DataFrame(
                                headers=[
                                    "ID",
                                    "Description",
                                    "Completed At",
                                    "Result Snippet",
                                ],
                                interactive=False, # Changed
                                wrap=True
                            )
                        gr.Markdown("#### Failed Tasks (Most Recent First)")
                        with gr.Row():
                            state_failed_tasks = gr.DataFrame(
                                headers=["ID", "Description", "Failed At", "Reason", "Reattempts"], # Added Reattempts header
                                interactive=False, # Changed
                                wrap=True
                            )

                        # --- NEW: Details Display Area ---
                        gr.Markdown("--- \n ### Selected Task Details:")
                        state_details_display = gr.Markdown("(Select a row from a task table above to see details)")
                        # --- End NEW ---

                with gr.Accordion("Memory Explorer", open=True):
                    state_memory_summary = gr.Markdown(
                        "Memory Summary\n(Press Load State)"
                    )
                    with gr.Column():
                        gr.Markdown("##### Task-Specific Memories")
                        with gr.Row(scale=1):
                            state_task_memory_select = gr.Dropdown(
                                label="Select Task ID (Completed/Failed)",
                                choices=[("Select Task ID...", None)],
                                value=None,
                                interactive=True,
                            )
                        with gr.Row(scale=1):
                            state_task_memory_display = gr.DataFrame(
                                headers=[
                                    "Relative Time",
                                    "Timestamp",
                                    "Type",
                                    "Content Snippet",
                                    "ID",
                                ],
                                interactive=False,
                                wrap=True
                            )
                        gr.Markdown("##### General Memories (Recent)")
                        with gr.Row(scale=1):
                            state_general_memory_display = gr.DataFrame(
                                headers=[
                                    "Relative Time",
                                    "Timestamp",
                                    "Type",
                                    "Content Snippet",
                                    "ID",
                                ],
                                interactive=False,
                                wrap=True
                            )
                        with gr.Row(scale=1):
                            refresh_general_mem_btn = gr.Button(
                                "Refresh General Memories"
                            )

                # --- MODIFIED: Agent State Tab Outputs ---
                # Now includes the raw data states and the details display
                state_tab_outputs = [
                    state_identity,
                    state_pending_tasks,
                    state_inprogress_tasks,
                    state_completed_tasks,
                    state_failed_tasks,
                    state_memory_summary,
                    state_task_memory_select,
                    state_task_memory_display,
                    state_general_memory_display,
                    # Raw data states
                    pending_tasks_data_state,
                    inprogress_tasks_data_state,
                    completed_tasks_data_state,
                    failed_tasks_data_state,
                    # Details display
                    state_details_display,
                ]
                load_state_button.click(
                    fn=refresh_agent_state_display,
                    inputs=None,
                    outputs=state_tab_outputs,
                )

                # Task memory dropdown click (unchanged)
                state_task_memory_select.change(
                    fn=update_task_memory_display,
                    inputs=[state_task_memory_select],
                    outputs=[state_task_memory_display],
                )
                # General memory refresh click (unchanged)
                refresh_general_mem_btn.click(
                    fn=update_general_memory_display,
                    inputs=None,
                    outputs=[state_general_memory_display],
                )

                # --- NEW: DataFrame Select Events ---
                # Connect .select event of each DF to the handler
                state_pending_tasks.select(
                    fn=show_task_details,
                    inputs=[pending_tasks_data_state], # Pass state holding raw list
                    outputs=[state_details_display]
                    )
                state_inprogress_tasks.select(
                    fn=show_task_details,
                    inputs=[inprogress_tasks_data_state],
                    outputs=[state_details_display]
                    )
                state_completed_tasks.select(
                    fn=show_task_details,
                    inputs=[completed_tasks_data_state],
                    outputs=[state_details_display]
                    )
                state_failed_tasks.select(
                    fn=show_task_details,
                    inputs=[failed_tasks_data_state],
                    outputs=[state_details_display]
                    )
                # --- End NEW ---


# --- Launch the App & Background Thread ---
if __name__ == "__main__":
    # <<< NEW: Redirect stdout/stderr and configure logging handler >>>
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_stream_handler = ConsoleLogHandler(console_log_buffer)

    # Redirect stdout and stderr
    sys.stdout = log_stream_handler
    sys.stderr = log_stream_handler

    # Configure the root logger to also send output to our handler
    root_logger = logging.getLogger()
    # Remove existing StreamHandlers to avoid double printing to original console if basicConfig added one
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream in [
            original_stdout,
            original_stderr,
            sys.__stdout__, # Catch default handler if it exists
            sys.__stderr__
        ]:
            log.debug(f"Removing existing StreamHandler: {handler}")
            root_logger.removeHandler(handler)

    ui_log_handler = logging.StreamHandler(log_stream_handler)
    log_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ui_log_handler.setFormatter(log_formatter)
    root_logger.addHandler(ui_log_handler)
    root_logger.setLevel(config.LOG_LEVEL)  # Ensure root logger level is set

    print("--- Autonomous Agent App Start ---")  # This should now go to the buffer
    log.info("Console output redirected to UI buffer.")
    # --- End NEW Redirection ---

    try:
        os.makedirs(config.SUMMARY_FOLDER, exist_ok=True)
        log.info(f"Summary directory: {config.SUMMARY_FOLDER}")
    except Exception as e:
        log.error(f"Could not create summary dir: {e}")

    log.info("Launching Gradio App Interface...")
    if agent_instance:
        log.info("Agent background processing thread started and paused.")
        try:
            log.info("UI defined. Launching Gradio server...")
            demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
        except Exception as e:
            log.critical(f"Gradio launch failed: {e}", exc_info=True)
            # Restore stdout/stderr before exiting if possible
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            if agent_instance:
                agent_instance.shutdown()
            sys.exit("Gradio launch failed.")
    else:
        log.warning("Agent init failed. Launching minimal error UI.")
        try:
            with gr.Blocks() as error_demo:
                gr.Markdown("# Fatal Error\nAgent backend failed. Check logs.")
            error_demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
        except Exception as e:
            log.critical(f"Gradio error UI launch failed: {e}", exc_info=True)
            # Restore stdout/stderr before exiting if possible
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            sys.exit("Gradio launch failed.")

    log.info("Gradio App stopped. Requesting agent shutdown...")
    if agent_instance:
        agent_instance.shutdown()

    # Restore stdout/stderr after Gradio stops
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log.info("Shutdown complete.")
    print("\n--- Autonomous Agent App End ---")  # This goes to the actual console