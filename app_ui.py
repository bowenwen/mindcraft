# FILE: app_ui.py
# autonomous_agent/app_ui.py
import gradio as gr
import datetime
import json
import traceback
import time
import os
import sys
import threading
from typing import List, Tuple, Optional, Dict, Any, Deque
import pandas as pd
from collections import deque

# --- Project Imports ---
import config
from utils import call_ollama_api, format_relative_time
from memory import AgentMemory, setup_chromadb
from task_manager import TaskQueue
from data_structures import Task
from agent import AutonomousAgent
import chromadb

# --- Logging Setup ---
import logging

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="[%(asctime)s] [%(levelname)s][%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("AppUI")


# --- Global Variables / Setup ---
agent_instance: Optional[AutonomousAgent] = None
try:
    mem_collection = setup_chromadb()
    if mem_collection is None:
        raise RuntimeError("Database setup failed.")
    agent_instance = AutonomousAgent(memory_collection=mem_collection)
    log.info("App components initialized successfully.")
except Exception as e:
    log.critical(f"Fatal error during App component initialization: {e}", exc_info=True)
    print(
        f"\n\nFATAL ERROR: Could not initialize agent components: {e}\n",
        file=sys.stderr,
    )
    agent_instance = None

initial_status_text = "Error: Agent not initialized."
# --- MODIFIED: Added initial values for new UI fields ---
initial_thinking_text = "(Agent Initializing...)"
initial_tool_results_text = "(No tool results yet)"
initial_dependent_tasks_text = "Dependent Tasks: (None)"
initial_step_history_data: List[Dict] = []  # Start with empty history for UI

if agent_instance:
    log.info("Setting initial agent state to paused for UI...")
    agent_instance.start_autonomous_loop()
    agent_instance.pause_autonomous_loop()  # Agent now starts paused by default
    initial_state = agent_instance.get_ui_update_state()
    initial_status_text = f"Agent Status: {initial_state.get('status', 'paused')} @ {initial_state.get('timestamp', 'N/A')}"
    initial_thinking_text = initial_state.get("thinking", "(Agent Paused/Idle)")
    # Populate initial step history if available from loaded state
    initial_step_history_data = initial_state.get("step_history", [])
    log.info(f"Calculated initial status text for UI: {initial_status_text}")


# --- Global variable to store the last known Monitor UI state ---
# --- MODIFIED: Update tuple signature to match new outputs ---
last_monitor_state: Optional[Tuple[str, str, str, str, str, str, str, str]] = None

# --- Formatting Functions ---


def format_memories_for_display(
    memories: List[Dict[str, Any]], context_label: str
) -> str:
    """Formats a list of memories for markdown display using expandable sections."""
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

        import html

        safe_content = html.escape(content)

        summary_line = (
            f"**{i+1}. {relative_time} - Type:** {mem_type} (Dist: {dist_str})"
        )

        # Limit initial display length in summary line for brevity
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


# --- NEW: Formatting functions for Tool Results ---
def format_web_search_results(results_data: Optional[Dict[str, Any]]) -> str:
    """Formats web search results for display."""
    if not results_data or not isinstance(results_data, dict):
        return "(Invalid web search results data)"
    results_list = results_data.get("results", [])
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
    if not results_data or not isinstance(results_data, dict):
        return "(Invalid memory search results data)"
    results_list = results_data.get("retrieved_memories", [])
    if not results_list:
        return "(No relevant memories found in search)"

    output = ["**🧠 Memory Search Results:**\n"]
    for res in results_list:
        rank = res.get("rank", "-")
        rel_time = res.get("relative_time", "N/A")
        mem_type = res.get("type", "N/A")
        dist = res.get("distance", "N/A")
        snippet = res.get("content_snippet", "...")
        mem_id = res.get("memory_id", "N/A")
        output.append(f"**Rank {rank} ({rel_time})** - Type: {mem_type}, Dist: {dist}")
        # output.append(f"   ID: {mem_id}") # Optional: Include memory ID
        output.append(f"   > {snippet}\n")
    return "\n".join(output)


def format_web_browse_results(results_data: Optional[Dict[str, Any]]) -> str:
    """Formats web browse results for display (no truncation)."""
    if not results_data or not isinstance(results_data, dict):
        return "(Invalid web browse results data)"
    content = results_data.get("content")
    url = results_data.get("url", "N/A")
    source = results_data.get("content_source", "N/A")
    message = results_data.get("message", "")

    output = [
        f"**📄 Web Browse Result**\n\nURL: {url}\n\nSource Type: {source}\n\n{message}\n---\n"
    ]
    # Use Markdown code block for potentially long text content
    output.append(
        "```text\n" + (content if content else "(No content extracted)") + "\n```"
    )
    return "\n".join(output)


# --- MODIFIED: format_other_tool_results ---
def format_other_tool_results(results_data: Optional[Dict[str, Any]]) -> str:
    """Formats results from other tools (like file, status) or errors as JSON/Text."""
    if not results_data or not isinstance(results_data, dict):
        return "(Invalid tool results data)"
    tool_name = results_data.get("tool_name", "Unknown Tool")
    action = results_data.get("action", "Unknown Action")
    status = results_data.get("status", "Unknown Status")
    error = results_data.get("error")
    message = results_data.get("message")  # General message from tool

    # --- Special handling for 'status' tool report ---
    if (
        tool_name == "status"
        and action == "status_report"
        and "report_content" in results_data
    ):
        report_content = results_data.get(
            "report_content", "(Status report content missing)"
        )
        # Use Markdown code block for the report
        return f"**📊 Status Report**\nStatus: {status}\n{message or ''}\n---\n```markdown\n{report_content}\n```"
    # --- End special handling ---

    # --- General handling ---
    output = [f"**🛠️ Tool Result**\n {tool_name}: ({action})\nStatus: {status}\n"]
    if message:
        output.append(f"{message}\n")
    if error:
        output.append(f"**Error:** {error}\n")

    # Try to display the 'result' part prettily if no major error
    # Exclude known top-level keys to avoid redundancy
    result_content_keys = [
        k
        for k in results_data.keys()
        if k
        not in ["tool_name", "action", "status", "error", "message", "report_content"]
    ]
    result_content_display = {}
    if result_content_keys:
        for k in result_content_keys:
            result_content_display[k] = results_data[k]
    elif (
        not error
    ):  # If no specific result keys and no error, maybe show the whole thing minus known keys?
        result_content_display = {
            k: v
            for k, v in results_data.items()
            if k
            not in [
                "tool_name",
                "action",
                "status",
                "error",
                "message",
                "report_content",
            ]
        }

    if result_content_display:
        try:
            # Pretty print the result content as JSON
            result_json = json.dumps(
                result_content_display, indent=2, ensure_ascii=False
            )
            output.append("**Details:**\n```json\n" + result_json + "\n```")
        except Exception:
            # Fallback if JSON fails
            output.append("**Details:**\n" + str(result_content_display))

    return "\n".join(output)


# --- END MODIFIED ---


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
        time.sleep(0.2)
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
def update_monitor_ui() -> (
    Tuple[str, str, str, str, str, str, str, str]  # <<< Increased tuple size
):
    # Initialize with error/default values
    current_task_display = "(Error)"
    dependent_tasks_display = "Dependent Tasks: (Error)"
    thinking_display = "(Error)"
    step_log_output = "(Error)"
    memory_display = "(Error)"
    tool_results_display = "(Error)"  # <<< Renamed/Repurposed
    status_bar_text = "Error"
    final_answer_display = "(None)"

    if not agent_instance:
        status_bar_text = "Error: Agent not initialized."
        current_task_display = "Not Initialized"
        dependent_tasks_display = "Dependent Tasks: (Not Initialized)"
        thinking_display = "Not Initialized"
        step_log_output = "Not Initialized"
        memory_display = "Not Initialized"
        tool_results_display = "Not Initialized"
        final_answer_display = "Not Initialized"
        return (
            current_task_display,
            dependent_tasks_display,
            thinking_display,
            step_log_output,
            memory_display,
            tool_results_display,
            status_bar_text,
            final_answer_display,
        )

    try:
        ui_state = agent_instance.get_ui_update_state()

        # Core Task Info
        current_task_id = ui_state.get("current_task_id", "None")
        task_status = ui_state.get("status", "unknown")
        current_task_desc = ui_state.get("current_task_desc", "N/A")
        current_task_display = f"**ID:** {current_task_id}\n**Status:** {task_status}\n**Desc:** {current_task_desc}"

        # Dependent Tasks
        deps = ui_state.get("dependent_tasks", [])
        if deps:
            dep_lines = [
                f"- {d['id'][:8]}...: {d['description'][:60]}..." for d in deps
            ]
            dependent_tasks_display = "**Dependent Tasks:**\n" + "\n".join(dep_lines)
        else:
            dependent_tasks_display = "Dependent Tasks: (None)"

        # Thinking Process
        thinking_display = ui_state.get("thinking", "(No thinking recorded)")

        # Step Log
        step_log_output = ui_state.get("log", "(No log)")

        # Recent Memories (Contextual)
        recent_memories = ui_state.get("recent_memories", [])
        memory_display = format_memories_for_display(
            recent_memories, context_label="Monitor"
        )

        # --- MODIFIED: Consolidated Tool Results Display ---
        last_action = ui_state.get("last_action_type")
        last_results = ui_state.get("last_tool_results")  # Raw results dict

        if last_action == "use_tool" and last_results:
            tool_name = last_results.get("tool_name")
            action_name = last_results.get("action")
            # Call formatting functions based on tool/action
            if tool_name == "web" and action_name == "search":
                tool_results_display = format_web_search_results(last_results)
            elif tool_name == "web" and action_name == "browse":
                tool_results_display = format_web_browse_results(last_results)
            elif tool_name == "memory" and action_name == "search":
                tool_results_display = format_memory_search_results(last_results)
            # Updated to explicitly check for status tool using 'format_other_tool_results'
            # which now has special handling for it.
            elif (
                tool_name == "status"
                or tool_name == "file"
                or (tool_name == "memory" and action_name == "write")
            ):
                tool_results_display = format_other_tool_results(last_results)
            else:  # Fallback for unknown/unhandled tools/actions
                tool_results_display = format_other_tool_results(last_results)
        elif last_action == "final_answer":
            tool_results_display = "(Provided Final Answer)"
        elif last_action == "error":
            # Try to get more specific error message from log if possible
            error_line = "(Agent action resulted in error)"
            log_lines = ui_state.get("log", "").splitlines()
            for line in reversed(log_lines):
                if "[ERROR]" in line or "[CRITICAL]" in line:
                    error_line = line.strip()
                    break
            tool_results_display = error_line
        else:
            tool_results_display = "(No recent tool execution or action)"
        # --- End Tool Results Display Modification ---

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
        step_log_output = error_msg
        status_bar_text = "Error"
        current_task_display = "Error"
        dependent_tasks_display = "Dependent Tasks: (Error)"
        thinking_display = "Error"
        memory_display = "Error"
        tool_results_display = "Error"
        final_answer_display = "Error"

    return (
        current_task_display,
        dependent_tasks_display,
        thinking_display,
        step_log_output,
        memory_display,
        tool_results_display,
        status_bar_text,
        final_answer_display,
    )


# --- Functions for Step History Navigation ---
def format_step_details_for_display(
    step_data: Optional[Dict],
) -> Tuple[str, str, str, str]:
    """Formats a single step's data for the history view components."""
    if not step_data or not isinstance(step_data, dict):
        return "(No Step Data)", "(N/A)", "(N/A)", "(N/A)"

    step_num = step_data.get("step", "N/A")
    task_id = step_data.get("task_id", "N/A")
    timestamp = step_data.get("timestamp", "N/A")
    rel_time = format_relative_time(timestamp)

    title = f"**Step {step_num}** (Task: {task_id[:8]}...) - {rel_time}"
    thinking = step_data.get("thinking", "(No thinking recorded)")
    log_snippet = step_data.get("log_snippet", "(No log snippet)")
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
            results_display += f"**Params:** {action_params}\n\n"  # Fallback
    results_display += (
        f"**Result Status:**\n```json\n{json.dumps(result_status, indent=2)}\n```\n"
    )
    results_display += (
        f"**Result/Answer Summary:**\n```markdown\n{result_summary}\n```\n"
    )

    return title, thinking, log_snippet, results_display


def view_step_relative(
    current_index_str: str,  # Comes from UI state as string
    step_delta: int,
    history_data: List[Dict],  # Comes from UI state
) -> Tuple[str, str, str, str, str]:
    """Updates the history view based on navigating forward/backward."""
    try:
        current_index = int(current_index_str)
    except (ValueError, TypeError):
        current_index = 0  # Default to 0 if invalid

    if not history_data:
        title, thinking, log_s, result_s = format_step_details_for_display(None)
        return str(0), title, thinking, log_s, result_s  # No history, return defaults

    new_index = current_index + step_delta
    # Clamp index within bounds [0, len-1]
    new_index = max(0, min(len(history_data) - 1, new_index))

    selected_step_data = history_data[new_index]
    title, thinking, log_s, result_s = format_step_details_for_display(
        selected_step_data
    )

    return str(new_index), title, thinking, log_s, result_s


def view_latest_step(history_data: List[Dict]) -> Tuple[str, str, str, str, str]:
    """Jumps the history view to the most recent step."""
    if not history_data:
        title, thinking, log_s, result_s = format_step_details_for_display(None)
        return str(0), title, thinking, log_s, result_s  # No history, return defaults

    latest_index = len(history_data) - 1
    selected_step_data = history_data[latest_index]
    title, thinking, log_s, result_s = format_step_details_for_display(
        selected_step_data
    )
    return str(latest_index), title, thinking, log_s, result_s


# --- Functions for Chat Tab (Largely Unchanged, only memory formatting) ---
def _should_generate_task(user_msg: str, assistant_response: str) -> bool:  # Unchanged
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


def prioritize_generated_task(last_generated_task_id: Optional[str]):  # Unchanged
    feedback = "Prioritization failed: Agent not initialized."
    if agent_instance:
        if last_generated_task_id:
            log.info(
                f"UI: User requested prioritization for task {last_generated_task_id}"
            )
            success = agent_instance.task_queue.set_priority(last_generated_task_id, 9)
            if success:
                task = agent_instance.task_queue.get_task(last_generated_task_id)
                feedback = (
                    f"Task '{task.description[:30]}...' priority set to 9."
                    if task
                    else f"Task {last_generated_task_id} priority set to 9 (task details not found)."
                )
            else:
                feedback = f"Failed to set priority for task {last_generated_task_id} (may already be completed/failed or ID invalid)."
        else:
            feedback = "Prioritization failed: No task ID provided (was a task generated this session?)."
    return feedback


def inject_chat_info(
    message_to_inject: str,
):  # Unchanged
    feedback = "Inject Info failed: Agent not initialized."
    if agent_instance:
        if message_to_inject and message_to_inject.strip():
            current_task_id = agent_instance.session_state.get("current_task_id")
            if agent_instance._is_running.is_set():
                log.info(
                    f"UI: Injecting info for current task ({current_task_id}): '{message_to_inject[:50]}...'"
                )
                mem_id = agent_instance.memory.add_memory(
                    content=f"User Provided Info (for current task):\n{message_to_inject}",
                    metadata={
                        "type": "user_provided_info_for_task",
                        "task_id_context": current_task_id or "idle",
                    },
                )
                if mem_id:
                    feedback = f"Information added to memory for agent to consider (for task {current_task_id or 'N/A'})."
                else:
                    feedback = "Error adding information to memory."
            else:
                log.info(
                    f"Agent paused, ignoring injected info for task {current_task_id or 'N/A'} (not adding to memory)."
                )
                feedback = f"Info noted, but agent is paused. Memory not updated. (Task: {current_task_id or 'N/A'})."
        else:
            feedback = "Inject Info failed: No message provided in the input box."
    return feedback


def chat_response(
    message: str,
    history: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], str, str, str, Optional[str]]:  # Unchanged Logic
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
        agent_is_running = agent_instance._is_running.is_set()  # Check agent run state
        agent_state = agent_instance.get_ui_update_state()
        agent_identity = agent_instance.identity_statement  # Fetch current identity
        agent_task_id = agent_state.get("current_task_id")
        agent_task_desc = agent_state.get("current_task_desc", "N/A")
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
            agent_activity_context = f"Working on Task {agent_task_id}: '{agent_task_desc}'. Status: {agent_status}."
        elif agent_status == "paused":
            agent_activity_context = (
                f"Paused. Active task {agent_task_id}: '{agent_task_desc}'."
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


# --- Functions for Agent State Tab (Unchanged) ---
def refresh_agent_state_display():
    log.info("State Tab: Refresh button clicked. Fetching latest state...")
    if not agent_instance:
        error_df = pd.DataFrame(columns=["Error"])
        return (
            "Agent not initialized.",  # identity
            error_df,  # pending
            error_df,  # inprogress
            error_df,  # completed
            error_df,  # failed
            "Error: Agent not initialized.",  # memory summary
            gr.Dropdown(choices=["Error"], value="Error"),  # dropdown
            error_df,  # task memory display
            error_df,  # general memory display
        )

    try:
        state_data = agent_instance.get_agent_dashboard_state()

        def create_or_default_df(data, columns):
            df = pd.DataFrame(data)
            if df.empty:
                return pd.DataFrame(columns=columns)
            for col in columns:
                if col not in df.columns:
                    df[col] = pd.NA
            # Ensure consistent column order and presence
            return df.reindex(columns=columns, fill_value=pd.NA)

        pending_df = create_or_default_df(
            state_data.get("pending_tasks", []),
            ["ID", "Priority", "Description", "Depends On", "Created"],
        )
        inprogress_df = create_or_default_df(
            state_data.get("in_progress_tasks", []),
            ["ID", "Priority", "Description", "Created"],
        )
        completed_df = create_or_default_df(
            state_data.get("completed_tasks", []),
            ["ID", "Description", "Completed At", "Result Snippet"],
        )
        failed_df = create_or_default_df(
            state_data.get("failed_tasks", []),
            ["ID", "Description", "Failed At", "Reason"],
        )

        completed_failed_tasks_data = state_data.get("completed_failed_tasks_data", [])
        task_id_choices_tuples = [
            (f"{t['ID']} - {t['Description'][:50]}...", t["ID"])
            for t in completed_failed_tasks_data
        ]
        dropdown_choices = [("Select Task ID...", None)] + task_id_choices_tuples
        initial_dropdown_value = None

        general_memories = agent_instance.get_formatted_general_memories()
        for mem in general_memories:
            mem["Relative Time"] = format_relative_time(mem.get("Timestamp"))
        general_mem_df = create_or_default_df(
            general_memories,
            ["Relative Time", "Timestamp", "Type", "Content Snippet", "ID"],
        )

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
                label="Select Task ID (Completed/Failed)",
            ),
            pd.DataFrame(
                columns=["Relative Time", "Timestamp", "Type", "Content Snippet", "ID"]
            ),  # Task memories display (start empty)
            general_mem_df,
        )

    except Exception as e:
        log.exception("Error refreshing agent state display")
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
            error_df,
        )


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
            return mem_df.reindex(
                columns=columns, fill_value=pd.NA
            )  # Use reindex for safety
    except Exception as e:
        log.exception(f"Error fetching memories for task {selected_task_id}")
        return pd.DataFrame([{"Error": str(e)}])


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
            return mem_df.reindex(
                columns=columns, fill_value=pd.NA
            )  # Use reindex for safety
    except Exception as e:
        log.exception("Error fetching general memories")
        return pd.DataFrame([{"Error": str(e)}])


# --- Gradio UI Definition ---
log.info("Defining Gradio UI...")
if agent_instance is None:
    log.critical("Agent failed init. Cannot launch UI.")
    with gr.Blocks() as demo:
        gr.Markdown("# Fatal Error\nAgent backend failed to initialize. Check logs.")
else:
    with gr.Blocks(theme=gr.themes.Glass(), title="Autonomous Agent Interface") as demo:
        gr.Markdown("# Autonomous Agent Control Center & Chat")

        # --- State for Step History ---
        step_history_state = gr.State(
            initial_step_history_data
        )  # Stores the list of step dicts
        step_history_index_state = gr.State(
            str(len(initial_step_history_data) - 1 if initial_step_history_data else 0)
        )  # Index of the currently viewed step (STORE AS STRING)

        with gr.Tabs():
            # --- MODIFIED: Monitor Tab ---
            with gr.TabItem("Agent Monitor"):
                gr.Markdown(
                    "Monitor/control agent processing. Suggestions added to memory for agent consideration. Use Step History for details."
                )
                with gr.Row():
                    start_resume_btn = gr.Button("Start / Resume", variant="primary")
                    pause_btn = gr.Button("Pause")
                    suggest_change_btn = gr.Button("Suggest Task Change")
                monitor_status_bar = gr.Textbox(
                    label="Agent Status", value=initial_status_text, interactive=False
                )
                suggestion_feedback_box = gr.Textbox(
                    label="Suggestion Feedback", value="", interactive=False, lines=1
                )
                with gr.Row():
                    with gr.Column(scale=1):  # Left Column
                        monitor_current_task = gr.Markdown("(Agent Initializing)")
                        monitor_dependent_tasks = gr.Markdown(
                            initial_dependent_tasks_text
                        )
                        monitor_thinking = gr.Textbox(
                            label="🤖 Agent Thinking (Current Step)",
                            value=initial_thinking_text,
                            lines=8,
                            interactive=False,
                            show_copy_button=True,
                        )
                        monitor_log = gr.Textbox(
                            label="Last Step Log Snippet",
                            lines=8,
                            interactive=False,
                            autoscroll=True,
                        )
                        monitor_final_answer = gr.Textbox(
                            label="Last Final Answer",
                            lines=5,
                            interactive=False,
                            show_copy_button=True,
                        )
                    with gr.Column(scale=1):  # Right Column
                        # --- MODIFIED: Consolidated Tool Result / Browse Box ---
                        monitor_tool_results = gr.Markdown(  # Changed to Markdown for better formatting
                            value=initial_tool_results_text,
                            label="Tool Output / Browse Content",
                            # lines=18, # lines property not applicable to Markdown
                            # interactive=False, # Not applicable
                            # show_copy_button=True, # Not applicable
                            # No autoscroll needed, implicitly scrollable
                        )
                        monitor_memory = gr.Markdown(
                            value="Recent Memories (Monitor)\n(Agent Initializing)"
                        )

                # --- NEW: Step History Viewer ---
                with gr.Accordion("Step History Explorer", open=False):
                    # Navigation
                    with gr.Row():
                        step_prev_btn = gr.Button("◀ Previous Step")
                        step_latest_btn = gr.Button("Latest Step", variant="secondary")
                        step_next_btn = gr.Button("Next Step ▶")
                    # Display Fields
                    step_hist_title = gr.Markdown(
                        "**Step History** (Load latest to view)"
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
                            )
                        with gr.Column(scale=1):
                            # Changed to Markdown for better formatting of results
                            step_hist_result = gr.Markdown(
                                label="Action/Result Summary",
                                value="(No history selected)",
                            )
                            # step_hist_result = gr.Textbox(label="Action/Result Summary", lines=15, interactive=False)

                # --- Connect Button/Timer Events ---
                start_resume_btn.click(
                    fn=start_agent_processing, inputs=None, outputs=monitor_status_bar
                )
                pause_btn.click(
                    fn=pause_agent_processing, inputs=None, outputs=monitor_status_bar
                )
                suggest_change_btn.click(
                    fn=suggest_task_change, inputs=None, outputs=suggestion_feedback_box
                )

                # Connect Step History Buttons
                # --- MODIFIED: Step History Outputs (Textbox -> Markdown) ---
                step_hist_view_outputs = [
                    step_history_index_state,
                    step_hist_title,
                    step_hist_thinking,
                    step_hist_log,
                    step_hist_result,
                ]
                step_prev_btn.click(
                    fn=view_step_relative,
                    inputs=[
                        step_history_index_state,
                        gr.State(-1),
                        step_history_state,
                    ],  # Pass delta -1
                    outputs=step_hist_view_outputs,
                )
                step_next_btn.click(
                    fn=view_step_relative,
                    inputs=[
                        step_history_index_state,
                        gr.State(1),
                        step_history_state,
                    ],  # Pass delta +1
                    outputs=step_hist_view_outputs,
                )
                step_latest_btn.click(
                    fn=view_latest_step,
                    inputs=[step_history_state],
                    outputs=step_hist_view_outputs,
                )

                # Global Timer for Monitor Tab & Step History Update
                timer = gr.Timer(config.UI_UPDATE_INTERVAL)  # Use configured interval

                # Update Function for Monitor Tab & Step History
                def update_monitor_and_history():
                    """Handles periodic UI updates ONLY for the Monitor tab, suggestion feedback, and step history state."""
                    global last_monitor_state
                    num_monitor_outputs = 8
                    monitor_updates_to_return = (
                        "(Initializing)",
                    ) * num_monitor_outputs
                    current_history_data = []

                    if agent_instance:
                        try:
                            agent_ui_state = agent_instance.get_ui_update_state()
                            current_history_data = agent_ui_state.get(
                                "step_history", []
                            )

                            if (
                                agent_instance._is_running.is_set()
                                or last_monitor_state is None
                            ):
                                log.debug(
                                    "Agent running or first update, getting fresh monitor state..."
                                )
                                current_monitor_state = update_monitor_ui()
                                last_monitor_state = current_monitor_state
                                monitor_updates_to_return = current_monitor_state
                            else:
                                log.debug(
                                    "Agent paused/stopped, using cached monitor state."
                                )
                                if last_monitor_state is not None:
                                    # If paused, ensure status text reflects pause
                                    cached_list = list(last_monitor_state)
                                    cached_list[6] = (
                                        f"Agent Status: paused @ {agent_ui_state.get('timestamp', 'N/A')}"
                                    )
                                    monitor_updates_to_return = tuple(cached_list)
                                else:
                                    monitor_updates_to_return = update_monitor_ui()
                        except Exception as e:
                            log.exception("Error during monitor update")
                            monitor_updates_to_return = (
                                "Error: Update Failed",
                            ) * num_monitor_outputs
                            current_history_data = [{"error": f"Update failed: {e}"}]
                    else:
                        monitor_updates_to_return = (
                            "Error: Agent Offline",
                        ) * num_monitor_outputs
                        current_history_data = [{"error": "Agent Offline"}]

                    if (
                        not isinstance(monitor_updates_to_return, tuple)
                        or len(monitor_updates_to_return) != num_monitor_outputs
                    ):
                        log.error(
                            f"Monitor update tuple structure mismatch. Expected {num_monitor_outputs}. Got {len(monitor_updates_to_return) if isinstance(monitor_updates_to_return, tuple) else type(monitor_updates_to_return)}"
                        )
                        monitor_updates_to_return = (
                            "Error: Struct Mismatch",
                        ) * num_monitor_outputs

                    # --- MODIFIED: Ensure tool results are formatted for Markdown ---
                    # The update_monitor_ui function now returns formatted strings suitable for Markdown components
                    formatted_monitor_updates = list(monitor_updates_to_return)
                    # Index 5 is monitor_tool_results, which is now a Markdown string
                    # Index 4 is monitor_memory, which is already Markdown

                    return *formatted_monitor_updates, current_history_data

                # Connect Timer
                monitor_timer_outputs = [
                    monitor_current_task,
                    monitor_dependent_tasks,
                    monitor_thinking,
                    monitor_log,
                    monitor_memory,
                    monitor_tool_results,  # These are now Markdown outputs
                    monitor_status_bar,
                    monitor_final_answer,
                    step_history_state,  # <<< Add history state to timer outputs
                ]
                timer.tick(
                    fn=update_monitor_and_history,
                    inputs=None,
                    outputs=monitor_timer_outputs,
                )

                # Initial population of step history viewer (if history exists)
                (
                    initial_hist_index_str,
                    initial_hist_title,
                    initial_hist_think,
                    initial_hist_log,
                    initial_hist_result,
                ) = view_latest_step(initial_step_history_data)
                # The lambda function ensures the latest initial values are used when the UI loads
                demo.load(
                    fn=lambda: (
                        initial_hist_index_str,
                        initial_hist_title,
                        initial_hist_think,
                        initial_hist_log,
                        initial_hist_result,
                    ),
                    inputs=[],
                    outputs=[
                        step_history_index_state,
                        step_hist_title,
                        step_hist_thinking,
                        step_hist_log,
                        step_hist_result,
                    ],
                )

            # --- MODIFIED Chat Tab Layout ---
            with gr.TabItem("Chat"):
                gr.Markdown(
                    "Interact with the agent. It considers its identity, activity, and your input."
                )
                last_generated_task_id_state = gr.State(None)
                with gr.Row():
                    with gr.Column(scale=3):
                        # --- FIX 2: Define Chatbot variable and RENDER it immediately ---
                        chat_chatbot = gr.Chatbot(
                            label="Conversation",
                            height=500,
                            show_copy_button=True,
                            type="messages",
                            render=False,
                        )
                        chat_chatbot.render()  # RENDER HERE TO PUT IT AT THE TOP OF THE COLUMN
                        # --- END FIX 2 ---

                        # Define other input elements AFTER chatbot is rendered
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
                                label="Your Message",
                                placeholder="Type message and press Enter or click Send...",
                                lines=3,
                                scale=4,
                                container=False,
                            )
                            chat_send_button = gr.Button(
                                "Send", variant="primary", scale=1
                            )
                            inject_info_btn = gr.Button("Inject Info for Task", scale=1)
                        # Removed chat_chatbot.render() from here

                    with gr.Column(scale=1):
                        chat_memory_panel = gr.Markdown(
                            value="Relevant Memories (Chat)\n...",
                            label="Memory Context",
                        )

                chat_outputs = [
                    chat_chatbot,
                    chat_memory_panel,
                    chat_task_panel,
                    chat_msg_input,
                    last_generated_task_id_state,
                ]
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
                prioritize_task_btn.click(
                    fn=prioritize_generated_task,
                    inputs=[last_generated_task_id_state],
                    outputs=[chat_interaction_feedback],
                )
                inject_info_btn.click(
                    fn=inject_chat_info,
                    inputs=[chat_msg_input],
                    outputs=[chat_interaction_feedback],
                )
            # --- END MODIFIED Chat Tab ---

            # --- Agent State Tab (Unchanged) ---
            with gr.TabItem("Agent State"):
                gr.Markdown(
                    "View the agent's current identity, task queues, and memory. **Use buttons to load/refresh data.**"
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
                        gr.Markdown("#### In Progress Task(s)")
                        with gr.Row():
                            state_inprogress_tasks = gr.DataFrame(
                                headers=["ID", "Priority", "Description", "Created"],
                                interactive=True,
                                wrap=True,
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
                                interactive=True,
                                wrap=True,
                            )
                        gr.Markdown("#### Failed Tasks (Most Recent First)")
                        with gr.Row():
                            state_failed_tasks = gr.DataFrame(
                                headers=["ID", "Description", "Failed At", "Reason"],
                                interactive=True,
                                wrap=True,
                            )

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
                                wrap=True,
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
                                wrap=True,
                            )
                        with gr.Row(scale=1):
                            refresh_general_mem_btn = gr.Button(
                                "Refresh General Memories"
                            )

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
                ]

                load_state_button.click(
                    fn=refresh_agent_state_display,
                    inputs=None,
                    outputs=state_tab_outputs,
                )
                state_task_memory_select.change(
                    fn=update_task_memory_display,
                    inputs=[state_task_memory_select],
                    outputs=[state_task_memory_display],
                )
                refresh_general_mem_btn.click(
                    fn=update_general_memory_display,
                    inputs=None,
                    outputs=[state_general_memory_display],
                )


# --- Launch the App & Background Thread ---
if __name__ == "__main__":
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
            # --- Initial population moved inside the 'with gr.Blocks' block ---
            # demo.load(...) is now correctly placed within the Monitor Tab definition section.
            demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
        except Exception as e:
            log.critical(f"Gradio launch failed: {e}", exc_info=True)
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
            sys.exit("Gradio launch failed.")
    log.info("Gradio App stopped. Requesting agent shutdown...")
    if agent_instance:
        agent_instance.shutdown()
    log.info("Shutdown complete.")
    print("\n--- Autonomous Agent App End ---")
