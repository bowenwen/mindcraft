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
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd  # <<<--- IMPORT REMAINS

# --- Project Imports ---
import config
from utils import call_ollama_api, format_relative_time
from memory import AgentMemory, setup_chromadb
from task_manager import TaskQueue
from data_structures import Task
from agent import AutonomousAgent
import chromadb

# --- Logging Setup ---
# ... (logging setup unchanged) ...
import logging

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="[%(asctime)s] [%(levelname)s][%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("AppUI")


# --- Global Variables / Setup ---
# ... (agent initialization unchanged) ...
log.info("Initializing components for Gradio App...")
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
if agent_instance:
    log.info("Setting initial agent state to paused for UI...")
    agent_instance.start_autonomous_loop()
    agent_instance.pause_autonomous_loop()  # Agent now starts paused by default
    initial_state = agent_instance.get_ui_update_state()
    initial_status_text = f"Agent Status: {initial_state.get('status', 'paused')} @ {initial_state.get('timestamp', 'N/A')}"
    log.info(f"Calculated initial status text for UI: {initial_status_text}")

# --- Global variable to store the last known Monitor UI state (unchanged) ---
last_monitor_state: Optional[Tuple[str, str, str, str, str, str]] = None


def format_memories_for_display(
    memories: List[Dict[str, Any]], context_label: str
) -> str:
    """Formats a list of memories for markdown display using expandable sections."""
    if not memories:
        return f"No relevant memories found for {context_label}."

    # --- MODIFIED: Use HTML <details> for expandability ---
    output_parts = [f"🧠 **Recent/Relevant Memories ({context_label}):**\n"]
    for i, mem in enumerate(
        memories[:7]
    ):  # Limit display slightly more due to expansion
        content = mem.get("content", "N/A")
        meta = mem.get("metadata", {})
        mem_type = meta.get("type", "memory")
        dist_str = (
            f"{mem.get('distance', -1.0):.3f}"
            if mem.get("distance") is not None
            else "N/A"
        )
        timestamp = meta.get("timestamp")
        relative_time = format_relative_time(timestamp)  # Get relative time

        # Sanitize content for HTML display within <pre>
        import html

        safe_content = html.escape(content)

        summary_line = (
            f"**{i+1}. {relative_time} - Type:** {mem_type} (Dist: {dist_str})"
        )

        # Use <details> and <summary> HTML tags for an expandable section
        # Use <pre> inside for preserving formatting of the content
        details_block = f"""
<details>
  <summary>{summary_line}</summary>
  <pre style="background-color: #f0f0f0; border: 1px solid #ccc; padding: 5px; margin-top: 5px; white-space: pre-wrap; word-wrap: break-word;">{safe_content}</pre>
</details>
"""
        output_parts.append(details_block)

    return "\n".join(output_parts)


# --- Functions for Monitor Tab (unchanged structure, but uses updated format_memories_for_display) ---
def start_agent_processing():  # Unchanged
    # ... (implementation unchanged) ...
    if agent_instance:
        log.info("UI: Start/Resume Agent")
        agent_instance.start_autonomous_loop()
        state = agent_instance.get_ui_update_state()
        return f"Agent Status: {state.get('status', 'running')} @ {state.get('timestamp', 'N/A')}"
    else:
        log.error("UI failed: Agent not initialized.")
        return "ERROR: Agent not initialized."


def pause_agent_processing():  # Unchanged
    # ... (implementation unchanged) ...
    if agent_instance:
        log.info("UI: Pause Agent")
        agent_instance.pause_autonomous_loop()
        time.sleep(0.2)
        state = agent_instance.get_ui_update_state()
        return f"Agent Status: {state.get('status', 'paused')} @ {state.get('timestamp', 'N/A')}"
    else:
        log.error("UI failed: Agent not initialized.")
        return "ERROR: Agent not initialized."


def suggest_task_change():  # Modified to use agent's dedicated handling
    feedback = "Suggestion ignored: Agent not initialized."
    if agent_instance:
        log.info("UI: User clicked 'Suggest Task Change'")
        # --- NEW: Call agent's method to handle this suggestion ---
        result = agent_instance.handle_user_suggestion_move_on()
        feedback = result  # Use the feedback message from the agent method
    else:
        log.error("UI suggest_task_change failed: Agent not initialized.")
    return feedback


def update_monitor_ui() -> (
    Tuple[str, str, str, str, str, str]
):  # Unchanged logic, uses helper function now
    # ... (implementation largely unchanged, calls updated format_memories_for_display) ...
    current_task_display = "(Error)"
    step_log_output = "(Error)"
    memory_display = "(Error)"
    web_content_display = "(Error)"
    status_bar_text = "Error"
    final_answer_display = "(None)"
    if not agent_instance:
        status_bar_text = "Error: Agent not initialized."
        current_task_display = "Not Initialized"
        step_log_output = "Not Initialized"
        memory_display = "Not Initialized"
        web_content_display = "Not Initialized"
        final_answer_display = "Not Initialized"
        return (
            current_task_display,
            step_log_output,
            memory_display,
            web_content_display,
            status_bar_text,
            final_answer_display,
        )
    try:
        ui_state = agent_instance.get_ui_update_state()
        current_task_id = ui_state.get("current_task_id", "None")
        task_status = ui_state.get("status", "unknown")
        current_task_desc = ui_state.get("current_task_desc", "N/A")
        step_log_output = ui_state.get("log", "(No log)")
        recent_memories = ui_state.get("recent_memories", [])
        last_browse_content = ui_state.get("last_web_content", "(None)")
        final_answer = ui_state.get("final_answer")
        current_task_display = f"**ID:** {current_task_id}\n**Status:** {task_status}\n**Desc:** {current_task_desc}"
        # --- Uses updated formatter ---
        memory_display = format_memories_for_display(
            recent_memories, context_label="Monitor"
        )
        web_content_limit = 2000
        web_content_display = (
            last_browse_content[:web_content_limit] + "\n... (truncated)"
            if last_browse_content and len(last_browse_content) > web_content_limit
            else last_browse_content
        ) or "(None)"
        status_bar_text = (
            f"Agent Status: {task_status} @ {ui_state.get('timestamp', 'N/A')}"
        )
        final_answer_display = (
            final_answer if final_answer else "(No recent final answer)"
        )
    except Exception as e:
        log.exception("Error in update_monitor_ui")
        error_msg = f"ERROR updating UI:\n{traceback.format_exc()}"
        step_log_output = error_msg
        status_bar_text = "Error"
        current_task_display = "Error"
        memory_display = "Error"
        web_content_display = "Error"
        final_answer_display = "Error"
    return (
        current_task_display,
        step_log_output,
        memory_display,
        web_content_display,
        status_bar_text,
        final_answer_display,
    )


# --- Functions for Chat Tab ---
def _should_generate_task(user_msg: str, assistant_response: str) -> bool:  # Unchanged
    # ... (implementation unchanged) ...
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
    # ... (implementation unchanged) ...
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
):  # Logic unchanged (conditional memory add already there)
    # ... (implementation unchanged) ...
    feedback = "Inject Info failed: Agent not initialized."
    if agent_instance:
        if message_to_inject and message_to_inject.strip():
            current_task_id = agent_instance.session_state.get("current_task_id")
            # --- MODIFIED: Only add memory if agent is running ---
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
) -> Tuple[List[Dict[str, str]], str, str, str, Optional[str]]:
    # ... (most implementation unchanged, only memory formatting for prompt modified) ...
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
        # --- Retrieval implicitly uses updated logic in agent.py ---
        relevant_memories, _ = agent_instance.memory.retrieve_and_rerank_memories(
            query=memory_query,
            task_description="Responding in chat, considering identity/activity.",
            context=f"{history_context_str}\nActivity: {agent_activity_context}\nIdentity: {agent_identity}",
            identity_statement=agent_instance.identity_statement,  # <<<--- PASS IDENTITY
            n_results=config.MEMORY_COUNT_CHAT_RESPONSE * 2,
            n_final=config.MEMORY_COUNT_CHAT_RESPONSE,  # Retrieve up to 10 memories for context
        )
        # --- Uses updated formatter for the UI panel ---
        memory_display_text = format_memories_for_display(
            relevant_memories, context_label="Chat"
        )
        log.info(f"Retrieved {len(relevant_memories)} memories for chat.")
        system_prompt = f"""You are helpful AI assistant ({agent_identity}). Answer user conversationally. Consider identity, chat history, memories, agent's background activity. Be aware of your capabilities and limitations."""

        # --- *** MODIFIED: Use memory SNIPPETS in prompt *** ---
        memories_for_prompt_list = []
        snippet_length = 250  # Max length for each memory snippet in the prompt
        for mem in relevant_memories:  # Use the retrieved memories
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
        # --- *** END MODIFICATION *** ---

        history_for_prompt = "\n".join(
            [
                f"{t.get('role','?').capitalize()}: {t.get('content','')}"
                for t in history
            ]
        )
        prompt = f"{system_prompt}\n\n## Agent Background Activity:\n{agent_activity_context}\n\n## Relevant Memory Snippets:\n{memories_for_prompt}\n\n## Chat History:\n{history_for_prompt}\n\n## Current Query:\nUser: {message}\nAssistant:"

        # --- Debugging: Log the approximate prompt length ---
        log.debug(f"Approximate prompt length for chat response: {len(prompt)} chars")

        log.info(f"Asking {config.OLLAMA_CHAT_MODEL} for chat response...")
        response_text = call_ollama_api(
            prompt=prompt,
            model=config.OLLAMA_CHAT_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            timeout=config.OLLAMA_TIMEOUT,  # Use configured timeout
        )
        if not response_text:
            response_text = (
                "Sorry, error generating response."  # This is the fallback users see
            )
            log.error(
                "LLM call failed for chat response. Prompt might be too long or Ollama error occurred."
            )
        history.append({"role": "assistant", "content": response_text})

        # --- MODIFIED: Only add chat history to memory if agent is running ---
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

        # --- Task generation logic remains the same ---
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
                    # Avoid modifying history if response was the error message
                    if response_text != "Sorry, error generating response.":
                        notification = f"\n\n*(Okay, based on our chat, I've created task {last_gen_id}: \"{first_task_generated}\". I'll work on it when possible. You can prioritize it if needed.)*"
                        history[-1]["content"] += notification
                    agent_instance._save_qlora_datapoint(
                        source_type="chat_task_generation",
                        instruction="User interaction led to this task. Respond confirming task creation.",
                        input_context=f"Identity: {agent_identity}\nUser: {message}\nActivity: {agent_activity_context}",
                        output=f"{response_text}{notification if response_text != 'Sorry, error generating response.' else ''}",  # Add notification to QLoRA only if included in actual response
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
# ... (refresh_agent_state_display, update_task_memory_display, update_general_memory_display remain the same) ...
def refresh_agent_state_display():
    """Fetches and formats all data needed for the state tab UI update."""
    log.info("State Tab: Refresh button clicked. Fetching latest state...")
    if not agent_instance:
        # Return default empty structures matching the output tuple
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
        # 1. Fetch core state data
        state_data = agent_instance.get_agent_dashboard_state()

        # 2. Format DataFrames (handle empty lists)
        def create_or_default_df(data, columns):
            df = pd.DataFrame(data)
            if df.empty:
                return pd.DataFrame(columns=columns)
            # Ensure all expected columns exist, add missing ones with default value (e.g., NA or empty string)
            for col in columns:
                if col not in df.columns:
                    df[col] = (
                        pd.NA
                    )  # Or use "" or specific default based on expected type
            # Reindex to ensure correct column order even if input data had different order
            return df.reindex(columns=columns)

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

        # 3. Prepare dropdown choices with description
        completed_failed_tasks_data = state_data.get("completed_failed_tasks_data", [])
        task_id_choices_tuples = [
            (
                f"{t['ID']} - {t['Description'][:50]}...",
                t["ID"],
            )  # Format: (Label, Value)
            for t in completed_failed_tasks_data
        ]
        dropdown_choices = [
            ("Select Task ID...", None)
        ] + task_id_choices_tuples  # Use None as value for placeholder
        initial_dropdown_value = None  # Default to placeholder

        # 4. Fetch general memories (with relative time)
        general_memories = (
            agent_instance.get_formatted_general_memories()
        )  # Should return dicts
        # Add relative time column
        for mem in general_memories:
            mem["Relative Time"] = format_relative_time(mem.get("Timestamp"))

        general_mem_df = create_or_default_df(
            general_memories,
            [
                "Relative Time",
                "Timestamp",
                "Type",
                "Content Snippet",
                "ID",
            ],  # Add Relative Time
        )

        # 5. Return all components needed for the state tab UI update
        # Note: The task-specific memory dataframe starts empty here; it gets populated by dropdown change
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
            ),  # Update Dropdown component fully
            pd.DataFrame(
                columns=["Relative Time", "Timestamp", "Type", "Content Snippet", "ID"]
            ),  # Task memories display (start empty, add Relative Time)
            general_mem_df,  # General memories display (updated by this refresh)
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
    """Fetches and formats memories for the selected task ID."""
    log.debug(f"State Tab: Task ID selected: {selected_task_id}")
    columns = [
        "Relative Time",
        "Timestamp",
        "Type",
        "Content Snippet",
        "ID",
    ]  # Include Relative Time
    if not agent_instance or not selected_task_id:
        return pd.DataFrame(columns=columns)  # Return empty df with correct columns

    try:
        memories = agent_instance.get_formatted_memories_for_task(
            selected_task_id
        )  # Returns list of dicts
        # Add relative time column
        for mem in memories:
            mem["Relative Time"] = format_relative_time(mem.get("Timestamp"))

        mem_df = pd.DataFrame(memories)
        # Ensure columns exist even if empty
        if mem_df.empty:
            return pd.DataFrame(columns=columns)
        else:
            # Reorder columns for consistency
            return mem_df.reindex(columns=columns)
    except Exception as e:
        log.exception(f"Error fetching memories for task {selected_task_id}")
        return pd.DataFrame([{"Error": str(e)}])


def update_general_memory_display():
    """Fetches and formats general memories."""
    log.debug("State Tab: Refreshing general memories display data...")
    columns = [
        "Relative Time",
        "Timestamp",
        "Type",
        "Content Snippet",
        "ID",
    ]  # Include Relative Time
    if not agent_instance:
        return pd.DataFrame(columns=columns)

    try:
        memories = (
            agent_instance.get_formatted_general_memories()
        )  # Returns list of dicts
        # Add relative time column
        for mem in memories:
            mem["Relative Time"] = format_relative_time(mem.get("Timestamp"))

        mem_df = pd.DataFrame(memories)
        if mem_df.empty:
            return pd.DataFrame(columns=columns)
        else:
            # Reorder columns for consistency
            return mem_df.reindex(columns=columns)
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

        with gr.Tabs():
            # --- Monitor Tab (Structure Unchanged, Timer targets this) ---
            with gr.TabItem("Agent Monitor"):
                gr.Markdown(
                    "Monitor/control agent processing. Suggestions added to memory for agent consideration."
                )
                with gr.Row():
                    start_resume_btn = gr.Button("Start / Resume", variant="primary")
                    pause_btn = gr.Button("Pause")
                    # --- MODIFIED: Button now calls agent's handler ---
                    suggest_change_btn = gr.Button("Suggest Task Change")
                monitor_status_bar = gr.Textbox(
                    label="Agent Status", value=initial_status_text, interactive=False
                )
                suggestion_feedback_box = gr.Textbox(
                    label="Suggestion Feedback", value="", interactive=False, lines=1
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        monitor_current_task = gr.Markdown("(Agent Initializing)")
                        monitor_log = gr.Textbox(
                            label="Last Step Log",
                            lines=10,
                            interactive=False,
                            autoscroll=True,
                        )
                        monitor_final_answer = gr.Textbox(
                            label="Last Final Answer",
                            lines=5,
                            interactive=False,
                            show_copy_button=True,
                        )
                    with gr.Column(scale=1):
                        # --- MODIFIED: Memory display is now Markdown (uses updated formatter) ---
                        monitor_memory = gr.Markdown(
                            value="Recent Memories (Monitor)\n(Agent Initializing)"
                        )
                        monitor_web_content = gr.Textbox(
                            label="Last Web Content Fetched",
                            lines=10,
                            interactive=False,
                            show_copy_button=True,
                        )

                # Define outputs for the monitor tab + feedback box (targeted by timer)
                monitor_outputs_with_feedback = [
                    monitor_current_task,
                    monitor_log,
                    monitor_memory,
                    monitor_web_content,
                    monitor_status_bar,
                    monitor_final_answer,
                    suggestion_feedback_box,
                ]
                start_resume_btn.click(
                    fn=start_agent_processing, inputs=None, outputs=monitor_status_bar
                )
                pause_btn.click(
                    fn=pause_agent_processing, inputs=None, outputs=monitor_status_bar
                )
                # --- MODIFIED: Connect button to the dedicated handler ---
                suggest_change_btn.click(
                    fn=suggest_task_change, inputs=None, outputs=suggestion_feedback_box
                )

            # --- Chat Tab (Structure Unchanged, uses updated memory formatter) ---
            with gr.TabItem("Chat"):
                # ... (Chat tab structure and event handling unchanged) ...
                gr.Markdown(
                    "Interact with the agent. It considers its identity, activity, and your input."
                )
                last_generated_task_id_state = gr.State(None)
                with gr.Row():
                    with gr.Column(scale=3):
                        chat_chatbot = gr.Chatbot(
                            label="Conversation",
                            height=500,
                            show_copy_button=True,
                            type="messages",
                        )
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
                    with gr.Column(scale=1):
                        # --- MODIFIED: Memory panel is now Markdown (uses updated formatter) ---
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

            # --- Agent State Tab (Added Relative Time column) ---
            with gr.TabItem("Agent State"):
                gr.Markdown(
                    "View the agent's current identity, task queues, and memory. **Use buttons to load/refresh data.**"
                )  # Added note
                with gr.Row():
                    state_identity = gr.Textbox(
                        label="Agent Identity Statement",
                        lines=3,
                        interactive=False,
                        value="(Press Load State)",
                    )  # Updated placeholder
                    # --- NEW Load Button ---
                    load_state_button = gr.Button("Load Agent State", variant="primary")

                with gr.Accordion("Task Status", open=True):
                    # ... (Task DataFrames remain the same structure, content refreshed by button) ...
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
                    # ... (Memory components remain the same structure, content refreshed by button) ...
                    state_memory_summary = gr.Markdown(
                        "Memory Summary\n(Press Load State)"
                    )  # Updated placeholder
                    with gr.Column():
                        gr.Markdown("##### Task-Specific Memories")
                        with gr.Row(scale=1):
                            state_task_memory_select = gr.Dropdown(
                                label="Select Task ID (Completed/Failed)",
                                choices=[
                                    ("Select Task ID...", None)
                                ],  # Use None value for placeholder
                                value=None,  # Default to placeholder
                                # type="value", # No longer needed with None value
                                interactive=True,  # Make dropdown interactive after load
                            )
                        with gr.Row(scale=1):
                            # --- MODIFIED: Added Relative Time column header ---
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
                            # --- MODIFIED: Added Relative Time column header ---
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

                # Define outputs list for the state tab components (used by Load button)
                state_tab_outputs = [
                    state_identity,
                    state_pending_tasks,
                    state_inprogress_tasks,
                    state_completed_tasks,
                    state_failed_tasks,
                    state_memory_summary,
                    state_task_memory_select,  # Dropdown component
                    state_task_memory_display,  # Task memory dataframe
                    state_general_memory_display,  # General memory dataframe
                ]

                # --- Connect NEW Load State button ---
                load_state_button.click(
                    fn=refresh_agent_state_display,
                    inputs=None,
                    outputs=state_tab_outputs,  # Update all state tab components
                )

                # Connect dropdown change event (independent of load button)
                state_task_memory_select.change(
                    fn=update_task_memory_display,
                    inputs=[state_task_memory_select],
                    outputs=[state_task_memory_display],
                )
                # Connect refresh button for general memory (independent of load button)
                refresh_general_mem_btn.click(
                    fn=update_general_memory_display,
                    inputs=None,
                    outputs=[state_general_memory_display],
                )

        # --- Global Timer (Only for Monitor Tab - Unchanged) ---
        timer = gr.Timer(0.5)  # Update interval

        # --- MODIFIED: Update Function ONLY for Monitor Tab (Unchanged logic, just structure) ---
        def update_monitor_and_feedback():
            """Handles periodic UI updates ONLY for the Monitor tab and suggestion feedback."""
            global last_monitor_state

            # --- Initialization / Error Fallback Setup ---
            if last_monitor_state is None:
                if agent_instance:
                    try:
                        last_monitor_state = update_monitor_ui()
                    except Exception as init_e:
                        log.error(f"Failed to get initial monitor state: {init_e}")
                        last_monitor_state = (
                            "Error: Init Failed",
                        ) * 6  # 6 outputs in monitor_outputs
                else:
                    last_monitor_state = ("Error: Agent Offline",) * 6

            # --- Default return values are the last known state ---
            monitor_updates_to_return = last_monitor_state

            if agent_instance and agent_instance._is_running.is_set():
                log.debug("Agent running, updating Monitor tab...")
                try:
                    current_monitor_state = update_monitor_ui()
                    last_monitor_state = current_monitor_state  # Store
                    monitor_updates_to_return = current_monitor_state  # Use fresh
                except Exception as e:
                    log.exception("Error during monitor update (running)")
                    # Fallback: monitor_updates_to_return remains last_monitor_state
            else:
                # Agent paused or stopped, still use last known state for display
                log.debug("Agent paused/stopped, using cached monitor state.")
                monitor_updates_to_return = last_monitor_state

            # --- Final Assembly & Checks ---
            expected_len = 6  # Timer targets 6 outputs now
            if (
                not isinstance(monitor_updates_to_return, tuple)
                or len(monitor_updates_to_return) != expected_len
            ):
                log.error(
                    f"Monitor update tuple structure mismatch. Expected {expected_len} elements. Got {len(monitor_updates_to_return) if isinstance(monitor_updates_to_return, tuple) else type(monitor_updates_to_return)}"
                )
                monitor_updates_to_return = ("Error: Struct Mismatch",) * expected_len

            # Return only the monitor components, excluding the feedback box
            return monitor_updates_to_return

        # --- Connect the timer ONLY to the monitor update function and its outputs ---
        # --- MODIFIED: Timer no longer updates suggestion_feedback_box ---
        monitor_timer_outputs = [
            monitor_current_task,
            monitor_log,
            monitor_memory,
            monitor_web_content,
            monitor_status_bar,
            monitor_final_answer,
        ]
        timer.tick(
            fn=update_monitor_and_feedback,
            inputs=None,
            outputs=monitor_timer_outputs,  # Link ONLY to monitor outputs
        )


# --- Launch the App & Background Thread ---
if __name__ == "__main__":
    # ... (rest of the launch code unchanged) ...
    try:
        os.makedirs(config.SUMMARY_FOLDER, exist_ok=True)
        log.info(f"Summary directory: {config.SUMMARY_FOLDER}")
    except Exception as e:
        log.error(f"Could not create summary dir: {e}")
    log.info("Launching Gradio App Interface...")
    if agent_instance:
        log.info("Agent background processing thread started and paused.")
        try:
            # Initial population of state tab components will happen on the first timer tick
            log.info("UI defined. Launching Gradio server...")

            demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
        except Exception as e:
            log.critical(f"Gradio launch failed: {e}", exc_info=True)
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
