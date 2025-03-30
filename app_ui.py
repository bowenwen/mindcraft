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

# --- Project Imports ---
import config
from llm_utils import call_ollama_api
from memory import AgentMemory, setup_chromadb
from task_manager import TaskQueue
from data_structures import Task
from agent import AutonomousAgent
import chromadb

# --- Logging Setup ---
import logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='[%(asctime)s] [%(levelname)s][%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger("AppUI")

# --- Global Variables / Setup ---
log.info("Initializing components for Gradio App...")
agent_instance: Optional[AutonomousAgent] = None
try:
    mem_collection = setup_chromadb()
    if mem_collection is None: raise RuntimeError("Database setup failed.")
    agent_instance = AutonomousAgent(memory_collection=mem_collection)
    log.info("App components initialized successfully.")
except Exception as e:
    log.critical(f"Fatal error during App component initialization: {e}", exc_info=True)
    print(f"\n\nFATAL ERROR: Could not initialize agent components: {e}\n", file=sys.stderr)
    agent_instance = None

# --- Calculate initial status BEFORE defining the UI ---
initial_status_text = "Error: Agent not initialized." # Default if agent fails
if agent_instance:
    # Start and pause the agent to get the correct initial 'paused' state for the UI
    log.info("Setting initial agent state to paused for UI...")
    agent_instance.start_autonomous_loop()
    agent_instance.pause_autonomous_loop() # Start paused
    # Wait briefly to ensure state update propagates if needed (usually not necessary here)
    # time.sleep(0.1)
    initial_state = agent_instance.get_ui_update_state()
    initial_status_text = f"Agent Status: {initial_state.get('status', 'paused')} @ {initial_state.get('timestamp', 'N/A')}"
    log.info(f"Calculated initial status text for UI: {initial_status_text}")


# --- Helper Functions ---
def format_memories_for_display(memories: List[Dict[str, Any]]) -> str:
    """Formats retrieved memories into a readable markdown string for UI panels."""
    if not memories:
        return "No relevant memories found."
    output = "🧠 **Recent/Relevant Memories:**\n\n"
    for i, mem in enumerate(memories[:5]): # Limit display
        content_snippet = mem.get('content', 'N/A').replace('\n', ' ')[:150]
        meta_type = mem.get('metadata', {}).get('type', 'memory')
        dist_str = f"{mem.get('distance', -1.0):.3f}" if mem.get('distance') is not None else "N/A"
        output += f"**{i+1}. Type:** {meta_type} (Dist: {dist_str})\n"
        output += f"   Content: _{content_snippet}..._\n\n"
    return output

# --- Functions for Monitor Tab ---
def start_agent_processing():
    """Called by Gradio button to start/resume the background loop."""
    if agent_instance:
        log.info("UI triggered: Start/Resume Agent Processing")
        agent_instance.start_autonomous_loop()
        # Update status immediately for better feedback
        agent_state = agent_instance.get_ui_update_state()
        status_text = f"Agent Status: {agent_state.get('status', 'running')} @ {agent_state.get('timestamp', 'N/A')}"
        return status_text
    else:
        log.error("UI trigger failed: Agent not initialized.")
        return "ERROR: Agent not initialized."

def pause_agent_processing():
    """Called by Gradio button to pause the background loop."""
    if agent_instance:
        log.info("UI triggered: Pause Agent Processing")
        agent_instance.pause_autonomous_loop()
        # Update status immediately for better feedback
        # Give it a moment for the loop to potentially update its status if running
        time.sleep(0.2)
        agent_state = agent_instance.get_ui_update_state()
        status_text = f"Agent Status: {agent_state.get('status', 'paused')} @ {agent_state.get('timestamp', 'N/A')}"
        return status_text
    else:
        log.error("UI trigger failed: Agent not initialized.")
        return "ERROR: Agent not initialized."

# --- Function to update Monitor UI (returns tuple) ---
def update_monitor_ui() -> Tuple[str, str, str, str, str, str]:
    """
    Called periodically by Gradio timer. Fetches agent state and returns
    updates for multiple components as a tuple matching the 'outputs' list order.
    """
    current_task_display = "(Error retrieving state)"; step_log_output = "(Error retrieving state)"
    memory_display = "(Error retrieving state)"; web_content_display = "(Error retrieving state)"
    status_bar_text = "Error: Cannot get state"; final_answer_display = "(No final answer generated recently)"

    if not agent_instance:
        status_bar_text = "Error: Agent not initialized."
        current_task_display = "Agent Not Initialized"
        step_log_output = "Agent Not Initialized"
        memory_display = "Agent Not Initialized"
        web_content_display = "Agent Not Initialized"
        final_answer_display = "Agent Not Initialized"
        return current_task_display, step_log_output, memory_display, web_content_display, status_bar_text, final_answer_display

    try:
        ui_state = agent_instance.get_ui_update_state()

        current_task_id = ui_state.get('current_task_id', 'None')
        task_status = ui_state.get('status', 'unknown')
        current_task_desc = ui_state.get('current_task_desc', 'N/A')
        step_log_output = ui_state.get('log', '(No recent log)')
        recent_memories = ui_state.get('recent_memories', [])
        last_browse_content = ui_state.get('last_web_content', '(No recent web browse)')
        final_answer = ui_state.get('final_answer')

        current_task_display = f"**ID:** {current_task_id}\n**Status:** {task_status}\n**Desc:** {current_task_desc}"
        memory_display = format_memories_for_display(recent_memories)

        web_content_limit = 2000
        if last_browse_content and len(last_browse_content) > web_content_limit:
            web_content_display = last_browse_content[:web_content_limit] + "\n\n... (content truncated)"
        else:
            web_content_display = last_browse_content if last_browse_content else "(No recent web browse)"

        status_bar_text = f"Agent Status: {task_status} @ {ui_state.get('timestamp', 'N/A')}"

        if final_answer:
            final_answer_display = final_answer
        else:
            final_answer_display = "(No final answer generated recently)" # Explicitly set default if None


    except Exception as e:
        log.exception("Error in update_monitor_ui")
        error_msg = f"ERROR updating UI:\n{traceback.format_exc()}"
        step_log_output = error_msg
        status_bar_text = "Error updating UI"
        current_task_display = "Error"
        memory_display = "Error"
        web_content_display = "Error"
        final_answer_display = "Error"

    return current_task_display, step_log_output, memory_display, web_content_display, status_bar_text, final_answer_display


# --- Functions for Chat Tab ---
def _should_generate_task(user_msg: str, assistant_response: str) -> bool:
    if not agent_instance: return False
    log.info("Evaluating if task generation is warranted for chat turn...")
    eval_model = os.environ.get("OLLAMA_EVAL_MODEL", config.OLLAMA_CHAT_MODEL)
    prompt = f"""Analyze the following chat interaction. Does the user's message OR the assistant's response strongly imply a need for a complex background task (e.g., deep research, multi-step analysis, external actions) that goes beyond a direct chat answer? Answer ONLY with "YES" or "NO".\n\nUser: {user_msg}\nAssistant: {assistant_response}"""
    log.debug(f"Asking {eval_model} to evaluate task need...")
    response = call_ollama_api(prompt=prompt, model=eval_model, base_url=config.OLLAMA_BASE_URL, timeout=30)
    decision = response and "yes" in response.strip().lower()
    log.info(f"Task evaluation result: {'YES' if decision else 'NO'} (LLM Response: '{response}')")
    return decision

def chat_response(
    message: str,
    history: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], str, str, str]: # history, memory_panel, task_panel, msg_input
    """Handles user input, interacts with LLM (aware of agent's current task), updates memory, conditionally triggers tasks."""
    memory_display_text = "Processing..."; task_display_text = "(No task generated this turn)"
    if not message: return history, "No input provided.", task_display_text, ""
    if not agent_instance:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "**ERROR:** Backend agent components not initialized."})
        return history, "Error: Backend components failed.", task_display_text, ""

    log.info(f"User message received: '{message}'");
    history.append({"role": "user", "content": message})

    try:
        agent_state = agent_instance.get_ui_update_state()
        agent_task_id = agent_state.get("current_task_id")
        agent_task_desc = agent_state.get("current_task_desc", "N/A")
        agent_status = agent_state.get("status", "unknown")

        agent_activity_context = "Currently idle."
        current_task_for_context = None # Store task object if found
        if agent_task_id:
             current_task_for_context = agent_instance.task_queue.get_task(agent_task_id)
             if current_task_for_context:
                  agent_task_desc = current_task_for_context.description # Use potentially more current desc
             else: # Task ID exists in state but not in queue (edge case?)
                  log.warning(f"Task ID {agent_task_id} found in state but not in queue during chat.")
                  agent_task_desc += " (Task details missing?)"

        if agent_status == "running" and agent_task_id:
            agent_activity_context = f"Currently working on background Task {agent_task_id}: '{agent_task_desc}'. Status: {agent_status}."
        elif agent_status == "paused":
            if agent_task_id:
                agent_activity_context = f"Currently paused, but the active background task is Task {agent_task_id}: '{agent_task_desc}'."
            else:
                agent_activity_context = "Currently paused and idle (no active background task)."
        elif agent_status in ["shutdown", "critical_error"]:
            agent_activity_context = f"The background agent is currently in a '{agent_status}' state."
        # If idle, the default "Currently idle." is kept.

        log.debug(f"Agent activity context for chat prompt: {agent_activity_context}")

        history_context_list = [f"{turn.get('role','?').capitalize()}: {turn.get('content','')}" for turn in history[-4:-1]] # Last 3 turns
        history_context_str = "\n".join(history_context_list)
        # Use a slightly different query/task description for memory retrieval in chat
        memory_query = f"Chat context relevant to user query: '{message}'\nUser History:\n{history_context_str}\nAgent is currently: {agent_activity_context}"
        relevant_memories = agent_instance.retrieve_and_rerank_memories(
            query=memory_query,
            task_description="Responding to user in chat interface, considering agent's background activity.", # More specific desc
            context=history_context_str + f"\nAgent Activity: {agent_activity_context}", # Add activity to context for reranking
            n_final=3
        )
        memory_display_text = format_memories_for_display(relevant_memories)
        log.info(f"Retrieved and re-ranked {len(relevant_memories)} memories for chat context.")

        system_prompt = "You are a helpful AI assistant integrated with an autonomous agent. Answer the user's query conversationally. Consider the chat history, relevant memories, and be aware of your own current background activity (if any), incorporating it into your response where relevant."

        memories_for_prompt = "\n".join([f"- {mem['content']}" for mem in relevant_memories])
        history_for_prompt = "\n".join([f"{t.get('role','?').capitalize()}: {t.get('content','')}" for t in history])

        prompt = f"{system_prompt}\n\n"
        prompt += f"## Current Agent Background Activity:\n{agent_activity_context}\n\n"
        prompt += f"## Relevant Memories:\n{memories_for_prompt if memories_for_prompt else 'None provided.'}\n\n"
        prompt += f"## Chat History:\n{history_for_prompt}\n\n"
        prompt += f"## Current Query:\nUser: {message}\nAssistant:"

        log.info(f"Asking {config.OLLAMA_CHAT_MODEL} for chat response (with agent activity context)...")
        response_text = call_ollama_api(prompt=prompt, model=config.OLLAMA_CHAT_MODEL, base_url=config.OLLAMA_BASE_URL, timeout=config.OLLAMA_TIMEOUT)
        if not response_text: response_text = "Sorry, I encountered an error generating a response."; log.error("LLM call failed for chat response.")

        history.append({"role": "assistant", "content": response_text})

        log.info("Adding chat interaction to agent long-term memory...")
        agent_instance.memory.add_memory(content=f"User query: {message}", metadata={"type": "chat_user_query"})
        agent_instance.memory.add_memory(content=f"Assistant response: {response_text}", metadata={"type": "chat_assistant_response", "agent_activity_at_time": agent_activity_context})

        if _should_generate_task(message, response_text):
            log.info("Interaction warrants task generation. Calling generate_new_tasks...")
            first_task_generated_desc = agent_instance.generate_new_tasks(max_new_tasks=1, last_user_message=message, last_assistant_response=response_text)
            if first_task_generated_desc:
                task_display_text = f"✅ Task Generated:\n\"{first_task_generated_desc}\""
                log.info(f"Task generated and displayed: {first_task_generated_desc[:60]}...")
                notification = f"\n\n*(Okay, I've created a background task for: \"{first_task_generated_desc}\". I'll work on it when I can.)*"
                history[-1]["content"] += notification # Append to the assistant's last message

                agent_instance._save_qlora_datapoint(
                    source_type="chat_task_generation",
                    instruction="User interaction led to this task. Respond to the user and confirm task creation.",
                    input_context=f"User: {message}\nAgent Activity: {agent_activity_context}",
                    output=f"{response_text}{notification}"
                )
            else: task_display_text = "(Evaluation suggested task, but none generated/added)"; log.info("Task generation warranted but no task added.")
        else: log.info("Interaction evaluated, task generation not warranted."); task_display_text = "(No new task warranted this turn)"

        return history, memory_display_text, task_display_text, ""

    except Exception as e:
        log.exception(f"Error during chat processing: {e}")
        error_message = f"An internal error occurred processing your message: {e}";
        history.append({"role": "assistant", "content": error_message})
        return history, f"Error:\n```\n{traceback.format_exc()}\n```", task_display_text, ""


# --- Gradio UI Definition ---
log.info("Defining Gradio UI...")
if agent_instance is None:
     log.critical("Agent instance failed to initialize. Cannot launch UI.")
     with gr.Blocks() as demo:
          gr.Markdown("# Fatal Error")
          gr.Markdown("The agent backend failed to initialize. Check logs and configuration then restart the application.")
else:
     # Define the main UI if agent is initialized
     with gr.Blocks(theme=gr.themes.Glass(), title="Autonomous Agent Interface") as demo:
        gr.Markdown("# Autonomous Agent Control Center & Chat")

        with gr.Tabs():
            # --- Monitor Tab ---
            with gr.TabItem("Agent Monitor"):
                gr.Markdown("Monitor and control the agent's autonomous processing.")
                with gr.Row():
                    start_resume_btn = gr.Button("Start / Resume", variant="primary")
                    pause_btn = gr.Button("Pause")
                # Define status bar WITH the calculated initial value
                monitor_status_bar = gr.Textbox(
                    label="Agent Status",
                    value=initial_status_text, # <<< SET INITIAL VALUE HERE
                    interactive=False
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        monitor_current_task = gr.Markdown("(Agent Initializing)")
                        monitor_log = gr.Textbox(label="Last Step Log", lines=10, interactive=False, autoscroll=True)
                        monitor_final_answer = gr.Textbox(label="Last Final Answer", lines=5, interactive=False, show_copy_button=True)
                    with gr.Column(scale=1):
                        monitor_memory = gr.Markdown("Recent Memories\n(Agent Initializing)")
                        monitor_web_content = gr.Textbox(label="Last Web Content Fetched", lines=10, interactive=False, show_copy_button=True)

                monitor_outputs = [
                    monitor_current_task, monitor_log, monitor_memory,
                    monitor_web_content, monitor_status_bar, monitor_final_answer
                ]

                start_resume_btn.click(fn=start_agent_processing, inputs=None, outputs=monitor_status_bar)
                pause_btn.click(fn=pause_agent_processing, inputs=None, outputs=monitor_status_bar)

                timer = gr.Timer(5)
                timer.tick(fn=update_monitor_ui, inputs=None, outputs=monitor_outputs)


            # --- Chat Tab ---
            with gr.TabItem("Chat"):
                gr.Markdown("Interact directly with the agent. It is aware of its background tasks.")
                with gr.Row():
                    with gr.Column(scale=3):
                        chat_chatbot = gr.Chatbot(
                            label="Conversation", bubble_full_width=False, height=550,
                            show_copy_button=True, type="messages"
                        )
                        chat_task_panel = gr.Textbox(label="💡 Last Generated Task (Chat)", value="(No task generated yet)", lines=3, interactive=False, show_copy_button=True)
                        with gr.Row():
                            chat_msg_input = gr.Textbox(label="Your Message", placeholder="Type message and press Enter or click Send...", lines=3, scale=5, container=False)
                            chat_send_button = gr.Button("Send", variant="primary", scale=1)
                    with gr.Column(scale=1):
                        chat_memory_panel = gr.Markdown(value="Relevant Memories (Chat)\nMemory context will appear here.", label="Memory Context")

                chat_inputs = [chat_msg_input, chat_chatbot]
                chat_outputs = [chat_chatbot, chat_memory_panel, chat_task_panel, chat_msg_input]

                chat_send_button.click(fn=chat_response, inputs=chat_inputs, outputs=chat_outputs, queue=True)
                chat_msg_input.submit(fn=chat_response, inputs=chat_inputs, outputs=chat_outputs, queue=True)


# --- Launch the App & Background Thread ---
if __name__ == "__main__":
    try: os.makedirs(config.SUMMARY_FOLDER, exist_ok=True); log.info(f"Summary directory ensured: {config.SUMMARY_FOLDER}")
    except Exception as e: log.error(f"Could not create summary directory {config.SUMMARY_FOLDER}: {e}")

    log.info("Launching Gradio App Interface...")

    if agent_instance:
        # Agent loop is ALREADY started and paused above where initial_status_text is calculated
        log.info("Agent background processing thread already started and paused.")
        # ---- REMOVED the problematic try/except block ----

        try:
            demo.launch(server_name="0.0.0.0", server_port=7860, share=False) # Blocks main thread
        except Exception as e:
            log.critical(f"Gradio demo.launch() failed: {e}", exc_info=True)
            log.info("Requesting agent shutdown due to Gradio launch error...")
            agent_instance.shutdown()
            sys.exit("Gradio launch failed.")
    else:
        # If agent init failed, Gradio defined a simple error message block above
        log.warning("Agent initialization failed. Launching minimal error UI.")
        try:
            with gr.Blocks() as error_demo:
                 gr.Markdown("# Fatal Error")
                 gr.Markdown("The agent backend failed to initialize. Check logs and configuration then restart the application.")
            error_demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
        except Exception as e:
             log.critical(f"Gradio demo.launch() failed even for error UI: {e}", exc_info=True)
             sys.exit("Gradio launch failed.")


    # --- Cleanup ---
    log.info("Gradio App stopped. Requesting agent shutdown...")
    if agent_instance:
        agent_instance.shutdown()
    log.info("Shutdown complete.")
    print("\n--- Autonomous Agent App End ---")