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
    agent_instance = None


# --- Helper Functions ---
# (format_memories_for_display - unchanged)
def format_memories_for_display(memories: List[Dict[str, Any]]) -> str:
    if not memories: return "No relevant memories found."
    output = "🧠 **Recent/Relevant Memories:**\n\n"; i = 0
    for mem in memories[:5]:
        content_snippet = mem.get('content', 'N/A').replace('\n', ' ')[:150]
        meta_type = mem.get('metadata', {}).get('type', 'memory')
        dist_str = f"{mem.get('distance', -1.0):.3f}" if mem.get('distance') is not None else "N/A"
        output += f"**{i+1}. Type:** {meta_type} (Dist: {dist_str})\n"; output += f"   Content: _{content_snippet}..._\n\n"; i+=1
    return output

# --- Functions for Monitor Tab ---
# (start_agent_processing, pause_agent_processing - unchanged)
def start_agent_processing():
    if agent_instance: log.info("UI triggered: Start/Resume Agent Processing"); agent_instance.start_autonomous_loop(); return "Agent loop started/resumed."
    else: log.error("UI trigger failed: Agent not initialized."); return "ERROR: Agent not initialized."
def pause_agent_processing():
    if agent_instance: log.info("UI triggered: Pause Agent Processing"); agent_instance.pause_autonomous_loop(); return "Agent loop pause requested."
    else: log.error("UI trigger failed: Agent not initialized."); return "ERROR: Agent not initialized."


# --- Function to update Monitor UI (returns tuple, ADDED final_answer output) ---
def update_monitor_ui() -> Tuple[str, str, str, str, str, str]: # Added one more string for final answer
    """
    Called periodically by Gradio timer. Fetches agent state and returns
    updates for multiple components as a tuple matching the 'outputs' list order.
    """
    # Default values
    current_task_display = "(Error retrieving state)"; step_log_output = "(Error retrieving state)"
    memory_display = "(Error retrieving state)"; web_content_display = "(Error retrieving state)"
    status_bar_text = "Error: Cannot get state"; final_answer_display = "(No final answer generated recently)" # Default for new panel

    if not agent_instance:
        status_bar_text = "Error: Agent not initialized."
        return current_task_display, step_log_output, memory_display, web_content_display, status_bar_text, final_answer_display # Return defaults

    try:
        ui_state = agent_instance.get_ui_update_state()

        # Safely extract data
        current_task_id = ui_state.get('current_task_id', 'None'); task_status = ui_state.get('status', 'unknown')
        current_task_desc = ui_state.get('current_task_desc', 'N/A'); step_log_output = ui_state.get('log', '(No recent log)')
        recent_memories = ui_state.get('recent_memories', []); last_browse_content = ui_state.get('last_web_content', '(No recent web browse)')
        final_answer = ui_state.get('final_answer') # Get the final answer state

        # Format standard UI elements
        current_task_display = f"**ID:** {current_task_id}\n**Status:** {task_status}\n**Desc:** {current_task_desc}"
        memory_display = format_memories_for_display(recent_memories)
        web_content_limit = 2000
        if last_browse_content and len(last_browse_content) > web_content_limit: web_content_display = last_browse_content[:web_content_limit] + "\n\n... (content truncated)"
        else: web_content_display = last_browse_content
        status_bar_text = f"Agent Status: {task_status} @ {ui_state.get('timestamp', 'N/A')}"

        # Format the final answer display
        if final_answer:
            final_answer_display = final_answer
            # Optionally clear the state after displaying once?
            # agent_instance._update_ui_state(final_answer=None)
        # else: keep default message

    except Exception as e:
        log.exception("Error in update_monitor_ui")
        error_msg = f"ERROR updating UI:\n{traceback.format_exc()}"; step_log_output = error_msg
        status_bar_text = "Error updating UI"; current_task_display = "Error"; memory_display = "Error"; web_content_display = "Error"; final_answer_display = "Error"

    # Return tuple including the final answer
    return current_task_display, step_log_output, memory_display, web_content_display, status_bar_text, final_answer_display


# --- Functions for Chat Tab ---
# (_should_generate_task, chat_response - unchanged)
def _should_generate_task(user_msg: str, assistant_response: str) -> bool:
    if not agent_instance: return False
    log.info("Evaluating if task generation is warranted for chat turn...")
    prompt = f"""Analyze the following chat interaction. Does the user's message OR the assistant's response strongly imply a need for a background task (e.g., deep research, complex analysis, fetching large data) beyond a simple chat answer? Answer ONLY "YES" or "NO".\n\nUser: {user_msg}\nAssistant: {assistant_response}"""
    eval_model = os.environ.get("OLLAMA_EVAL_MODEL", config.OLLAMA_CHAT_MODEL)
    log.debug(f"Asking {eval_model} to evaluate task need...")
    response = call_ollama_api(prompt=prompt, model=eval_model, timeout=30)
    decision = response and "yes" in response.strip().lower()
    log.info(f"Task evaluation result: {'YES' if decision else 'NO'} (Response: '{response}')")
    return decision

def chat_response(
    message: str,
    history: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], str, str, str]:
    memory_display_text = "Processing..."; task_display_text = "(No task generated this turn)"
    if not message: return history, "No input provided.", task_display_text, ""
    if not agent_instance: history.append({"role": "user", "content": message}); history.append({"role": "assistant", "content": "**ERROR:** Backend components not initialized."}); return history, "Error: Backend components failed.", task_display_text, ""
    log.info(f"User message received: '{message}'"); history.append({"role": "user", "content": message})
    try:
        history_context_list = [f"{turn.get('role','?').capitalize()}: {turn.get('content','')}" for turn in history[-4:-1]]
        history_context = "\n".join(history_context_list)
        memory_query = f"Context relevant to user query: '{message}' based on recent chat:\n{history_context}"
        relevant_memories = agent_instance.memory.retrieve_raw_candidates(query=memory_query, n_results=7)
        memory_display_text = format_memories_for_display(relevant_memories)
        log.info(f"Retrieved {len(relevant_memories)} memories for chat context.")
        system_prompt = "You are a helpful AI assistant. Answer the user's query using chat history and relevant memories."
        memories_for_prompt = "\n".join([f"- {mem['content']}" for mem in relevant_memories[:3]])
        history_for_prompt_list = [f"{t.get('role','?').capitalize()}: {t.get('content','')}" for t in history]
        history_for_prompt = "\n".join(history_for_prompt_list)
        prompt = f"{system_prompt}\n\n## Relevant Memories:\n{memories_for_prompt if memories_for_prompt else 'None'}\n\n## Chat History:\n{history_for_prompt}\n\n## Current Query:\nUser: {message}\nAssistant:"
        log.info(f"Asking {config.OLLAMA_CHAT_MODEL} for chat response...")
        response_text = call_ollama_api(prompt=prompt, model=config.OLLAMA_CHAT_MODEL)
        if not response_text: response_text = "Sorry, error generating response."; log.error("LLM call failed.")
        history.append({"role": "assistant", "content": response_text})
        log.info("Adding chat interaction to agent memory...")
        agent_instance.memory.add_memory(content=f"User query: {message}", metadata={"type": "chat_user_query"})
        agent_instance.memory.add_memory(content=f"Assistant response: {response_text}", metadata={"type": "chat_assistant_response"})
        if _should_generate_task(message, response_text):
            log.info("Interaction warrants task generation. Calling generate_new_tasks...")
            first_task_generated_desc = agent_instance.generate_new_tasks(max_new_tasks=1, last_user_message=message, last_assistant_response=response_text)
            if first_task_generated_desc:
                task_display_text = f"✅ Task Generated:\n\"{first_task_generated_desc}\""
                log.info(f"Task generated and displayed: {first_task_generated_desc[:60]}...")
                agent_instance._save_qlora_datapoint(source_type="chat_task_generation", instruction="User interaction led to this task. Respond and confirm task creation.", input_context=f"User: {message}", output=f"{response_text}\n\n[Action: Created background task: {first_task_generated_desc}]")
            else: task_display_text = "(Evaluation suggested task, but none generated/added)"; log.info("Task generation warranted but no task added.")
        else: log.info("Interaction evaluated, task generation not warranted."); task_display_text = "(No new task warranted this turn)"
        return history, memory_display_text, task_display_text, ""
    except Exception as e:
        log.exception(f"Error during chat processing: {e}")
        error_message = f"An internal error occurred: {e}"; history.append({"role": "assistant", "content": error_message})
        return history, f"Error:\n```\n{traceback.format_exc()}\n```", task_display_text, ""


# --- Gradio UI Definition ---
log.info("Defining Gradio UI...")
if agent_instance is None:
     log.critical("Agent instance failed to initialize. Cannot launch UI.")
     with gr.Blocks() as demo: gr.Markdown("# Fatal Error\nAgent backend failed to initialize. Check logs.")
else:
     with gr.Blocks(theme=gr.themes.Glass(), title="Autonomous Agent Interface") as demo:
        gr.Markdown("# Autonomous Agent Control Center & Chat")

        with gr.Tabs():
            # --- Monitor Tab ---
            with gr.TabItem("Agent Monitor"):
                gr.Markdown("Monitor and control the agent's autonomous processing.")
                with gr.Row():
                    start_resume_btn = gr.Button("Start / Resume", variant="primary")
                    pause_btn = gr.Button("Pause")
                monitor_status_bar = gr.Textbox(label="Agent Status", value="Paused (initially)", interactive=False)
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Current Task")
                        monitor_current_task = gr.Markdown("(Agent Paused)")
                        gr.Markdown("### Last Step Log")
                        monitor_log = gr.Textbox(label="Log Output", lines=10, interactive=False, autoscroll=True) # Adjusted lines
                        # --- NEW: Final Answer Panel ---
                        gr.Markdown("### Last Final Answer")
                        monitor_final_answer = gr.Textbox(label="Final Answer Display", lines=5, interactive=False, show_copy_button=True)
                        # ------------------------------
                    with gr.Column(scale=1):
                        gr.Markdown("### Recent Memories")
                        monitor_memory = gr.Markdown("(Memories will appear here)")
                        gr.Markdown("### Last Web Content")
                        monitor_web_content = gr.Textbox(label="Last Web Content Fetched", lines=10, interactive=False, show_copy_button=True)

                # Define list of output components including the new one
                monitor_outputs = [
                    monitor_current_task,
                    monitor_log,
                    monitor_memory,
                    monitor_web_content,
                    monitor_status_bar,
                    monitor_final_answer # <<< ADDED final answer component
                ]

                # Button actions
                start_resume_btn.click(fn=start_agent_processing, inputs=None, outputs=monitor_status_bar)
                pause_btn.click(fn=pause_agent_processing, inputs=None, outputs=monitor_status_bar)

                # Periodic Update using Timer
                timer = gr.Timer(5) # Define Timer component with 5 sec interval
                timer.tick(fn=update_monitor_ui, inputs=None, outputs=monitor_outputs)


            # --- Chat Tab ---
            with gr.TabItem("Chat"):
                # ... (Chat Tab UI definition unchanged) ...
                 gr.Markdown("Interact directly with the agent.")
                 with gr.Row():
                     with gr.Column(scale=3):
                         chat_chatbot = gr.Chatbot(label="Conversation", bubble_full_width=False, height=550, show_copy_button=True, type="messages")
                         chat_task_panel = gr.Textbox(label="💡 Last Generated Task (Chat)", value="(No task generated yet)", lines=3, interactive=False, show_copy_button=True)
                         with gr.Row():
                             chat_msg_input = gr.Textbox(label="Your Message", placeholder="Type message and press Enter or click Send...", lines=3, scale=5, container=False)
                             chat_send_button = gr.Button("Send", variant="primary", scale=1)
                     with gr.Column(scale=1):
                         gr.Markdown("### Relevant Memories (Chat)")
                         chat_memory_panel = gr.Markdown(value="Memory context will appear here.", label="Memory Context")

                 chat_inputs = [chat_msg_input, chat_chatbot]
                 chat_outputs = [chat_chatbot, chat_memory_panel, chat_task_panel, chat_msg_input]
                 chat_send_button.click(fn=chat_response, inputs=chat_inputs, outputs=chat_outputs, queue=True)
                 chat_msg_input.submit(fn=chat_response, inputs=chat_inputs, outputs=chat_outputs, queue=True)


# --- Launch the App & Background Thread ---
if __name__ == "__main__":
    try: os.makedirs(config.SUMMARY_FOLDER, exist_ok=True); log.info(f"Summary directory: {config.SUMMARY_FOLDER}")
    except Exception as e: log.error(f"Could not create summary directory {config.SUMMARY_FOLDER}: {e}")

    log.info("Launching Gradio App Interface...")

    if agent_instance:
        log.info("Starting agent background processing thread (initially paused)...")
        agent_instance.start_autonomous_loop()
        agent_instance.pause_autonomous_loop()
        try:
            demo.launch(server_name="0.0.0.0", share=False) # This blocks
        except Exception as e:
            log.critical(f"Gradio demo.launch() failed: {e}", exc_info=True)
            log.info("Requesting agent shutdown due to Gradio launch error...")
            agent_instance.shutdown()
            sys.exit("Gradio launch failed.")
    else:
        log.critical("Agent initialization failed earlier. Cannot launch UI properly.")
        # Define and launch the minimal error UI if agent_instance is None
        with gr.Blocks() as error_demo:
            gr.Markdown("# Fatal Error")
            gr.Markdown("The agent backend failed to initialize. Check logs and configuration then restart the application.")
        try:
            error_demo.launch(server_name="0.0.0.0", share=False)
        except Exception as e:
             log.critical(f"Gradio demo.launch() failed even for error UI: {e}", exc_info=True)
             sys.exit("Gradio launch failed.")


    # --- Cleanup (runs when Gradio server is stopped) ---
    log.info("Gradio App stopped. Requesting agent shutdown...")
    if agent_instance:
        agent_instance.shutdown() # Signal background thread to stop
    log.info("Shutdown complete.")
    print("\n--- Autonomous Agent App End ---")