# autonomous_agent/chat_app.py
import gradio as gr
import datetime
import json
import traceback
from typing import List, Tuple, Optional, Dict, Any # Keep Tuple for now if needed elsewhere? No, change to Dict

# --- Project Imports ---
# ... (imports remain the same) ...
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
    format='[%(asctime)s] [%(levelname)s][CHAT_APP] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger("ChatApp")

# --- Global Variables / Setup ---
# (Initialization remains the same)
# ...

# --- Helper Functions ---
# (format_memories_for_display remains the same)
def format_memories_for_display(memories: List[Dict[str, Any]]) -> str:
    # ... (implementation unchanged) ...
    if not memories: return "No relevant memories found for this turn."
    output = "🧠 **Relevant Memories:**\n\n"; i = 0
    for mem in memories[:5]:
        content_snippet = mem.get('content', 'N/A').replace('\n', ' ')[:150]
        meta_type = mem.get('metadata', {}).get('type', 'memory')
        dist_str = f"{mem.get('distance', -1.0):.3f}" if mem.get('distance') is not None else "N/A"
        output += f"**{i+1}. Type:** {meta_type} (Dist: {dist_str})\n"
        output += f"   Content: _{content_snippet}..._\n\n"; i+=1
    return output

# (_should_generate_task remains the same)
def _should_generate_task(user_msg: str, assistant_response: str) -> bool:
    # ... (implementation unchanged) ...
    log.info("Evaluating if task generation is warranted...")
    prompt = f"""Analyze the following chat interaction. Does the user's message OR the assistant's response strongly imply a need for a background task (e.g., deep research, complex analysis, fetching large data) beyond a simple chat answer? Answer ONLY "YES" or "NO".\n\nUser: {user_msg}\nAssistant: {assistant_response}"""
    eval_model = os.environ.get("OLLAMA_EVAL_MODEL", config.OLLAMA_CHAT_MODEL)
    log.debug(f"Asking {eval_model} to evaluate task need...")
    response = call_ollama_api(prompt=prompt, model=eval_model, timeout=30)
    decision = response and "yes" in response.strip().lower()
    log.info(f"Task evaluation result: {'YES' if decision else 'NO'} (Response: '{response}')")
    return decision


# --- Gradio Chat Logic (MODIFIED HISTORY HANDLING) ---
def chat_response(
    message: str,
    history: List[Dict[str, str]], # <<< HISTORY TYPE CHANGED to List[Dict]
) -> Tuple[List[Dict[str, str]], str, str, str]: # <<< RETURN TYPE HINT CHANGED
    """Handles user input, interacts with LLM, updates memory, conditionally triggers tasks."""
    memory_display_text = "Processing..."
    task_display_text = "(No task generated this turn)"

    if not message: return history, "No input provided.", task_display_text, ""
    if not agent_memory or not task_queue or not agent_instance:
         # Append error in the new format
         history.append({"role": "user", "content": message})
         history.append({"role": "assistant", "content": "**ERROR:** Backend components not initialized."})
         return history, "Error: Backend components failed.", task_display_text, ""

    log.info(f"User message received: '{message}'")
    # Append user message in the new format
    history.append({"role": "user", "content": message})

    try:
        # 1. Retrieve Memories
        # Construct context from the new history format
        history_context_list = []
        for turn in history[-3:]: # Look at last 3 turns (user + potential assistant)
             role = turn.get("role", "unknown")
             content = turn.get("content", "")
             history_context_list.append(f"{role.capitalize()}: {content}")
        history_context = "\n".join(history_context_list)

        memory_query = f"Context relevant to user query: '{message}' based on recent chat:\n{history_context}"
        relevant_memories = agent_memory.retrieve_raw_candidates(query=memory_query, n_results=7)
        memory_display_text = format_memories_for_display(relevant_memories)
        log.info(f"Retrieved {len(relevant_memories)} memories for chat context.")

        # 2. Construct LLM Prompt using the new history format
        system_prompt = "You are a helpful AI assistant. Answer the user's query using chat history and relevant memories."
        memories_for_prompt = "\n".join([f"- {mem['content']}" for mem in relevant_memories[:3]])

        # Format history for the LLM prompt (simple User/Assistant lines)
        history_for_prompt_list = []
        for turn in history: # Iterate through the dict list
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            history_for_prompt_list.append(f"{role.capitalize()}: {content}")
        history_for_prompt = "\n".join(history_for_prompt_list)


        prompt = f"{system_prompt}\n\n## Relevant Memories:\n{memories_for_prompt if memories_for_prompt else 'None'}\n\n## Chat History:\n{history_for_prompt}\n\n## Current Query:\nUser: {message}\nAssistant:" # LLM completes after Assistant:

        # 3. Call LLM for Chat Response
        log.info(f"Asking {config.OLLAMA_CHAT_MODEL} for chat response...")
        response_text = call_ollama_api(prompt=prompt, model=config.OLLAMA_CHAT_MODEL)
        if not response_text: response_text = "Sorry, error generating response."; log.error("LLM call failed.")

        # Append assistant response in the new format
        history.append({"role": "assistant", "content": response_text})

        # 4. Add Chat Interaction to Agent Memory (logic unchanged)
        log.info("Adding chat interaction to agent memory...")
        agent_memory.add_memory(content=f"User query: {message}", metadata={"type": "chat_user_query"})
        agent_memory.add_memory(content=f"Assistant response: {response_text}", metadata={"type": "chat_assistant_response"})

        # 5. Conditionally Generate Task & Save QLoRA datapoint
        if _should_generate_task(message, response_text):
            log.info("Interaction warrants task generation. Calling generate_new_tasks...")
            first_task_generated_desc = agent_instance.generate_new_tasks(
                max_new_tasks=1,
                last_user_message=message,
                last_assistant_response=response_text
            )
            if first_task_generated_desc:
                task_display_text = f"✅ Task Generated:\n\"{first_task_generated_desc}\""
                log.info(f"Task generated and displayed: {first_task_generated_desc[:60]}...")
                agent_instance._save_qlora_datapoint(
                    source_type="chat_task_generation",
                    instruction="The user's message required a background task. Respond appropriately and indicate a task will be created.",
                    input_context=message,
                    output=response_text
                )
                # Add notification as a separate assistant message
                task_notification = f"*Okay, I've created a background task to: \"{first_task_generated_desc[:100]}...\"*"
                history.append({"role": "assistant", "content": task_notification}) # Append notification
            else:
                 task_display_text = "(Evaluation suggested task, but generation failed or returned none)"
                 log.info("Task generation was warranted but no task was added.")
        else:
             log.info("Interaction evaluated, task generation not warranted.")
             task_display_text = "(No new task warranted this turn)"

        # 6. Return final updates
        return history, memory_display_text, task_display_text, "" # history, memory_panel, task_panel, clear_textbox

    except Exception as e:
        log.exception(f"Error during chat processing: {e}")
        error_message = f"An internal error occurred: {e}"
        # Append error as assistant message
        history.append({"role": "assistant", "content": error_message})
        return history, f"Error:\n```\n{traceback.format_exc()}\n```", task_display_text, ""


# --- Gradio UI Definition ---
log.info("Defining Gradio UI...")
with gr.Blocks(theme=gr.themes.Glass(), title="Agent Chat Interface") as demo:
    gr.Markdown("# Chat with the Autonomous Agent")
    gr.Markdown("Interact with the agent, view relevant memories, see generated tasks.")

    with gr.Tabs():
        # --- Monitor Tab ---
        with gr.TabItem("Agent Monitor"):
            # ... (Monitor Tab UI definition unchanged) ...
            gr.Markdown("Monitor the agent's autonomous processing steps.")
            with gr.Row():
                process_step_btn = gr.Button("Process One Task Step", variant="primary")
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Current Task")
                    monitor_current_task = gr.Markdown("(No task active)")
                    gr.Markdown("### Step Log")
                    monitor_log = gr.Textbox(label="Last Step Log", lines=15, interactive=False, autoscroll=True)
                with gr.Column(scale=1):
                    gr.Markdown("### Recent Memories")
                    monitor_memory = gr.Markdown("(Memories will appear here)")
                    gr.Markdown("### Last Web Content")
                    monitor_web_content = gr.Textbox(label="Last Web Content Fetched", lines=10, interactive=False)

            monitor_outputs = [monitor_current_task, monitor_log, monitor_memory, monitor_web_content, monitor_log]
            process_step_btn.click(fn=process_one_step_ui, inputs=[], outputs=monitor_outputs, queue=False)


        # --- Chat Tab ---
        with gr.TabItem("Chat"):
            gr.Markdown("Interact directly with the agent.")
            with gr.Row():
                with gr.Column(scale=3): # Main chat area
                    # --- FIX: Add type="messages" ---
                    chat_chatbot = gr.Chatbot(
                        label="Conversation",
                        bubble_full_width=False,
                        height=550,
                        show_copy_button=True,
                        type="messages" # <<< SET TYPE HERE
                    )
                    # -----------------------------
                    chat_task_panel = gr.Textbox(label="💡 Last Generated Task (Chat)", value="(No task generated yet)", lines=3, interactive=False, show_copy_button=True)
                    with gr.Row():
                        chat_msg_input = gr.Textbox(label="Your Message", placeholder="Type message and press Enter or click Send...", lines=3, scale=5, container=False)
                        chat_send_button = gr.Button("Send", variant="primary", scale=1)
                with gr.Column(scale=1): # Side panel for memories
                    gr.Markdown("### Relevant Memories (Chat)")
                    chat_memory_panel = gr.Markdown(value="Memory context will appear here.", label="Memory Context")

            # --- FIX: Input/Output types match Chatbot type ---
            # Inputs: Textbox, Chatbot (which now uses List[Dict])
            # Outputs: Chatbot, Markdown, Textbox, Textbox (clear input)
            chat_inputs = [chat_msg_input, chat_chatbot]
            chat_outputs = [chat_chatbot, chat_memory_panel, chat_task_panel, chat_msg_input]

            chat_send_button.click(fn=chat_response, inputs=chat_inputs, outputs=chat_outputs, queue=True)
            chat_msg_input.submit(fn=chat_response, inputs=chat_inputs, outputs=chat_outputs, queue=True)


# --- Launch the App ---
if __name__ == "__main__":
    # ... (Ensure summary folder exists - unchanged) ...
    try:
        os.makedirs(config.SUMMARY_FOLDER, exist_ok=True)
        log.info(f"Summary directory ensured at: {config.SUMMARY_FOLDER}")
    except Exception as e:
        log.error(f"Could not create summary directory {config.SUMMARY_FOLDER}: {e}")

    log.info("Launching Gradio App Interface...")
    demo.launch(server_name="0.0.0.0", share=False)
    log.info("Gradio App stopped.")