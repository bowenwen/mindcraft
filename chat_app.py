# autonomous_agent/chat_app.py
import gradio as gr
import datetime
import json
import traceback
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
    format='[%(asctime)s] [%(levelname)s][CHAT_APP] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger("ChatApp")

# --- Global Variables / Setup ---
log.info("Initializing components for Chat App...")
try:
    mem_collection = setup_chromadb()
    if mem_collection is None: raise RuntimeError("Database setup failed.")
    agent_memory = AgentMemory(mem_collection)
    task_queue = TaskQueue()
    agent_instance = AutonomousAgent(memory_collection=mem_collection)
    log.info("Chat App components initialized successfully.")
except Exception as e:
    log.critical(f"Fatal error during Chat App component initialization: {e}", exc_info=True)
    agent_memory = None; task_queue = None; agent_instance = None

# --- Helper Functions ---
def format_memories_for_display(memories: List[Dict[str, Any]]) -> str:
    # ... (unchanged) ...
    if not memories: return "No relevant memories found for this turn."
    output = "🧠 **Relevant Memories:**\n\n"; i = 0
    for mem in memories[:5]:
        content_snippet = mem.get('content', 'N/A').replace('\n', ' ')[:150]
        meta_type = mem.get('metadata', {}).get('type', 'memory')
        dist_str = f"{mem.get('distance', -1.0):.3f}" if mem.get('distance') is not None else "N/A"
        output += f"**{i+1}. Type:** {meta_type} (Dist: {dist_str})\n"
        output += f"   Content: _{content_snippet}..._\n\n"; i+=1
    return output

# --- NEW: Helper Function to Decide on Task Generation ---
def _should_generate_task(user_msg: str, assistant_response: str) -> bool:
    """
    Uses a lightweight LLM call to determine if the last chat interaction
    warrants generating a background task.
    """
    log.info("Evaluating if task generation is warranted...")
    prompt = f"""You are an assistant evaluating a conversation turn between a user and an AI assistant. Your goal is to decide if the user's request or the assistant's response implies a need for a more complex, time-consuming background task (like deep research, document analysis, code generation, etc.) that the assistant cannot fully handle immediately in the chat response.

Consider the following interaction:
User: {user_msg}
Assistant: {assistant_response}

Does this interaction strongly suggest a need for a separate background task?
Answer ONLY with "YES" or "NO". Do not provide any explanation.
"""
    # Use a potentially faster/smaller model for this quick check if available,
    # otherwise default to the main chat model.
    eval_model = config.OLLAMA_CHAT_MODEL
    log.debug(f"Asking {eval_model} to evaluate task need...")

    response = call_ollama_api(
        prompt=prompt,
        model=eval_model,
        base_url=config.OLLAMA_BASE_URL,
        timeout=30 # Shorter timeout for evaluation
    )

    if response and "yes" in response.strip().lower():
        log.info("Evaluation result: YES, task generation warranted.")
        return True
    else:
        log.info(f"Evaluation result: NO ({response}), task generation not warranted.")
        return False

# --- Gradio Chat Logic ---
def chat_response(
    message: str,
    history: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], str, str, str]: # Added task display output
    """Handles user input, interacts with LLM, updates memory, conditionally triggers tasks."""
    # --- Initialize outputs ---
    memory_display_text = "Processing..."
    task_display_text = "(No task generated this turn)" # Default message

    if not message: return history, "No input provided.", task_display_text, ""
    if not agent_memory or not task_queue or not agent_instance:
         history.append((message, "**ERROR:** Backend components not initialized."))
         return history, "Error: Backend components failed.", task_display_text, ""

    log.info(f"User message received: '{message}'")
    current_turn_history = history + [(message, None)] # Show user message immediately

    # --- Need to yield intermediate state to show user message before processing ---
    # Gradio requires the function signature to match the final return, even for yields.
    # So we yield the initial state before heavy processing.
    # yield current_turn_history, memory_display_text, task_display_text, message # Yield message to keep it? No, clear it.

    try:
        # 1. Retrieve Memories
        history_context = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history[-3:] if h[1]])
        memory_query = f"Context relevant to user query: '{message}' based on recent chat:\n{history_context}"
        relevant_memories = agent_memory.retrieve_raw_candidates(query=memory_query, n_results=7)
        memory_display_text = format_memories_for_display(relevant_memories) # Update memory display text
        log.info(f"Retrieved {len(relevant_memories)} memories for chat context.")

        # 2. Construct LLM Prompt
        system_prompt = "You are a helpful AI assistant. Answer the user's query using chat history and relevant memories." # Simplified
        memories_for_prompt = "\n".join([f"- {mem['content']}" for mem in relevant_memories[:3]])
        history_for_prompt = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history if h[1]])
        prompt = f"{system_prompt}\n\n## Relevant Memories:\n{memories_for_prompt if memories_for_prompt else 'None'}\n\n## Chat History:\n{history_for_prompt}\n\n## Current Query:\nUser: {message}\nAssistant:"

        # 3. Call LLM for Chat Response
        log.info(f"Asking {config.OLLAMA_CHAT_MODEL} for chat response...")
        response_text = call_ollama_api(prompt=prompt, model=config.OLLAMA_CHAT_MODEL)
        if not response_text: response_text = "Sorry, error generating response."; log.error("LLM call failed.")

        # Update history with the main response
        current_turn_history[-1] = (message, response_text)

        # 4. Add Chat Interaction to Agent Memory
        log.info("Adding chat interaction to agent memory...")
        agent_memory.add_memory(content=f"User query: {message}", metadata={"type": "chat_user_query"})
        agent_memory.add_memory(content=f"Assistant response: {response_text}", metadata={"type": "chat_assistant_response"})

        # --- 5. Conditionally Generate Task ---
        if _should_generate_task(message, response_text):
            log.info("Interaction warrants task generation. Calling generate_new_tasks...")
            first_task_generated_desc = agent_instance.generate_new_tasks(
                max_new_tasks=1,
                last_user_message=message,
                last_assistant_response=response_text
            )
            if first_task_generated_desc:
                # Update the dedicated task display box
                task_display_text = f"✅ Task Generated:\n\"{first_task_generated_desc}\""
                log.info(f"Task generated and displayed: {first_task_generated_desc[:60]}...")
            else:
                 task_display_text = "(Evaluation suggested task, but generation failed or returned none)"
                 log.info("Task generation was warranted but no task was added.")
        else:
             log.info("Interaction evaluated, task generation not warranted.")
             task_display_text = "(No new task warranted this turn)"
        # ------------------------------------

        # 6. Return final updates for Gradio UI
        return current_turn_history, memory_display_text, task_display_text, "" # history, memory_panel, task_panel, clear_textbox

    except Exception as e:
        log.exception(f"Error during chat processing: {e}")
        error_message = f"An internal error occurred: {e}"
        current_turn_history[-1] = (message, error_message)
        # Show error in memory panel for debugging, keep task display default
        return current_turn_history, f"Error during processing:\n```\n{traceback.format_exc()}\n```", task_display_text, ""


# --- Gradio UI Definition ---
log.info("Defining Gradio UI...")
with gr.Blocks(theme=gr.themes.Glass(), title="Agent Chat Interface") as demo: # Changed theme for variety
    gr.Markdown("# Chat with the Autonomous Agent")
    gr.Markdown("Interact with the agent, view relevant memories, see generated tasks.")

    with gr.Row():
        with gr.Column(scale=3): # Main chat area
            chatbot = gr.Chatbot(
                label="Conversation",
                bubble_full_width=False,
                height=550, # Reduced height slightly
                show_copy_button=True,
            )
            # --- Moved Task Display Below Chat ---
            task_panel = gr.Textbox(
                label="💡 Last Generated Task",
                value="(No task generated yet)",
                lines=3,
                interactive=False, # Read-only
                show_copy_button=True
            )
            # -------------------------------------
            with gr.Row(): # Row for input textbox and button
                msg_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Type message and press Enter or click Send...",
                    lines=3, # Kept lines=3 for multiline input possibility
                    scale=5,
                    container=False
                )
                send_button = gr.Button("Send", variant="primary", scale=1)

        with gr.Column(scale=1): # Side panel for memories
            gr.Markdown("### Relevant Memories")
            memory_panel = gr.Markdown(
                value="Memory context will appear here.",
                label="Memory Context"
            )

    # --- Event Handlers ---
    # Define inputs/outputs including the new task panel
    chat_inputs = [msg_input, chatbot]
    # Order matters: chatbot, memory_panel, task_panel, msg_input (to clear)
    chat_outputs = [chatbot, memory_panel, task_panel, msg_input]

    # Bind events to the chat_response function
    send_button.click(
        fn=chat_response,
        inputs=chat_inputs,
        outputs=chat_outputs,
        queue=True # Enable queuing
    )
    msg_input.submit(
         fn=chat_response,
         inputs=chat_inputs,
         outputs=chat_outputs,
         queue=True # Enable queuing
    )

# --- Launch the App ---
if __name__ == "__main__":
    log.info("Launching Gradio Chat App...")
    demo.launch(server_name="0.0.0.0", share=False)
    log.info("Gradio App stopped.")