# Project Context

## Included Files (17 total):

- `.gitignore`
- `LICENSE`
- `README.md`
- `agent.py`
- `app_ui.py`
- `chat_app.py`
- `config.py`
- `data_structures.py`
- `llm_utils.py`
- `main.py`
- `memory.py`
- `requirements.txt`
- `task_manager.py`
- `tools/__init__.py`
- `tools/base.py`
- `tools/web_browse.py`
- `tools/web_search.py`

================================================================================

## File Contents:

### `.gitignore`

```
__pycache__/
*/__pycache__/

chroma_db*/*
output/*
.env
concatenate_project.py
project_context.txt
```

### `LICENSE`

```
Copyright (c) 2025 Bo Wen, David Long

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
```

### `README.md`

```markdown
# Modular Autonomous Agent Framework 🧠🤖

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Explore the capabilities of local Large Language Models (LLMs) with this autonomous agent framework designed for research, task management, and extensibility.**

This project provides a foundation for building autonomous agents that can:

*   Reason and plan using local LLMs via **Ollama**.
*   Maintain persistent memory using **ChromaDB** vector storage.
*   Utilize tools like web search (**SearXNG**) and web browsing (**Requests/BeautifulSoup/PyMuPDF**) to gather information.
*   Manage a queue of tasks with priorities and dependencies.
*   Generate new tasks based on ongoing work or chat interactions.
*   Interact with users via a **Gradio** web interface.
*   Generate data suitable for fine-tuning models (**QLoRA format**).

The agent operates autonomously in the background, processing tasks while providing visibility and control through a web UI.

## ✨ Features

*   **🧠 Intelligent Core:** Leverages local LLMs (configurable via Ollama) for planning, reasoning, reflection, and task generation.
*   **💾 Persistent Vector Memory:** Uses ChromaDB to store and retrieve relevant information (task steps, tool results, reflections, chat history) based on semantic similarity.
*   **🔍 LLM-Powered Memory Re-ranking:** Enhances memory retrieval by using the LLM to re-rank initial candidates for contextual relevance.
*   **🛠️ Extensible Tool System:**
    *   **Web Search:** Integrated with SearXNG for versatile web searching.
    *   **Web Browser:** Can fetch and parse content from URLs (HTML, PDF, TXT, JSON).
    *   Easily add new tools by inheriting from a base class.
*   **📋 Task Management:**
    *   Persistent task queue (`task_queue.json`).
    *   Supports task priorities.
    *   Supports task dependencies (tasks wait for prerequisites).
*   **🤖 Autonomous Operation:** Runs task processing loop in a background thread.
*   **🌐 Gradio Web Interface:**
    *   **Monitor Tab:** Visualize the agent's current task, step logs, recently accessed memories, and last browsed web content. Control buttons to Start/Pause the autonomous loop.
    *   **Chat Tab:** Interact directly with the agent. Chat history is saved to memory, and interactions can trigger new task generation. Displays recently generated tasks and relevant memories for the chat context.
*   **💡 Reactive Task Generation:** The agent can evaluate chat interactions or its idle state to generate new, relevant tasks (including exploratory ones) with priorities and dependencies.
*   **📊 QLoRA Dataset Generation:** Automatically saves completed task `(description, final_answer)` pairs and qualifying chat interactions to a `.jsonl` file suitable for fine-tuning.
*   **📝 Activity Summaries:** Generates daily summary logs (`agent_summaries/summary_YYYY-MM-DD.txt`) for high-level tracking.


## 🏗️ Architecture Overview

The project is structured into several key Python modules within the `autonomous_agent` package:

*   `app_ui.py`: Main entry point, defines the Gradio interface and orchestrates agent startup/shutdown.
*   `agent.py`: Contains the `AutonomousAgent` class, the central orchestrator managing the main loop, state, thinking process, and interactions between components.
*   `memory.py`: Implements the `AgentMemory` class for interacting with ChromaDB and handling embedding pre-checks.
*   `task_manager.py`: Defines the `TaskQueue` for managing tasks and their persistence.
*   `data_structures.py`: Defines the `Task` class.
*   `llm_utils.py`: Handles communication with the Ollama API.
*   `tools/`: A sub-package for tools:
    *   `base.py`: Abstract `Tool` base class.
    *   `web_search.py`: `WebSearchTool` using SearXNG.
    *   `web_browse.py`: `WebBrowserTool` using Requests, BeautifulSoup, PyMuPDF.
*   `config.py`: Loads and stores configuration from the `.env` file.

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   An running **Ollama** instance ([ollama.com](https://ollama.com/)) accessible from where you run the script.
*   Required Ollama models pulled (e.g., `ollama pull gemma3`, `ollama pull nomic-embed-text`). Check `config.py` for defaults.
*   An running **SearXNG** instance ([docs.searxng.org](https://docs.searxng.org/)) accessible from where you run the script (or use a public one, configured in `.env`).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/bowenwen/mindcraft.git
    cd mindcraft
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # Linux/macOS
    # venv\Scripts\activate # Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Environment Variables:**
    *   Copy the example environment file: `cp .env.example .env` (or create `.env` manually).
    *   Edit the `.env` file with your specific settings:
        *   `OLLAMA_BASE_URL`: URL of your Ollama instance (e.g., `http://localhost:11434`). **Do not include a trailing slash.**
        *   `OLLAMA_CHAT_MODEL`: The chat model to use (e.g., `gemma3`).
        *   `OLLAMA_EMBED_MODEL`: The embedding model to use (e.g., `nomic-embed-text`).
        *   `SEARXNG_BASE_URL`: URL of your SearXNG instance (e.g., `http://localhost:8080`).
        *   **(Important!)** `ANONYMIZED_TELEMETRY=False`: Add this line to disable ChromaDB's default telemetry and prevent potential connection errors.
        *   *(Optional)* Set other paths/timeouts defined in `config.py`.

### Running the Application

Run the Gradio application from the root directory of the project:

```bash
python -m autonomous_agent.app_ui
```

This will start the background agent thread (initially paused) and launch the Gradio web interface. Open the URL provided in your terminal (usually `http://127.0.0.1:7860` or `http://0.0.0.0:7860`).

##  How to use

The Gradio interface has two main tabs:

1.  **Agent Monitor:**
    *   **Controls:** Use "Start / Resume" and "Pause" to control the agent's autonomous background task processing.
    *   **Current Task:** Shows the ID, status, and description of the task the agent is currently focused on (or "Idle").
    *   **Last Step Log:** Displays detailed logs generated during the most recent processing step.
    *   **Recent Memories:** Shows a summary of the latest memories retrieved by the agent relevant to its current step.
    *   **Last Web Content:** Displays the text content fetched by the last `web_browse` tool execution.
2.  **Chat:**
    *   **Conversation:** Interact directly with the agent using natural language.
    *   **Relevant Memories (Chat):** Shows memories retrieved specifically for the context of the current chat turn.
    *   **Last Generated Task (Chat):** If your chat interaction warrants a background task, its description will appear here.
    *   Chat history is added to the agent's long-term memory.
    *   Chat interactions deemed sufficiently complex trigger background task generation.
    *   Interactions leading to task generation are saved to the QLoRA dataset file.

## ⚙️ Configuration

Key settings are managed via the `.env` file. See `config.py` for all available options and their default values. Most importantly, set your Ollama and SearXNG URLs.

## 🔧 Extending the Agent

*   **Adding Tools:**
    1.  Create a new Python file in the `tools/` directory (e.g., `tools/my_tool.py`).
    2.  Define a class that inherits from `tools.base.Tool`.
    3.  Implement the `__init__` (setting `name` and `description`) and `run` methods.
    4.  Import your new tool class in `tools/__init__.py` and add an instance to the `AVAILABLE_TOOLS` dictionary.
*   **Modifying Prompts:** Core prompts for agent thinking, task generation, reflection, etc., are located within the methods of the `AutonomousAgent` class in `agent.py`.
*   **Changing Models:** Update the `OLLAMA_CHAT_MODEL` and `OLLAMA_EMBED_MODEL` variables in your `.env` file. Ensure the chosen models are available in your Ollama instance.

## 🛣️ Future Work / Roadmap

*   [ ] Implement more robust error handling and recovery strategies within task steps.
*   [ ] Develop a memory summarization or pruning strategy for very long-running agents.
*   [ ] Add more tools (e.g., file system access, code execution sandbox, specific APIs).
*   [ ] Enhance task duplicate checking using semantic similarity instead of just string matching.
*   [ ] Introduce asynchronous processing for tool execution and potentially LLM calls to improve responsiveness.
*   [ ] Implement a more formal state machine for task/agent status.
*   [ ] Add comprehensive unit and integration tests.
*   [ ] Refine QLoRA data generation format and options.
*   [ ] Integrate Python's `logging` module more deeply for configurable log levels and file output.

## 🙏 Contributing

Contributions are welcome! Please feel free to open an issue to discuss bugs or feature requests, or submit a pull request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (or add the license text directly).
```

### `agent.py`

```python
# autonomous_agent/agent.py
import os
import json
import time
import datetime
import uuid
import re
import traceback
from typing import List, Dict, Any, Optional, Tuple
import threading
from queue import Queue, Empty

# --- Project Imports ---
import config
from data_structures import Task
from task_manager import TaskQueue
from memory import AgentMemory
from tools import load_tools
from llm_utils import call_ollama_api
import chromadb

# --- Logging Setup ---
import logging
log = logging.getLogger("AGENT")

class AutonomousAgent:
    """Orchestrates tasks, memory, tools, LLM calls, and background processing."""
    def __init__(self, memory_collection: Optional[chromadb.Collection] = None):
        log.info("Initializing AutonomousAgent...")
        self.task_queue = TaskQueue()
        if memory_collection is None:
             raise ValueError("Memory collection is required.")
        self.memory = AgentMemory(memory_collection)
        self.tools = load_tools()
        self.ollama_base_url = config.OLLAMA_BASE_URL
        self.ollama_chat_model = config.OLLAMA_CHAT_MODEL
        self.ollama_embed_model = config.OLLAMA_EMBED_MODEL
        self.session_state_path = config.AGENT_STATE_PATH
        self.session_state = {"current_task_id": None, "investigation_context": "", "last_checkpoint": None, "last_web_browse_content": None}
        self.qlora_dataset_path = config.QLORA_DATASET_PATH
        self.load_session_state() # Load previous state first

        # --- State for Background Loop and UI Updates ---
        self._is_running = threading.Event()
        self._shutdown_request = threading.Event()
        self._agent_thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        # --- FIX: Add 'final_answer' to initial state ---
        self._ui_update_state: Dict[str, Any] = {
            "status": "paused",
            "log": "Agent paused.",
            "current_task_id": self.session_state.get("current_task_id"),
            "current_task_desc": "N/A",
            "recent_memories": [],
            "last_web_content": self.session_state.get("last_web_browse_content", "(None)"),
            "final_answer": None, # Initialize final_answer key
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
         }
        # ---------------------------------------------

        # Update initial task description if resuming
        if self.session_state.get("current_task_id"):
             task = self.task_queue.get_task(self.session_state["current_task_id"])
             if task: self._ui_update_state["current_task_desc"] = task.description

        log.info("Agent Initialized.")

    def _update_ui_state(self, **kwargs):
        """Safely updates the shared UI state dictionary."""
        with self._state_lock:
            self._ui_update_state["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            for key, value in kwargs.items():
                if key in self._ui_update_state:
                    self._ui_update_state[key] = value
                else:
                    # This warning should no longer appear for final_answer
                    log.warning(f"Attempted to update unknown UI state key: {key}")
            # Reset final answer display after it's shown once? Optional.
            # if "final_answer" in kwargs and kwargs["final_answer"] is not None:
            #     # Optionally clear it after one update cycle so it only shows briefly
            #     pass # For now, let it persist until overwritten

    def get_ui_update_state(self) -> Dict[str, Any]:
        """Safely retrieves the current UI state."""
        with self._state_lock:
            return self._ui_update_state.copy()

    # --- Memory, State, Task, Tool Methods ---
    # (retrieve_and_rerank_memories, load/save_session_state, create_task,
    #  get_available_tools_description, generate_thinking, execute_tool,
    #  _save_qlora_datapoint, generate_new_tasks, reflections - all unchanged)
    # ... (Paste implementations from previous agent.py here) ...
    def retrieve_and_rerank_memories(self, query: str, task_description: str, context: str, n_candidates: int = 10, n_final: int = 4) -> List[Dict[str, Any]]:
        if not query or not isinstance(query, str) or not query.strip(): return []
        log.info(f"Retrieving & re-ranking memories. Query: '{query[:50]}...'")
        candidates = self.memory.retrieve_raw_candidates(query, n_results=n_candidates)
        if not candidates: log.info("No initial memory candidates found for re-ranking."); return []
        if len(candidates) <= n_final: log.info(f"Fewer candidates ({len(candidates)}) than requested ({n_final}). Returning all."); return candidates
        rerank_prompt = f"""<REDACTED>""" # Keep prompt content
        rerank_prompt = f"""You are an AI assistant helping an agent decide which memories are most relevant for its current task step. Your goal is to select the MOST relevant memories from the list provided below to help the agent achieve its immediate goal.\n\n**Current Task:**\n{task_description}\n\n**Agent's Current Context/Goal for this Step (Use this heavily for ranking):**\n{query}\n\n**Recent Conversation/Action History (for context):**\n{context[-1000:]}\n\n**Candidate Memories (with index, distance, and metadata):**\n"""
        candidate_details = []
        for idx, mem in enumerate(candidates): meta_info = f"Type: {mem['metadata'].get('type', 'N/A')}, Time: {mem['metadata'].get('timestamp', 'N/A')}"; dist_info = f"Distance: {mem.get('distance', 'N/A'):.4f}"; candidate_details.append(f"--- Memory Index {idx} ({dist_info}) ---\nMetadata: {meta_info}\nContent: {mem['content']}\n---")
        rerank_prompt += "\n".join(candidate_details)
        rerank_prompt += f"""\n\n**Instructions:** Review the **Agent's Current Context/Goal**, the **Current Task**, and the **Recent History**. Based on this, identify the **{n_final} memories** (by their index) from the list above that are MOST RELEVANT and MOST USEFUL for the agent to consider *right now* to make progress. Consider relevance, usefulness, recency, type, and similarity (distance).\n**Output Format:** Provide *only* a comma-separated list of the numerical indices (starting from 0) of the {n_final} most relevant memories, ordered from most relevant to least relevant if possible. Do not include any other text, explanation, brackets, or formatting.\nExample: 3, 0, 7, 5"""
        log.info(f"Asking {self.ollama_chat_model} to re-rank {len(candidates)} memories down to {n_final}.")
        rerank_response = call_ollama_api(rerank_prompt, self.ollama_chat_model, self.ollama_base_url, timeout=90)
        if not rerank_response: log.warning("LLM re-ranking failed. Falling back to top N by similarity."); return sorted(candidates, key=lambda m: m.get('distance', float('inf')))[:n_final]
        try:
            matches = re.findall(r'\d+', rerank_response)
            if not matches: log.warning(f"LLM re-ranking response ('{rerank_response}') had no numbers. Falling back to top N."); return sorted(candidates, key=lambda m: m.get('distance', float('inf')))[:n_final]
            selected_indices = [int(idx_str) for idx_str in matches]; valid_indices = [idx for idx in selected_indices if 0 <= idx < len(candidates)]; valid_indices = list(dict.fromkeys(valid_indices))
            if not valid_indices: log.warning(f"LLM re-ranking response ('{rerank_response}') had invalid indices. Falling back to top N."); return sorted(candidates, key=lambda m: m.get('distance', float('inf')))[:n_final]
            final_indices = valid_indices[:n_final]
            log.info(f"LLM selected memory indices for re-ranking: {final_indices}")
            reranked_memories = [candidates[i] for i in final_indices]; return reranked_memories
        except ValueError as e: log.error(f"Error parsing LLM re-ranking indices ('{rerank_response}'): {e}. Falling back to top N."); return sorted(candidates, key=lambda m: m.get('distance', float('inf')))[:n_final]
        except Exception as e: log.exception(f"Unexpected error parsing LLM re-ranking response ('{rerank_response}'): {e}. Falling back to top N."); return sorted(candidates, key=lambda m: m.get('distance', float('inf')))

    def load_session_state(self):
         if os.path.exists(self.session_state_path):
            try:
                with open(self.session_state_path, "r", encoding='utf-8') as f: content = f.read()
                if not content or not content.strip(): log.info(f"Session state file '{self.session_state_path}' empty."); return
                loaded = json.loads(content)
                if isinstance(loaded, dict):
                    self.session_state.setdefault("last_web_browse_content", None) # Ensure key exists
                    self.session_state.update(loaded); log.info(f"Loaded session state from {self.session_state_path}.")
                else: log.warning(f"Session state file '{self.session_state_path}' invalid format.")
            except json.JSONDecodeError as e: log.warning(f"Failed loading session state JSON '{self.session_state_path}': {e}.")
            except Exception as e: log.warning(f"Failed loading session state '{self.session_state_path}': {e}.")
         else: log.info(f"Session state file '{self.session_state_path}' not found. Using defaults.")

    def save_session_state(self):
        try:
            self.session_state["last_checkpoint"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with open(self.session_state_path, "w", encoding='utf-8') as f: json.dump(self.session_state, f, indent=2, ensure_ascii=False)
        except Exception as e: log.error(f"Error saving session state to '{self.session_state_path}': {e}")

    def create_task(self, description: str, priority: int = 1, depends_on: Optional[List[str]] = None) -> Optional[str]:
        return self.task_queue.add_task(Task(description, priority, depends_on=depends_on))

    def get_available_tools_description(self) -> str:
        if not self.tools: return "No tools available."
        active_tools = [];
        for name, tool in self.tools.items():
            is_active = True
            if name == "web_search" and not config.SEARXNG_BASE_URL: is_active = False
            if is_active: active_tools.append(f"- Name: {name}\n  Description: {tool.description}")
        if not active_tools: return "No tools currently active or available."
        return "\n".join(active_tools)

    def generate_thinking(self, task_description: str, context: str = "", tool_results: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        tool_desc = self.get_available_tools_description()
        memory_query = f"Info/past actions relevant to next step? Goal: {context[-500:]}\nTask: {task_description}"
        if tool_results: memory_query += f"\nLast result summary: {str(tool_results)[:200]}"
        relevant_memories = self.retrieve_and_rerank_memories(query=memory_query, task_description=task_description, context=context, n_candidates=10, n_final=4)
        memory_context_str = "\n\n".join([f"Relevant Memory (ID: {mem.get('id', 'N/A')}, Type: {mem['metadata'].get('type', 'N/A')}, Time: {mem['metadata'].get('timestamp', 'N/A')}, Dist: {mem.get('distance', 'N/A'):.4f}):\n{mem['content']}" for mem in relevant_memories]) if relevant_memories else "No relevant memories selected."
        prompt = f"""You are an autonomous AI agent. Your primary goal is to achieve the **Overall Task** by deciding the best next step.\n\n**Overall Task:**\n{task_description}\n\n**Available Tools:**\n{tool_desc}\n\n**Relevant Memories (Re-ranked):**\n{memory_context_str}\n\n**Current Investigation Context (History - may contain previous errors):**\n{context if context else "First step."}\n"""
        if tool_results: prompt += f"\n**Results from Last Action:**\nTool: {tool_results.get('tool_name', 'Unknown')}\nResult:\n```json\n{json.dumps(tool_results.get('result', {}), indent=2, ensure_ascii=False)}\n```\n"
        else: prompt += "\n**Results from Last Action:**\nNone.\n"
        prompt += """**Your Task Now:**\n1. **Analyze:** Review info.\n2. **Reason:** Determine single best next action.\n3. **Handle Previous Errors:** If context mentions an error, *do not give up*. Acknowledge it, but focus on achieving the **Overall Task** (retry, rephrase, different tool). Only use `final_answer` for the *original* task goal.\n4. **Choose Action:** `use_tool` or `final_answer`.
**Output Format (Strict):**\nTHINKING:\n<Reasoning. Explain recovery strategy if handling error.>\n\nNEXT_ACTION: <"use_tool" or "final_answer">\n\nIf "use_tool":\nTOOL: <Exact tool name>\nPARAMETERS: <**Valid JSON object**, **Use ONLY standard double quotes (")**. Example: {{"query": "term"}}>\n\nIf "final_answer":\nANSWER: <Complete answer to **Overall Task**.>\nREFLECTIONS: <Optional thoughts.>
**Critical Formatting Reminder:** Start *immediately* with "THINKING:". Only one action. `PARAMETERS` JSON *must* use standard double quotes ("). `ANSWER` must address the original task."""
        log.info(f"Asking {self.ollama_chat_model} for next action (with error handling guidance)...")
        llm_response_text = call_ollama_api(prompt, self.ollama_chat_model, self.ollama_base_url)
        if llm_response_text is None: log.error("Failed to get thinking response from Ollama."); return None, {"type": "error", "message": "LLM communication failure (thinking)."}
        try:
            action: Dict[str, Any] = {"type": "unknown"}; raw_thinking = llm_response_text
            thinking_marker = "THINKING:"; action_marker = "NEXT_ACTION:"; tool_marker = "TOOL:"; params_marker = "PARAMETERS:"; answer_marker = "ANSWER:"; reflections_marker = "REFLECTIONS:"
            thinking_start = llm_response_text.find(thinking_marker); action_start = llm_response_text.find(action_marker)
            if thinking_start != -1: end_think = action_start if action_start > thinking_start else len(llm_response_text); raw_thinking = llm_response_text[thinking_start + len(thinking_marker):end_think].strip()
            else: log.warning("'THINKING:' marker not found."); raw_thinking = llm_response_text
            action_type_str = ""
            if action_start != -1: end_action = llm_response_text.find('\n', action_start); end_action = end_action if end_action != -1 else len(llm_response_text); action_type_str = llm_response_text[action_start + len(action_marker):end_action].strip()
            else: log.error("'NEXT_ACTION:' marker not found."); action = {"type": "error", "message": "Missing NEXT_ACTION marker."}; return raw_thinking, action
            if "use_tool" in action_type_str:
                action["type"] = "use_tool"; tool_name = None; params_json = None
                tool_start = llm_response_text.find(tool_marker, action_start); params_start = llm_response_text.find(params_marker, action_start)
                if tool_start != -1: end_tool = llm_response_text.find('\n', tool_start); end_tool = params_start if params_start > tool_start and (end_tool == -1 or params_start < end_tool) else end_tool; end_tool = end_tool if end_tool != -1 else len(llm_response_text); tool_name = llm_response_text[tool_start + len(tool_marker):end_tool].strip()
                if params_start != -1:
                    params_str_start = params_start + len(params_marker); end_params = len(llm_response_text)
                    next_marker_pos = -1
                    for marker in [f"\n{answer_marker}", f"\n{reflections_marker}", f"\n{tool_marker}", f"\n{params_marker}"]:
                         pos = llm_response_text.find(marker, params_str_start)
                         if pos != -1 and (next_marker_pos == -1 or pos < next_marker_pos): next_marker_pos = pos
                    if next_marker_pos != -1: end_params = next_marker_pos
                    raw_params = llm_response_text[params_str_start:end_params].strip()
                    raw_params = re.sub(r'^```json\s*', '', raw_params, flags=re.I|re.M); raw_params = re.sub(r'^```\s*', '', raw_params, flags=re.M); raw_params = re.sub(r'\s*```$', '', raw_params, flags=re.M); raw_params = raw_params.strip()
                    json_str = raw_params; params_json = None
                    log.debug(f"Raw PARAMS string: '{json_str}'")
                    if json_str:
                        try: params_json = json.loads(json_str); log.debug("PARAMS parsed directly.")
                        except json.JSONDecodeError as e1:
                            log.debug(f"Direct JSON parse failed: {e1}. Fixing quotes...")
                            try: fixed_str = json_str.replace('“', '"').replace('”', '"').replace("'", '"'); log.debug(f"Attempting parse after quote fix: '{fixed_str}'"); params_json = json.loads(fixed_str); log.debug("PARAMS parsed after quote fix.")
                            except json.JSONDecodeError as e2:
                                first_brace = raw_params.find('{'); last_brace = raw_params.rfind('}')
                                if first_brace != -1 and last_brace > first_brace:
                                    extracted_str = raw_params[first_brace : last_brace + 1]; log.debug(f"Extracting braces: '{extracted_str}'")
                                    try: fixed_extracted_str = extracted_str.replace('“', '"').replace('”', '"').replace("'", '"'); log.debug(f"Trying extracted after quote fix: '{fixed_extracted_str}'"); params_json = json.loads(fixed_extracted_str); log.debug("PARAMS parsed after extraction+fix.")
                                    except json.JSONDecodeError as e3: err_msg = f"Invalid JSON in PARAMETERS after all attempts: {e3}. Original: '{raw_params}'"; log.error(err_msg); action = {"type": "error", "message": err_msg}; return raw_thinking, action
                                else: err_msg = f"Invalid JSON in PARAMETERS: {e2}. Fixed attempt: '{fixed_str if 'fixed_str' in locals() else json_str}'"; log.error(err_msg); action = {"type": "error", "message": err_msg}; return raw_thinking, action
                    else: log.warning("PARAMETERS block empty after cleaning."); action = {"type": "error", "message": "Empty PARAMETERS block."}; return raw_thinking, action
                if params_json is not None and not isinstance(params_json, dict): err_msg = f"Parsed PARAMETERS not dict: {type(params_json)}"; log.error(err_msg); action = {"type": "error", "message": err_msg}; return raw_thinking, action
                if action.get("type") == "use_tool":
                     if tool_name and params_json is not None:
                         if tool_name not in self.tools: err_msg = f"Tool '{tool_name}' not available."; log.error(err_msg); action = {"type": "error", "message": err_msg}
                         else: action["tool"] = tool_name; action["parameters"] = params_json
                     else: missing = []; missing.append("TOOL") if not tool_name else None; missing.append("PARAMETERS") if params_json is None else None; err_msg = f"Missing info for use_tool: {', '.join(missing)}."; log.error(err_msg); action = {"type": "error", "message": err_msg}
            elif "final_answer" in action_type_str:
                action["type"] = "final_answer"; answer = ""; reflections = ""
                answer_start = llm_response_text.find(answer_marker, action_start); reflections_start = llm_response_text.find(reflections_marker, action_start)
                if answer_start != -1: end_answer = reflections_start if reflections_start > answer_start else len(llm_response_text); answer = llm_response_text[answer_start + len(answer_marker):end_answer].strip()
                if reflections_start != -1: reflections = llm_response_text[reflections_start + len(reflections_marker):].strip()
                action["answer"] = answer; action["reflections"] = reflections
                if not answer: log.warning("LLM chose 'final_answer' but ANSWER empty."); action = {"type": "error", "message": "LLM chose final_answer but provided no ANSWER."}
                elif len(answer) < 150 and ("error" in answer.lower() or "cannot proceed" in answer.lower() or "unable to" in answer.lower()): log.warning(f"LLM 'final_answer' seems like error/give up: '{answer[:100]}...'."); action = {"type": "error", "message": f"LLM gave up: {answer}"}
            if action["type"] == "unknown": err_msg = "Could not parse valid action."; log.error(f"{err_msg}\nLLM Response:\n{llm_response_text}"); action = {"type": "error", "message": err_msg}
            return raw_thinking, action
        except Exception as e: log.exception(f"CRITICAL: Unexpected failure parsing LLM thinking response: {e}\nResponse:\n{llm_response_text}"); return raw_thinking, {"type": "error", "message": f"Internal error parsing LLM response: {e}"}

    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
         if tool_name not in self.tools: log.error(f"Tool '{tool_name}' requested but not found."); return {"error": f"Tool '{tool_name}' not found"}
         tool = self.tools[tool_name];
         try:
             result = tool.run(parameters); log.info(f"Tool '{tool_name}' finished.")
             try: json.dumps(result)
             except TypeError: log.warning(f"Result from '{tool_name}' not JSON serializable."); return {"tool_name": tool_name, "result": {"unserializable_result": str(result)}, "status": "completed_unserializable"}
             return {"tool_name": tool_name, "result": result, "status": "completed"}
         except Exception as e: log.exception(f"Error executing tool '{tool_name}': {e}"); return {"tool_name": tool_name, "error": f"Tool execution failed: {e}", "status": "failed"}

    def _save_qlora_datapoint(self, source_type: str, instruction: str, input_context: str, output: str):
        if not output: log.warning("Skipping QLoRA datapoint save due to empty output."); return
        try:
            datapoint = {"instruction": instruction, "input": input_context, "output": output, "source_type": source_type}
            with open(self.qlora_dataset_path, 'a', encoding='utf-8') as f: json_line = json.dumps(datapoint, ensure_ascii=False); f.write(json_line + '\n')
            log.info(f"QLoRA datapoint saved (Source: {source_type})")
        except Exception as e: log.exception(f"Failed to save QLoRA datapoint to {self.qlora_dataset_path}: {e}")

    def _execute_step(self, task: Task) -> Dict[str, Any]:
        """Executes a single thinking/action step for the given task. Internal use."""
        step_start_time = time.time(); step_log = []; final_answer_text = None
        step_status = "processing"; task_status_updated = False

        current_context = self.session_state.get("investigation_context", "")
        step_num = self.session_state.get(f"task_{task.id}_step", 0) + 1
        step_log.append(f"--- Task '{task.id}' | Step {step_num} ---")
        log.info(f"Executing step {step_num} for task {task.id}")

        raw_thinking, action = self.generate_thinking(task.description, current_context, None)

        if action is None:
            log.critical("LLM communication failed during thinking. Failing task."); step_log.append("[CRITICAL] LLM thinking failed.")
            self.task_queue.update_task(task.id, "failed", reflections="LLM communication failed (thinking).")
            self.session_state["current_task_id"] = None; self.session_state["investigation_context"] = ""
            step_status = "failed"; task_status_updated = True
        else:
            thinking_to_store = raw_thinking if raw_thinking else "Thinking process not extracted."; step_log.append(f"Thinking:\n{thinking_to_store}")
            self.memory.add_memory(f"Step {step_num} Thinking:\n{thinking_to_store}", {"type": "task_thinking", "task_id": task.id, "step": step_num, "action_type": action.get('type', 'unknown')})
            action_type = action.get("type", "error")

            if action_type == "error":
                err_msg = action.get('message', 'Unknown LLM action error.'); log.error(f"[ACTION] LLM Action Error: {err_msg}"); step_log.append(f"[ERROR] Action Error: {err_msg}")
                current_context += f"\n--- Step {step_num} Error: {err_msg} ---\nAttempting to proceed..."; self.memory.add_memory(f"Step {step_num} Error: {err_msg}", {"type": "agent_error", "task_id": task.id, "step": step_num})
                step_status = "error"
            elif action_type == "use_tool":
                tool_name = action.get("tool"); params = action.get("parameters")
                log.info(f"[ACTION] Chosen: Use Tool '{tool_name}'"); step_log.append(f"[ACTION] Use Tool: {tool_name}")
                if not tool_name or not isinstance(params, dict): log.error(f"Internal Error: Invalid 'use_tool' action: {action}. Failing task."); step_log.append("[ERROR] Malformed use_tool action."); step_status = "failed"; self.task_queue.update_task(task.id, "failed", reflections="Malformed use_tool."); task_status_updated = True
                else:
                    tool_results = self.execute_tool(tool_name, params)
                    step_context_update = (f"\n--- Step {step_num} ---\nAction: Use Tool '{tool_name}'\nParams: {json.dumps(params, ensure_ascii=False)}\nResult:\n```json\n{json.dumps(tool_results, indent=2, ensure_ascii=False)}\n```\n"); current_context += step_context_update
                    self.memory.add_memory(f"Step {step_num} Tool Result...", {"type": "tool_result", "task_id": task.id, "step": step_num, "tool_name": tool_name, "result_status": tool_results.get("status", "unknown")})
                    if tool_name == 'web_browse' and tool_results.get('status') == 'completed': self.session_state["last_web_browse_content"] = tool_results.get('result', {}).get('content', '(No content)'); step_log.append(f"[INFO] Web browse successful.")
                    elif tool_results.get('status') == 'failed': step_log.append(f"[ERROR] Tool '{tool_name}' failed: {tool_results.get('error')}")
                    step_status = "processing"
            elif action_type == "final_answer":
                log.info("[ACTION] Chosen: Provide Final Answer."); step_log.append("[ACTION] Provide Final Answer.")
                answer = action.get("answer", "").strip(); reflections = action.get("reflections", "").strip()
                if not answer: log.warning("LLM 'final_answer' but no ANSWER text."); step_log.append("[WARN] Final answer empty."); current_context += f"\n--- Step {step_num} Warning: No answer text. ---\n"; self.memory.add_memory(f"Step {step_num} Warning...", {"type": "agent_warning", "task_id": task.id, "step": step_num}); step_status = "error"
                else:
                    final_answer_text = answer
                    print("\n" + "="*15 + f" FINAL ANSWER (Task {task.id}) " + "="*15); print(answer); print("="* (34+len(task.id)) + "\n")
                    result_payload = {"answer": answer, "steps_taken": step_num}; self.task_queue.update_task(task.id, "completed", result=result_payload, reflections=reflections)
                    self._save_qlora_datapoint(source_type="task_completion", instruction="Given the following task, provide a comprehensive answer.", input_context=task.description, output=answer)
                    self.memory.add_memory(f"Final Answer:\n{answer}", {"type": "task_result", "task_id": task.id, "final_step": step_num})
                    if reflections: self.memory.add_memory(f"Final Reflections:\n{reflections}", {"type": "task_reflection", "task_id": task.id, "final_step": step_num})
                    self.session_state["current_task_id"] = None; self.session_state["investigation_context"] = ""
                    step_status = "completed"; task_status_updated = True

        if step_status in ["processing", "error"]:
            context_limit = config.CONTEXT_TRUNCATION_LIMIT
            if len(current_context) > context_limit: log.info(f"Context length ({len(current_context)}) exceeded {context_limit}. Truncating..."); current_context = f"(Context truncated)\n...\n{current_context[-context_limit:]}"
            self.session_state["investigation_context"] = current_context
            self.session_state[f"task_{task.id}_step"] = step_num
        elif step_status in ["completed", "failed"]:
             self.session_state.pop(f"task_{task.id}_step", None)

        self.save_session_state()
        step_duration = time.time() - step_start_time
        step_log.append(f"Step {step_num} duration: {step_duration:.2f}s. Task Status: {step_status}")

        final_task_status = self.task_queue.get_task(task.id).status if task else "unknown" # Get final status
        return {
            "status": final_task_status,
            "log": "\n".join(step_log),
            "current_task_id": self.session_state.get("current_task_id"),
            "current_task_desc": task.description if self.session_state.get("current_task_id") else "None",
            "recent_memories": self.memory.retrieve_raw_candidates(query=f"Activity related to task {task.id}", n_results=5),
            "last_web_content": self.session_state.get("last_web_browse_content", "(No recent web browse)"),
            "final_answer": final_answer_text # This specific step's answer
        }

    def process_one_step(self) -> Dict[str, Any]:
        """Processes a single step of the highest priority runnable task. Returns UI state dict."""
        log.info("Processing one autonomous step...")
        step_result_log = ["Attempting to process one step..."]
        # Default values for UI state
        default_state = {
            "status": "idle", "log": "Agent idle.", "current_task_id": None,
            "current_task_desc": "None", "recent_memories": [],
            "last_web_content": self.session_state.get("last_web_browse_content", "(No recent web browse)"),
            "final_answer": None
        }

        try:
            task_id_to_process = self.session_state.get("current_task_id")
            task: Optional[Task] = None # Explicitly type task

            if task_id_to_process:
                task = self.task_queue.get_task(task_id_to_process)
                if not task or task.status != "in_progress":
                    log.warning(f"Task {task_id_to_process} from state invalid. Searching for next.")
                    step_result_log.append(f"Task {task_id_to_process} invalid/finished.")
                    task = None; self.session_state["current_task_id"] = None

            if not task:
                task = self.task_queue.get_next_task()
                if task:
                    log.info(f"Starting next pending task: {task.id}")
                    step_result_log.append(f"Starting task {task.id}: {task.description[:60]}...")
                    self.task_queue.update_task(task.id, "in_progress")
                    self.session_state["current_task_id"] = task.id
                    self.session_state["investigation_context"] = f"Objective: {task.description}\n"
                    self.session_state.pop(f"task_{task.id}_step", None)
                    self.save_session_state()
                    task = self.task_queue.get_task(task.id) # Re-fetch task with 'in_progress' status
                else:
                    log.info("No runnable tasks found.")
                    step_result_log.append("No runnable tasks found.")
                    self._update_ui_state(status="idle", log="\n".join(step_result_log), current_task_id=None, current_task_desc="None") # Update shared state
                    return self.get_ui_update_state() # Return the updated idle state

            # --- Execute one step ---
            if task: # Ensure task is valid before executing step
                 step_result = self._execute_step(task)
                 # Update the shared state with the result of the step
                 self._update_ui_state(**step_result)
                 return self.get_ui_update_state() # Return the latest state
            else:
                 # Should not happen if logic above is correct, but handle defensively
                 log.error("Task became None unexpectedly before step execution.")
                 step_result_log.append("[ERROR] Internal error: Task lost before execution.")
                 self._update_ui_state(status="error", log="\n".join(step_result_log))
                 return self.get_ui_update_state()


        except Exception as e:
            log.exception("Error during process_one_step")
            step_result_log.append(f"[CRITICAL ERROR] during step processing: {e}")
            # Update shared state with error info
            self._update_ui_state(status="error", log="\n".join(step_result_log), current_task_id=self.session_state.get("current_task_id"))
            return self.get_ui_update_state() # Return error state


    def generate_new_tasks(self, max_new_tasks: int = 3, last_user_message: Optional[str] = None, last_assistant_response: Optional[str] = None) -> Optional[str]:
        # ... (implementation unchanged) ...
        log.info("\n--- Attempting to Generate New Tasks ---")
        if last_user_message and last_assistant_response: context_query = f"Last User Query: {last_user_message}\nLast Assistant Response: {last_assistant_response}"; mem_query = f"Context relevant to last chat: {last_user_message}"; critical_evaluation_instruction = "\n**Critically Evaluate Need:** Based *specifically* on the **Last Interaction**, is a new background task *truly necessary*? If resolved, output `[]`."
        else: log.info("Task generation triggered without specific chat context."); context_query = "General agent status and recent activity."; mem_query = "Recent agent activities, conclusions, errors, reflections, open questions."; critical_evaluation_instruction = ""
        log.info("Retrieving context for task generation...")
        recent_mems = self.memory.retrieve_raw_candidates(query=mem_query, n_results=15)
        mem_summary = "\n".join([f"- Type: {m['metadata'].get('type','mem')}, Time: {m['metadata'].get('timestamp', 'N/A')}\n  Content: {m['content'][:150].strip()}..." for m in recent_mems]) if recent_mems else "No recent memories relevant."
        existing_tasks_info = [{"id": t.id, "description": t.description} for t in self.task_queue.tasks.values() if t.status in ["pending", "in_progress"]]
        existing_tasks_summary = "\n".join([f"- ID: {t['id']}\n  Desc: {t['description'][:100]}..." for t in existing_tasks_info]) if existing_tasks_info else "None"
        valid_existing_task_ids = set(t['id'] for t in existing_tasks_info)
        prompt = f"""You are the planning component of an AI agent. Generate new, actionable tasks based on the agent's context.\n\n**Current Context Focus:**\n{context_query}\n\n**Recent Activity & Memory Snippets:**\n{mem_summary}\n\n**Existing Pending/In-Progress Tasks (with IDs):**\n{existing_tasks_summary}{critical_evaluation_instruction}\n\n**Your Task:** Review context. Identify gaps, next steps, tangents *relevant to Context Focus*. Generate up to {max_new_tasks} *new, specific, actionable* task descriptions.\n**Include a Mix (if appropriate):** 1. **Follow-up Tasks:** Build on results. Use `depends_on` with *existing* IDs if needed. 2. **Exploratory Task (ONE max, if relevant):** Explore related tangent *only if strongly suggested*.\n**Guidelines:** Actionable (verb), Specific, Novel (no duplicates), Relevant, Concise. **Dependencies:** `depends_on`: [list of *existing* task IDs] only if logically required.\n**Output Format (Strict JSON):** Provide *only* a valid JSON list of objects. Each: {{"description": "...", "priority": (opt, 1-10, def 5), "depends_on": (opt, list_of_existing_ids)}}. Example: `[ {{"description": "Summarize task abc.", "depends_on": ["abc"]}}, {{"description": "Explore X."}} ]`. Output `[]` if no tasks needed."""
        log.info(f"Asking {self.ollama_chat_model} to generate up to {max_new_tasks} new tasks...")
        llm_response = call_ollama_api(prompt, self.ollama_chat_model, self.ollama_base_url, timeout=180)
        if not llm_response: log.error("Failed to get response from LLM for task generation."); return None
        first_task_desc_added = None; new_tasks_added = 0
        try:
            llm_response = re.sub(r'^```json\s*', '', llm_response, flags=re.I|re.M); llm_response = re.sub(r'\s*```$', '', llm_response, flags=re.M).strip()
            list_start = llm_response.find('['); list_end = llm_response.rfind(']')
            if list_start == -1 or list_end == -1 or list_end < list_start: raise json.JSONDecodeError("JSON list '[]' not found.", llm_response, 0)
            json_str = llm_response[list_start : list_end + 1]
            suggested_tasks = json.loads(json_str)
            if not isinstance(suggested_tasks, list): log.warning(f"LLM task gen response not a list: {suggested_tasks}"); return None
            if not suggested_tasks: log.info("LLM suggested no new tasks."); return None
            log.info(f"LLM suggested {len(suggested_tasks)} tasks. Validating...")
            current_task_ids_in_batch = set()
            for task_data in suggested_tasks:
                if not isinstance(task_data, dict): log.warning(f"Skipping invalid item (not dict): {task_data}"); continue
                description = task_data.get("description")
                if not description or not isinstance(description, str) or not description.strip(): log.warning(f"Skipping task with invalid description: {task_data}"); continue
                is_duplicate = any(description.strip().lower() == existing_desc['description'].strip().lower() for existing_desc in existing_tasks_info)
                if is_duplicate: log.warning(f"Skipping exact duplicate task: '{description[:80]}...'"); continue
                priority = task_data.get("priority", 5); 
                try: 
                    priority = max(1, min(10, int(priority))) 
                except: 
                    priority = 5
                dependencies_raw = task_data.get("depends_on"); validated_dependencies = []
                if dependencies_raw:
                    if isinstance(dependencies_raw, list):
                        for dep_id in dependencies_raw:
                            if isinstance(dep_id, str) and (dep_id in valid_existing_task_ids or dep_id in current_task_ids_in_batch): validated_dependencies.append(dep_id)
                            else: log.warning(f"Ignoring invalid/unknown dependency ID '{dep_id}' for task '{description[:50]}...'")
                    else: log.warning(f"Invalid 'depends_on' format for '{description[:50]}...'. Ignoring.")
                new_task_id = self.create_task(description, priority, depends_on=validated_dependencies or None)
                if new_task_id:
                     if new_tasks_added == 0: first_task_desc_added = description
                     new_tasks_added += 1; existing_tasks_info.append({"id": new_task_id, "description": description}); valid_existing_task_ids.add(new_task_id); current_task_ids_in_batch.add(new_task_id)
                if new_tasks_added >= max_new_tasks: break
        except json.JSONDecodeError as e: log.error(f"Failed JSON parse for task gen: {e}\nLLM Resp:\n{llm_response}\n---"); return None
        except Exception as e: log.exception(f"Error processing task gen response: {e}"); return None
        log.info(f"Finished Task Generation: Added {new_tasks_added} new tasks.");
        return first_task_desc_added

    def _autonomous_loop(self, initial_delay: float = 2.0, step_delay: float = 5.0):
        """The main loop for autonomous task processing, runs in a thread."""
        log.info("Background agent loop starting.")
        time.sleep(initial_delay)

        while not self._shutdown_request.is_set():
            if not self._is_running.is_set():
                current_state_status = self.get_ui_update_state().get("status")
                if current_state_status != "paused":
                     log.info("Agent loop pausing...")
                     self._update_ui_state(status="paused", log="Agent loop paused.")
                time.sleep(1)
                continue

            # Agent is running
            log.debug("Autonomous loop: Processing step...")
            try:
                # Use process_one_step which handles task selection and step execution
                # It internally calls _execute_step and updates shared state
                self.process_one_step()
            except Exception as e:
                log.exception(f"CRITICAL ERROR in autonomous loop iteration.")
                # Update UI state to reflect critical error
                self._update_ui_state(status="critical_error", log=f"CRITICAL ERROR in loop: {e}")
                time.sleep(step_delay * 3) # Longer sleep after error

            time.sleep(step_delay) # Delay between autonomous steps

        log.info("Background agent loop shutting down.")

    def start_autonomous_loop(self):
        """Starts the background processing thread if not already running."""
        if self._agent_thread and self._agent_thread.is_alive():
            if not self._is_running.is_set():
                log.info("Agent loop resuming...")
                self._is_running.set()
                self._update_ui_state(status="running", log="Agent resumed.")
            else:
                log.info("Agent loop is already running.")
        else:
            log.info("Starting new background agent loop...")
            self._shutdown_request.clear()
            self._is_running.set() # Start in running state
            # Update state immediately
            self._update_ui_state(status="running", log="Agent started.")
            self._agent_thread = threading.Thread(target=self._autonomous_loop, daemon=True)
            self._agent_thread.start()


    def pause_autonomous_loop(self):
        """Signals the background loop to pause."""
        if self._is_running.is_set():
            log.info("Pausing agent loop...")
            self._is_running.clear() # Loop will see this and update status
        else:
            log.info("Agent loop is already paused.")

    def shutdown(self):
        """Signals the background loop to stop."""
        log.info("Shutdown requested for agent background thread.")
        self._shutdown_request.set()
        self._is_running.clear()
        if self._agent_thread and self._agent_thread.is_alive():
            log.info("Waiting for agent thread to join...")
            self._agent_thread.join(timeout=10)
            if self._agent_thread.is_alive():
                log.warning("Agent thread did not join cleanly.")
        self._update_ui_state(status="shutdown", log="Agent shut down.")
        log.info("Agent shutdown sequence complete.")


    def add_self_reflection(self, reflection: str, reflection_type: str = "self_reflection"):
        # ... (implementation unchanged) ...
         if not reflection or not isinstance(reflection, str) or not reflection.strip(): log.warning("Attempted to add empty reflection."); return None
         log.info(f"Adding {reflection_type} to memory...")
         return self.memory.add_memory(reflection, {"type": reflection_type})

    def generate_and_add_session_reflection(self, start: datetime.datetime, end: datetime.datetime, completed_count: int, processed_count: int):
        # ... (implementation unchanged) ...
         duration_minutes = (end - start).total_seconds() / 60
         log.info("Retrieving context for session reflection...")
         recent_mems = self.memory.retrieve_raw_candidates(query="Summary of session activities, errors, outcomes.", n_results=15)
         mem_summary = "\n".join([f"- {m['metadata'].get('type','mem')}: {m['content'][:100].strip()}..." for m in recent_mems]) if recent_mems else "None"
         prompt = f"""You are the AI agent. Reflect on your work session/period.\n**Period Start:** {start.isoformat()}\n**Period End:** {end.isoformat()}\n**Duration:** {duration_minutes:.1f} min\n**Tasks Completed:** {completed_count}\n**Tasks Processed:** {processed_count}\n**Recent Activity Snippets:**\n{mem_summary}\n\n**Reflection Task:** Provide concise reflection: 1. Progress? 2. Efficiency? 3. Challenges? 4. Insights? 5. Improvements?"""
         log.info(f"Asking {self.ollama_chat_model} for session reflection...")
         reflection = call_ollama_api(prompt, self.ollama_chat_model, self.ollama_base_url, timeout=120)
         if reflection and reflection.strip():
             print("\n--- Generated Session Reflection ---"); print(reflection); print("--- End Reflection ---")
             self.add_self_reflection(reflection, "session_reflection")
         else: log.warning("Failed to generate session reflection.")
```

### `app_ui.py`

```python
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
# Configure root logger (adjust level and format as needed)
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='[%(asctime)s] [%(levelname)s][%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger("AppUI") # Logger for this UI script

# --- Global Variables / Setup ---
# Initialize components needed for the UI and Agent Interaction
log.info("Initializing components for Gradio App...")
agent_instance: Optional[AutonomousAgent] = None
try:
    mem_collection = setup_chromadb()
    if mem_collection is None: raise RuntimeError("Database setup failed.")
    # Instantiate the agent - it holds the core logic and components
    agent_instance = AutonomousAgent(memory_collection=mem_collection)
    log.info("App components initialized successfully.")
except Exception as e:
    log.critical(f"Fatal error during App component initialization: {e}", exc_info=True)
    # Display error prominently if possible, or exit
    print(f"\n\nFATAL ERROR: Could not initialize agent components: {e}\n", file=sys.stderr)
    # Set agent_instance to None so UI can show error state
    agent_instance = None


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
        return "Agent loop started/resumed."
    else:
        log.error("UI trigger failed: Agent not initialized.")
        return "ERROR: Agent not initialized."

def pause_agent_processing():
    """Called by Gradio button to pause the background loop."""
    if agent_instance:
        log.info("UI triggered: Pause Agent Processing")
        agent_instance.pause_autonomous_loop()
        return "Agent loop pause requested."
    else:
        log.error("UI trigger failed: Agent not initialized.")
        return "ERROR: Agent not initialized."

# --- Function to update Monitor UI (returns tuple) ---
def update_monitor_ui() -> Tuple[str, str, str, str, str, str]: # Added final_answer output
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
        ui_state = agent_instance.get_ui_update_state() # Get state dict

        # Safely extract data
        current_task_id = ui_state.get('current_task_id', 'None')
        task_status = ui_state.get('status', 'unknown')
        current_task_desc = ui_state.get('current_task_desc', 'N/A')
        step_log_output = ui_state.get('log', '(No recent log)')
        recent_memories = ui_state.get('recent_memories', [])
        last_browse_content = ui_state.get('last_web_content', '(No recent web browse)')
        final_answer = ui_state.get('final_answer') # Get the final answer state

        # Format for UI
        current_task_display = f"**ID:** {current_task_id}\n**Status:** {task_status}\n**Desc:** {current_task_desc}"
        memory_display = format_memories_for_display(recent_memories)

        web_content_limit = 2000
        if last_browse_content and len(last_browse_content) > web_content_limit:
            web_content_display = last_browse_content[:web_content_limit] + "\n\n... (content truncated)"
        else:
            web_content_display = last_browse_content

        status_bar_text = f"Agent Status: {task_status} @ {ui_state.get('timestamp', 'N/A')}"

        # Format the final answer display
        if final_answer:
            final_answer_display = final_answer
            # Optionally clear the state after displaying once? If you want it to disappear after one refresh:
            # agent_instance._update_ui_state(final_answer=None)
        # else: keep default message "(No final answer...)"

    except Exception as e:
        log.exception("Error in update_monitor_ui")
        error_msg = f"ERROR updating UI:\n{traceback.format_exc()}"
        step_log_output = error_msg # Show error in log box
        status_bar_text = "Error updating UI"
        # Keep other fields showing error state? Or use defaults? Let's use defaults.
        current_task_display = "Error"
        memory_display = "Error"
        web_content_display = "Error"
        final_answer_display = "Error"


    # Return tuple in the order defined in monitor_outputs list for timer.tick
    return current_task_display, step_log_output, memory_display, web_content_display, status_bar_text, final_answer_display


# --- Functions for Chat Tab ---
# (_should_generate_task unchanged)
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

# (chat_response unchanged)
def chat_response(
    message: str,
    history: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], str, str, str]: # history, memory_panel, task_panel, msg_input
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
     # Define a simple error UI if agent failed
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
                monitor_status_bar = gr.Textbox(label="Agent Status", value="Paused (initially)", interactive=False)
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Current Task")
                        monitor_current_task = gr.Markdown("(Agent Paused)") # Define component
                        gr.Markdown("### Last Step Log")
                        monitor_log = gr.Textbox(label="Log Output", lines=10, interactive=False, autoscroll=True) # Define component
                        gr.Markdown("### Last Final Answer")
                        monitor_final_answer = gr.Textbox(label="Final Answer Display", lines=5, interactive=False, show_copy_button=True) # Define component
                    with gr.Column(scale=1):
                        gr.Markdown("### Recent Memories")
                        monitor_memory = gr.Markdown("(Memories will appear here)") # Define component
                        gr.Markdown("### Last Web Content")
                        monitor_web_content = gr.Textbox(label="Last Web Content Fetched", lines=10, interactive=False, show_copy_button=True) # Define component

                # Define list of output components for the timer tick (ORDER MATTERS!)
                monitor_outputs = [
                    monitor_current_task,
                    monitor_log,
                    monitor_memory,
                    monitor_web_content,
                    monitor_status_bar,
                    monitor_final_answer # Must match return order of update_monitor_ui
                ]

                # Button actions - Must be defined AFTER components they output to
                start_resume_btn.click(fn=start_agent_processing, inputs=None, outputs=monitor_status_bar)
                pause_btn.click(fn=pause_agent_processing, inputs=None, outputs=monitor_status_bar)

                # Periodic Update using Timer
                timer = gr.Timer(5) # Define Timer component with 5 sec interval
                # Link the timer tick to the update function and outputs
                timer.tick(fn=update_monitor_ui, inputs=None, outputs=monitor_outputs)


            # --- Chat Tab ---
            with gr.TabItem("Chat"):
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
                # Link chat events AFTER components are defined
                chat_send_button.click(fn=chat_response, inputs=chat_inputs, outputs=chat_outputs, queue=True)
                chat_msg_input.submit(fn=chat_response, inputs=chat_inputs, outputs=chat_outputs, queue=True)


# --- Launch the App & Background Thread ---
if __name__ == "__main__":
    # Ensure summary directory exists
    try: os.makedirs(config.SUMMARY_FOLDER, exist_ok=True); log.info(f"Summary directory: {config.SUMMARY_FOLDER}")
    except Exception as e: log.error(f"Could not create summary directory {config.SUMMARY_FOLDER}: {e}")

    log.info("Launching Gradio App Interface...")

    if agent_instance:
        log.info("Starting agent background processing thread (initially paused)...")
        agent_instance.start_autonomous_loop()
        agent_instance.pause_autonomous_loop() # Start paused
        try:
            demo.launch(server_name="0.0.0.0", share=False) # This blocks main thread
        except Exception as e:
            log.critical(f"Gradio demo.launch() failed: {e}", exc_info=True)
            log.info("Requesting agent shutdown due to Gradio launch error...")
            agent_instance.shutdown() # Attempt shutdown
            sys.exit("Gradio launch failed.")
    else:
        # If agent init failed, Gradio defined a simple error message block above
        log.warning("Agent initialization failed. Launching minimal error UI.")
        try:
            # Need to define the error demo within this block if agent_instance is None
            with gr.Blocks() as error_demo:
                 gr.Markdown("# Fatal Error")
                 gr.Markdown("The agent backend failed to initialize. Check logs and configuration then restart the application.")
            error_demo.launch(server_name="0.0.0.0", share=False)
        except Exception as e:
             log.critical(f"Gradio demo.launch() failed even for error UI: {e}", exc_info=True)
             sys.exit("Gradio launch failed.")


    # --- Cleanup (runs when Gradio server is stopped, e.g., Ctrl+C in terminal) ---
    log.info("Gradio App stopped. Requesting agent shutdown...")
    if agent_instance:
        agent_instance.shutdown() # Signal background thread to stop
    log.info("Shutdown complete.")
    print("\n--- Autonomous Agent App End ---") # Use print for final exit message
```

### `chat_app.py`

```python
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
```

### `config.py`

```python
# autonomous_agent/config.py
import os
from dotenv import load_dotenv

print("[CONFIG] Loading environment variables...")
load_dotenv()

# --- Common Configuration ---
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", "./output")

# --- Ollama Configuration ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "gemma3")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", 120))

# --- SearXNG Configuration ---
SEARXNG_BASE_URL = os.environ.get("SEARXNG_BASE_URL", "http://localhost:8080")
SEARXNG_TIMEOUT = int(os.environ.get("SEARXNG_TIMEOUT", 20))

# --- Database Configuration ---
DB_PATH = os.environ.get("DB_PATH", "./chroma_db")
MEMORY_COLLECTION_NAME = os.environ.get("MEMORY_COLLECTION_NAME", "agent_memory")

# --- Task Queue Configuration ---
TASK_QUEUE_PATH = f"{OUTPUT_FOLDER}/task_queue.json"
INVESTIGATION_RES_PATH = f"{OUTPUT_FOLDER}/investigation_results.json"

# --- Agent Configuration ---
AGENT_STATE_PATH = f"{OUTPUT_FOLDER}/session_state.json"
DEFAULT_MAX_STEPS_PER_TASK = int(os.environ.get("DEFAULT_MAX_STEPS_PER_TASK", 8))
DEFAULT_SESSION_DURATION_MINUTES = int(
    os.environ.get("DEFAULT_SESSION_DURATION_MINUTES", 30)
)
CONTEXT_TRUNCATION_LIMIT = int(os.environ.get("CONTEXT_TRUNCATION_LIMIT", 40000))
MODEL_CONTEXT_LENGTH = CONTEXT_TRUNCATION_LIMIT + 10000

# --- Summary Configuration ---
SUMMARY_FOLDER = os.environ.get("SUMMARY_FOLDER", "./output/summary")

# --- NEW: QLoRA Dataset Configuration ---
QLORA_DATASET_PATH = f"{OUTPUT_FOLDER}/qlora_finetune_data.json"

# --- Logging Configuration ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

print("[CONFIG] Configuration loaded.")
# (Optionally print other loaded configs)
print(f"  QLORA_DATASET_PATH={QLORA_DATASET_PATH}")
```

### `data_structures.py`

```python
# autonomous_agent/data_structures.py
import uuid
import datetime
from typing import List, Dict, Any, Optional

class Task:
    """Represents a single task with description, priority, status, and dependencies."""
    def __init__(self, description: str, priority: int = 1, depends_on: Optional[List[str]] = None):
        self.id = str(uuid.uuid4())
        self.description = description
        self.priority = max(1, min(10, priority)) # Ensure priority is 1-10
        self.depends_on = depends_on if isinstance(depends_on, list) else ([depends_on] if isinstance(depends_on, str) else [])
        self.status = "pending" # pending, in_progress, completed, failed
        self.created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.completed_at = None
        self.result = None # Store final answer or outcome
        self.reflections = None # Store LLM reflections on task execution

    def to_dict(self) -> Dict[str, Any]:
        """Serializes task object to a dictionary."""
        return {
            "id": self.id, "description": self.description, "priority": self.priority,
            "depends_on": self.depends_on, "status": self.status, "created_at": self.created_at,
            "completed_at": self.completed_at, "result": self.result, "reflections": self.reflections
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Deserializes task object from a dictionary."""
        task = cls(
            description=data.get("description", "No description provided"),
            priority=data.get("priority", 1), depends_on=data.get("depends_on")
        )
        # Restore all fields
        task.id = data.get("id", task.id)
        task.status = data.get("status", "pending")
        task.created_at = data.get("created_at", task.created_at)
        task.completed_at = data.get("completed_at")
        task.result = data.get("result")
        task.reflections = data.get("reflections")
        return task
```

### `llm_utils.py`

```python
# autonomous_agent/llm_utils.py
import requests
import time
import json
from typing import Optional, List, Dict, Any

# Import specific config values needed
from config import OLLAMA_BASE_URL, OLLAMA_TIMEOUT, MODEL_CONTEXT_LENGTH

# Suggestion: Consider adding basic logging setup here or importing from a dedicated logging module
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s][LLM] %(message)s")
log = logging.getLogger(__name__)


def call_ollama_api(
    prompt: str,
    model: str,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = OLLAMA_TIMEOUT,
) -> Optional[str]:
    """Calls the Ollama chat API and returns the content of the response message."""
    api_url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.8, "num_ctx": MODEL_CONTEXT_LENGTH},
    }
    headers = {"Content-Type": "application/json"}
    max_retries = 1
    retry_delay = 3
    # log.debug(f"Sending request to Ollama ({model}). Prompt snippet: {prompt[:100]}...") # Use logger

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                api_url, json=payload, headers=headers, timeout=timeout
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            response_data = response.json()
            if "message" in response_data and "content" in response_data["message"]:
                # log.debug(f"Received response from Ollama ({model}).") # Use logger
                return response_data["message"]["content"]
            elif "error" in response_data:
                log.error(f"Error from Ollama API ({model}): {response_data['error']}")
                return None
            else:
                log.error(
                    f"Unexpected Ollama response structure ({model}): {response_data}"
                )
                return None
        except requests.exceptions.Timeout:
            log.error(f"Ollama API request timed out after {timeout}s ({model}).")
            if attempt < max_retries:
                log.info(f"Retrying Ollama call in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                log.error("Max retries reached for Ollama timeout.")
                return None
        except requests.exceptions.RequestException as e:
            log.error(f"Error calling Ollama API ({model}) at {api_url}: {e}")
            if hasattr(e, "response") and e.response is not None:
                log.error(
                    f"  Response status: {e.response.status_code}, Text: {e.response.text[:200]}..."
                )
            return None
        except Exception as e:
            log.exception(
                f"Unexpected error calling Ollama ({model}): {e}"
            )  # Logs stack trace
            return None
    return None  # Should only be reached if retries fail
```

### `main.py`

```python
# autonomous_agent/main.py
import os
import json
import traceback
import sys
import logging
from typing import Optional

# --- Project Imports ---
import config
from llm_utils import call_ollama_api
from memory import setup_chromadb # Function to setup DB
from agent import AutonomousAgent
import chromadb

# --- Basic Logging Setup for Main Script ---
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='[%(asctime)s] [%(levelname)s][%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger("MAIN")

# --- Function to Generate Initial Topic ---
def generate_initial_topic(model: str = config.OLLAMA_CHAT_MODEL, base_url: str = config.OLLAMA_BASE_URL) -> Optional[str]:
    """Asks the LLM to generate an interesting research topic."""
    log.info("Asking LLM to generate an initial research topic...")
    prompt = """Suggest one interesting, specific, and researchable topic suitable for an AI agent with web search capabilities to investigate and summarize. The topic should require gathering and synthesizing information from multiple sources. Avoid overly broad or simple topics. Output only the topic description itself, without any preamble, quotes, or explanation."""
    log.info(f"Asking {model} for initial topic suggestion...")
    topic = call_ollama_api(prompt, model, base_url, timeout=60)
    if topic and topic.strip():
        cleaned_topic = topic.strip().strip('"').strip("'").strip()
        log.info(f"LLM suggested initial topic: '{cleaned_topic}'")
        return cleaned_topic
    else:
        log.warning("LLM failed to generate an initial topic.")
        return None

# --- Main Execution Helper ---
def run_agent_and_collect_data(investigation_topic: str,
                               duration_minutes: int = config.DEFAULT_SESSION_DURATION_MINUTES,
                               max_steps_per_task: int = config.DEFAULT_MAX_STEPS_PER_TASK,
                               memory_collection: Optional[chromadb.Collection] = None):
    """Initializes agent, ensures initial task, runs session, returns results."""
    print(f"\n{'#'*10} Starting Agent Investigation {'#'*10}") # Use print for major demarcations
    log.info(f"Topic: '{investigation_topic}'")
    log.info(f"Duration: {duration_minutes} min, Max Steps/Task: {max_steps_per_task}")

    try:
        agent = AutonomousAgent(memory_collection=memory_collection)
    except Exception as e:
        log.critical(f"Failed to initialize AutonomousAgent: {e}", exc_info=True)
        return None # Cannot proceed if agent fails init

    task_desc = f"Investigate and provide a comprehensive summary of: {investigation_topic}"; existing_task = None
    for task in agent.task_queue.tasks.values():
        if task.description == task_desc and task.status in ["pending", "in_progress"]: existing_task = task; break

    if not existing_task:
        log.info(f"Creating initial investigation task: '{task_desc[:60]}...' (Prio 10)")
        agent.create_task(task_desc, priority=10)
    else:
         log.info(f"Found existing relevant task {existing_task.id} (Status: {existing_task.status}). Not creating duplicate.")

    # Session results now include counts for completed, failed, generated
    session_results = agent.run_session(duration_minutes, max_steps_per_task)

    all_tasks_final_state = list(agent.task_queue.tasks.values())
    print(f"\n[MAIN] {'#'*10} Investigation Session Finished {'#'*10}") # Use print
    return {
        "investigation_topic": investigation_topic,
        "session_parameters": {
            "duration_minutes": duration_minutes,
            "max_steps_per_task": max_steps_per_task
        },
        "session_results": session_results, # Pass the dict containing counts
        "final_task_queue_state": [t.to_dict() for t in all_tasks_final_state]
    }

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Autonomous Agent Script Start ---") # Use print for initial start

    log.info("--- Configuration Check ---")
    log.info(f"Ollama Chat: {config.OLLAMA_CHAT_MODEL}, Embed: {config.OLLAMA_EMBED_MODEL} @ {config.OLLAMA_BASE_URL}")
    log.info(f"ChromaDB Path: {config.DB_PATH}")
    if not config.SEARXNG_BASE_URL: log.error("SEARXNG_BASE_URL not set. Web search will fail.")
    else: log.info(f"SearXNG URL: {config.SEARXNG_BASE_URL}")
    log.info(f"Log Level Set To: {config.LOG_LEVEL}")
    log.info(f"Summary Folder: {config.SUMMARY_FOLDER}") # Log summary folder path
    log.info("-" * 25)

    log.info("--- Database Setup ---")
    mem_collection = setup_chromadb()
    if mem_collection is None: log.critical("Exiting due to database setup failure."); sys.exit(1)
    log.info("-" * 25)

    try:
        log.info("--- Initial Topic Generation ---")
        initial_topic = generate_initial_topic()
        if not initial_topic:
            log.warning("Using fallback topic as LLM failed to generate one.")
            initial_topic = "The future potential and challenges of fusion power generation" # Example fallback
        log.info("-" * 25)

        session_duration = config.DEFAULT_SESSION_DURATION_MINUTES
        steps_limit = config.DEFAULT_MAX_STEPS_PER_TASK

        investigation_data = run_agent_and_collect_data(
            investigation_topic=initial_topic,
            duration_minutes=session_duration,
            max_steps_per_task=steps_limit,
            memory_collection=mem_collection # Pass collection
        )

        if investigation_data:
            results_file = config.INVESTIGATION_RES_PATH
            try:
                log.info(f"Saving investigation results to {results_file}...")
                with open(results_file, "w", encoding='utf-8') as f:
                    json.dump(investigation_data, f, indent=2, ensure_ascii=False, default=str)
                log.info(f"Results saved.")
            except Exception as e: log.error(f"Error saving results to {results_file}: {e}")
        else:
             log.error("Agent investigation run failed, skipping results save.")

    except KeyboardInterrupt: print("\n--- KeyboardInterrupt received. Shutting down agent gracefully. ---")
    except Exception as main_err: log.critical(f"UNEXPECTED MAIN EXECUTION ERROR: {main_err}", exc_info=True)
    finally:
        print("\n--- Autonomous Agent Script End ---") # Use print
```

### `memory.py`

```python
# autonomous_agent/memory.py
import uuid
import datetime
import json
import traceback
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import ollama # Direct client for pre-checks

# Import necessary config values
from config import (
    OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, DB_PATH, MEMORY_COLLECTION_NAME
)

# Suggestion: Use logging module
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s][MEMORY] %(message)s')
log = logging.getLogger(__name__)


# --- ChromaDB Setup ---
# Moved setup into a function to handle potential errors gracefully at startup
def setup_chromadb() -> Optional[chromadb.Collection]:
    """Initializes ChromaDB client and collection."""
    log.info(f"Initializing ChromaDB client at path: {DB_PATH}")
    try:
        vector_db = chromadb.PersistentClient(path=DB_PATH)
        log.info(f"Getting or creating ChromaDB collection '{MEMORY_COLLECTION_NAME}'")

        # Initialize embedding function (error if Ollama URL/model invalid from Chroma's perspective)
        embedding_function = OllamaEmbeddingFunction(
            url=f"{OLLAMA_BASE_URL}/api/embeddings",
            model_name=OLLAMA_EMBED_MODEL
        )

        memory_collection = vector_db.get_or_create_collection(
            name=MEMORY_COLLECTION_NAME,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        log.info("ChromaDB collection ready.")
        return memory_collection
    except Exception as e:
        log.critical(f"Could not initialize ChromaDB at {DB_PATH}.")
        log.critical(f"  Potential Issues: Ollama server/model, path permissions, DB errors.")
        log.critical(f"  Error details: {e}", exc_info=True) # Log stack trace
        return None

# --- Agent Memory Class ---
class AgentMemory:
    """Manages interaction with the ChromaDB vector store for agent memories."""
    def __init__(self, collection: chromadb.Collection):
        if collection is None:
             raise ValueError("ChromaDB collection cannot be None for AgentMemory.")
        self.collection = collection
        # Use Ollama client *only* for pre-checks
        try:
             self.ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
        except Exception as e:
             log.error(f"Failed to initialize Ollama client for pre-checks: {e}")
             self.ollama_client = None # Allow continuation but pre-check will fail
        self.ollama_embed_model = OLLAMA_EMBED_MODEL

    def _test_embedding_generation(self, text_content: str) -> bool:
        """Attempts to generate an embedding via Ollama client to pre-check."""
        if not self.ollama_client:
             log.warning("_test_embedding_generation: Ollama client not initialized, skipping pre-check.")
             return True # Assume okay if client failed init, let Chroma handle it
        if not text_content: log.warning("_test_embedding_generation: Empty content passed."); return False
        try:
            # log.debug(f"Testing embedding for: {text_content[:60]}...") # Very Verbose
            response = self.ollama_client.embeddings(model=self.ollama_embed_model, prompt=text_content)
            if response and isinstance(response.get('embedding'), list) and len(response['embedding']) > 0: return True
            else: log.error(f"Ollama client returned empty/invalid embedding. Response: {response}"); return False
        except Exception as e:
            log.error(f"Failed embedding test call to Ollama. Error: {e}")
            log.error(f"  Content Snippet: {text_content[:100]}...")
            log.error(f"  Troubleshoot: Check Ollama server at {OLLAMA_BASE_URL}, model '{self.ollama_embed_model}' availability, content length.")
            return False

    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Adds content to ChromaDB after pre-checking embedding generation."""
        if not content or not isinstance(content, str) or not content.strip(): log.warning("Skipped adding empty memory content."); return None
        if not self._test_embedding_generation(content): log.error(f"Embedding pre-check failed. Skipping add to ChromaDB for content: {content[:100]}..."); return None

        memory_id = str(uuid.uuid4()); metadata = metadata if isinstance(metadata, dict) else {}; metadata["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            # Ensure metadata values are simple types for ChromaDB
            cleaned_metadata = {k: (str(v) if not isinstance(v, (str, int, float, bool)) else v) for k, v in metadata.items()}
            # Let ChromaDB handle the actual embedding using the configured function
            self.collection.add(documents=[content], metadatas=[cleaned_metadata], ids=[memory_id])
            log.info(f"Memory added: ID {memory_id} (Type: {metadata.get('type', 'N/A')})")
            return memory_id
        except Exception as e:
            log.error(f"Failed during collection.add operation. Error: {e}"); log.error(f"  Content: {content[:100]}..."); log.exception("Add Memory Traceback:") # Log stack trace
            return None

    def retrieve_raw_candidates(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Retrieves raw candidate memories based on vector similarity."""
        if not query or not isinstance(query, str) or not query.strip() or n_results <= 0: return []
        # log.debug(f"Retrieving {n_results} candidates for query: '{query[:50]}...'") # Verbose
        try:
            # Ensure n_results doesn't exceed the number of items in the collection
            collection_count = self.collection.count()
            actual_n_results = min(n_results, collection_count) if collection_count > 0 else 0

            if actual_n_results == 0:
                 # log.debug("Collection is empty, returning no candidates.") # Verbose
                 return []

            results = self.collection.query(query_texts=[query], n_results=actual_n_results, include=['metadatas', 'documents', 'distances'])
            memories = []
            # Robust check of ChromaDB response structure
            if results and isinstance(results.get("ids"), list) and results["ids"] and \
               isinstance(results.get("documents"), list) and results.get("documents") is not None and \
               isinstance(results.get("metadatas"), list) and results.get("metadatas") is not None and \
               isinstance(results.get("distances"), list) and results.get("distances") is not None and \
               len(results["ids"][0]) == len(results["documents"][0]) == len(results["metadatas"][0]) == len(results["distances"][0]):
                 for i, doc_id in enumerate(results["ids"][0]):
                     doc = results["documents"][0][i]; meta = results["metadatas"][0][i]; dist = results["distances"][0][i]
                     if doc is not None and isinstance(meta, dict) and isinstance(dist, (float, int)):
                         memories.append({"id": doc_id, "content": doc, "metadata": meta, "distance": dist})
            # log.debug(f"Retrieved {len(memories)} raw candidates.") # Verbose
            return memories
        except Exception as e: log.exception(f"Error retrieving raw memories from ChromaDB: {e}"); return [] # Log stack trace
```

### `requirements.txt`

```
python-dotenv
requests
chromadb >= 0.4.22 # Or later version supporting OllamaEmbeddingFunction well
ollama >= 0.1.7   # Or later version
beautifulsoup4 >= 4.9.0
PyMuPDF >= 1.23.0
gradio >= 4.0.0
```

### `task_manager.py`

```python
# autonomous_agent/task_manager.py
import os
import json
import datetime
from typing import List, Dict, Any, Optional

# Import Task from data_structures
from data_structures import Task
# Import specific config value needed
from config import TASK_QUEUE_PATH

# Suggestion: Use logging module
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s][TASK] %(message)s')
log = logging.getLogger(__name__)


class TaskQueue:
    """Manages the persistence and retrieval of tasks."""
    def __init__(self, queue_path: str = TASK_QUEUE_PATH):
        self.queue_path = queue_path
        self.tasks: Dict[str, Task] = {} # {task_id: Task object}
        self.load_queue()

    def load_queue(self):
        """Loads tasks from the JSON file."""
        if os.path.exists(self.queue_path):
            try:
                with open(self.queue_path, 'r', encoding='utf-8') as f: content = f.read()
                if not content or not content.strip(): log.info(f"Task queue file '{self.queue_path}' empty."); self.tasks = {}; return
                data = json.loads(content)
                if not isinstance(data, list): log.error(f"Task queue '{self.queue_path}' not a list. Starting fresh."); self.tasks = {}; return
                loaded = 0
                for task_data in data:
                    if isinstance(task_data, dict) and "id" in task_data:
                        try: self.tasks[task_data["id"]] = Task.from_dict(task_data); loaded += 1
                        except Exception as e: log.warning(f"Skipping task data parse error: {e}. Data: {task_data}")
                    else: log.warning(f"Skipping invalid task data item: {task_data}")
                log.info(f"Loaded {loaded} tasks from {self.queue_path}")
            except json.JSONDecodeError as e: log.error(f"Invalid JSON in '{self.queue_path}': {e}. Starting fresh."); self.tasks = {}
            except Exception as e: log.error(f"Failed loading task queue '{self.queue_path}': {e}. Starting fresh."); self.tasks = {}
        else: log.info(f"Task queue file '{self.queue_path}' not found. Starting fresh."); self.tasks = {}

    def save_queue(self):
        """Saves the current task list to the JSON file."""
        try:
            with open(self.queue_path, 'w', encoding='utf-8') as f:
                sorted_tasks = sorted(self.tasks.values(), key=lambda t: t.created_at)
                json.dump([task.to_dict() for task in sorted_tasks], f, indent=2, ensure_ascii=False)
        except Exception as e: log.error(f"Error saving task queue to '{self.queue_path}': {e}")

    def add_task(self, task: Task) -> Optional[str]:
        """Adds a task to the queue and saves."""
        if not isinstance(task, Task): log.error(f"Attempted to add non-Task object: {task}"); return None
        self.tasks[task.id] = task; self.save_queue()
        log.info(f"Task added: {task.id} - '{task.description[:60]}...' (Prio: {task.priority}, Depends: {task.depends_on})"); return task.id

    def get_next_task(self) -> Optional[Task]:
        """Gets the highest priority 'pending' task whose dependencies are 'completed'."""
        available = []
        tasks_list = list(self.tasks.values()) # Stable list for iteration
        for task in tasks_list:
            if task.status == "pending":
                deps_met = True
                if task.depends_on: # Only check if dependencies exist
                    for dep_id in task.depends_on:
                        dep_task = self.tasks.get(dep_id)
                        # Dependency must exist AND be completed
                        if not dep_task or dep_task.status != "completed":
                            # log.debug(f"Task {task.id} dependency {dep_id} not met (status: {dep_task.status if dep_task else 'Not Found'}).") # Verbose Debug
                            deps_met = False; break
                if deps_met:
                    available.append(task)
        if not available: return None
        # Sort by priority (desc), then creation time (asc)
        return sorted(available, key=lambda t: (-t.priority, t.created_at))[0]

    def update_task(self, task_id: str, status: str, result: Any = None, reflections: Optional[str] = None):
        """Updates the status and optionally result/reflections of a task."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status == status and result is None and reflections is None: return # Avoid redundant saves/logs if nothing changed
            task.status = status
            log.info(f"Task {task_id} status updated to: {status}")
            if status in ["completed", "failed"]: task.completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            if result is not None: task.result = result
            if reflections is not None: task.reflections = reflections
            self.save_queue() # Save after any update
        else: log.error(f"Cannot update task - ID '{task_id}' not found.")

    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieves a task by its ID."""
        return self.tasks.get(task_id)
```

### `tools/__init__.py`

```python
# autonomous_agent/tools/__init__.py
from typing import Dict
# Import tool implementations
from .base import Tool
from .web_search import WebSearchTool
from .web_browse import WebBrowserTool # <<<--- ADD IMPORT

# Suggestion: Automatically discover tools or use a registration pattern?
# For now, manually list them.
AVAILABLE_TOOLS: Dict[str, Tool] = {
    "web_search": WebSearchTool(),
    "web_browse": WebBrowserTool(), # <<<--- ADD NEW TOOL INSTANCE
    # Add other tools here: e.g., "code_interpreter": CodeInterpreterTool()
}

def load_tools() -> Dict[str, Tool]:
    """Returns a dictionary of available tool instances."""
    # Could add logic here to dynamically load tools if needed
    print(f"[SETUP] Loading tools: {list(AVAILABLE_TOOLS.keys())}")
    return AVAILABLE_TOOLS
```

### `tools/base.py`

```python
# autonomous_agent/tools/base.py
from typing import Dict, Any

class Tool:
    """Base class for tools the agent can use."""
    def __init__(self, name: str, description: str):
        if not name or not description:
            raise ValueError("Tool name and description cannot be empty.")
        self.name = name
        self.description = description

    def run(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the tool's action with the given parameters.
        Returns a dictionary containing the results or an error.
        """
        raise NotImplementedError("Tool subclass must implement run method.")
```

### `tools/web_browse.py`

```python
# autonomous_agent/tools/web_browse.py
import requests
import re
import traceback
import json # <<<--- ADD JSON import
from typing import Dict, Any
from bs4 import BeautifulSoup
import fitz # PyMuPDF

# Import base class and config
from .base import Tool
import config

# Use logging module
import logging
# Ensure basicConfig is called in main.py, just get logger here
log = logging.getLogger("TOOL_WebBrowse") # Use specific logger name


class WebBrowserTool(Tool):
    """
    Tool to fetch the textual content of a specific web page URL.
    Supports HTML, PDF, JSON, and plain text documents.
    NOTE: HTML parsing primarily works for static content.
          PDF/text extraction depends on the file's structure.
    """
    def __init__(self):
        super().__init__(
            name="web_browse",
            description=(
                "Fetches the primary textual content from a given URL (HTML, PDF, JSON, TXT). " # Updated description
                "Useful for reading articles, blog posts, documentation, PDF reports, JSON data, or plain text files when you have the URL. "
                "Parameters: url (str, required)"
            )
        )
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Tags for HTML extraction
        self.content_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'article', 'main', 'section', 'pre', 'code']

    def _clean_text(self, text: str) -> str:
        """Removes excessive whitespace and performs basic cleaning."""
        if not text:
            return ""
        # Replace multiple whitespace characters (newlines, tabs, spaces) with a single space
        text = re.sub(r'\s+', ' ', text)
        # You could add more specific cleaning here if needed
        return text.strip()

    def run(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Fetches and extracts text content from the provided URL."""
        if not isinstance(parameters, dict):
            log.error(f"Invalid params type: {type(parameters)}")
            return {"error": "Invalid parameters format. Expected a JSON object (dict)."}

        url = parameters.get("url")
        if not url or not isinstance(url, str) or not url.strip():
            return {"error": "Missing or invalid 'url' parameter (must be a non-empty string)."}

        if not url.startswith(("http://", "https://")):
            # Attempting without scheme can lead to errors, better to require it
            log.error(f"Invalid URL format: '{url}'. Must start with http:// or https://.")
            return {"error": "Invalid URL format. Must start with http:// or https://."}
            # Alternatively: try adding 'https://' but that's guesswork
            # log.warning(f"URL '{url}' missing scheme. Prepending https://.")
            # url = f"https://{url}"


        log.info(f"Attempting to browse URL: {url}")

        extracted_text = ""
        content_source = "unknown"

        try:
            response = requests.get(url, headers=self.headers, timeout=25)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '').lower().split(';')[0].strip() # Get main type before charset etc.
            log.info(f"URL Content-Type detected: '{content_type}'")

            # --- PDF Handling ---
            if content_type == 'application/pdf':
                content_source = "pdf"
                log.info("Parsing PDF content...")
                pdf_texts = []
                try:
                    doc = fitz.open(stream=response.content, filetype="pdf")
                    log.info(f"PDF has {len(doc)} pages.")
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        page_text = page.get_text("text")
                        if page_text: pdf_texts.append(page_text)
                    doc.close()
                    full_text = "\n".join(pdf_texts) # Join pages with newline
                    extracted_text = self._clean_text(full_text) # Clean after joining
                    log.info(f"Extracted text from PDF (Length: {len(extracted_text)}).")
                except Exception as pdf_err:
                    log.error(f"Error parsing PDF from {url}: {pdf_err}", exc_info=True)
                    return {"error": f"Failed to parse PDF content. Error: {pdf_err}"}

            # --- HTML Handling ---
            elif content_type == 'text/html':
                content_source = "html"
                log.info("Parsing HTML content...")
                soup = BeautifulSoup(response.text, 'html.parser')
                extracted_texts = []
                main_content = soup.find('main') or soup.find('article') or soup.find('body')
                if main_content:
                     elements = main_content.find_all(self.content_tags)
                     if not elements: extracted_texts.append(self._clean_text(main_content.get_text(separator=' ', strip=True)))
                     else: extracted_texts.extend(self._clean_text(el.get_text(separator=' ', strip=True)) for el in elements)
                else: extracted_texts.extend(self._clean_text(el.get_text(separator=' ', strip=True)) for el in soup.find_all(self.content_tags))
                full_text = " ".join(filter(None, extracted_texts)) # Join non-empty strings
                extracted_text = self._clean_text(full_text) # Final cleanup
                log.info(f"Extracted text from HTML (Length: {len(extracted_text)}).")

            # --- JSON Handling ---
            elif content_type == 'application/json':
                 content_source = "json"
                 log.info("Parsing JSON content...")
                 try:
                     json_data = json.loads(response.text)
                     # Pretty-print JSON back into a string for the LLM
                     extracted_text = json.dumps(json_data, indent=2, ensure_ascii=False)
                     # No extra cleaning needed for JSON string usually
                     log.info(f"Formatted JSON content (Length: {len(extracted_text)}).")
                 except json.JSONDecodeError as json_err:
                     log.error(f"Invalid JSON received from {url}: {json_err}")
                     log.debug(f"Raw JSON text received: {response.text[:500]}...") # Log snippet
                     return {"error": f"Failed to parse JSON content. Invalid format. Error: {json_err}"}

            # --- Plain Text Handling ---
            elif content_type.startswith('text/'): # Catch text/plain, text/csv, etc.
                 content_source = "text"
                 log.info(f"Parsing plain text content ({content_type})...")
                 # Use response.text directly and clean it
                 extracted_text = self._clean_text(response.text)
                 log.info(f"Extracted plain text (Length: {len(extracted_text)}).")

            # --- Other Content Types ---
            else:
                log.warning(f"Unsupported Content-Type '{content_type}' for URL {url}.")
                # Maybe try a generic text extraction as a fallback? Or just fail.
                # Option: try generic text extraction
                try:
                     log.info("Attempting generic text extraction as fallback...")
                     fallback_text = self._clean_text(response.text)
                     if fallback_text and len(fallback_text) > 50: # Heuristic: only keep if substantial text found
                          extracted_text = fallback_text
                          content_source = "text_fallback"
                          log.info(f"Extracted fallback text (Length: {len(extracted_text)}).")
                     else:
                           return {"error": f"Cannot browse unsupported content type: {content_type}"}
                except Exception:
                     return {"error": f"Cannot browse unsupported content type: {content_type}"}


            # --- Post-processing ---
            if not extracted_text:
                log.warning(f"Could not extract significant textual content ({content_source}) from {url}")
                return {"url": url, "content": None, "message": f"No significant textual content found via {content_source} parser."}

            # Truncate content if it's too long
            limit = config.CONTEXT_TRUNCATION_LIMIT
            truncated = False
            if len(extracted_text) > limit:
                log.info(f"Content from {url} truncated from {len(extracted_text)} to {limit} characters.")
                # Simple truncation, could be smarter (e.g., preserve JSON structure)
                extracted_text = extracted_text[:limit]
                truncated = True

            result = {
                "url": url,
                "content_source": content_source,
                "content": extracted_text
            }
            if truncated:
                result["message"] = f"Content truncated to {limit} characters."

            log.info(f"Successfully browsed and extracted content from {url} (Source: {content_source}, Length: {len(extracted_text)} chars).")
            return result

        except requests.exceptions.Timeout:
            log.error(f"Request timed out while browsing {url}")
            return {"error": f"Request timed out accessing URL: {url}"}
        except requests.exceptions.RequestException as e:
            # Catch connection errors, HTTP errors, etc.
            log.error(f"Request failed for URL {url}: {e}")
            return {"error": f"Failed to retrieve URL {url}. Error: {e}"}
        except Exception as e:
            log.exception(f"Unexpected error during web browse for {url}: {e}")
            return {"error": f"Unexpected error browsing URL {url}: {e}"}
```

### `tools/web_search.py`

```python
# autonomous_agent/tools/web_search.py
import requests
import json
import traceback
from typing import Dict, Any

# Import base class and config
from .base import Tool
from config import SEARXNG_BASE_URL, SEARXNG_TIMEOUT

# Suggestion: Use logging module
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s][TOOL_WebSearch] %(message)s')
log = logging.getLogger(__name__)


class WebSearchTool(Tool):
    """Tool for performing web searches using a SearXNG instance."""
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Performs a web search using a SearXNG instance to find information. Parameters: query (str, required)"
        )
        # Config check message moved to agent init for better flow

    def run(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a search query against the configured SearXNG instance."""
        if not SEARXNG_BASE_URL: return {"error": "SearXNG base URL is not configured."}
        if not isinstance(parameters, dict): log.error(f"Invalid params type: {type(parameters)}"); return {"error": "Invalid parameters format."}
        query = parameters.get("query")
        if not query or not isinstance(query, str) or not query.strip(): return {"error": "Missing or invalid 'query' parameter."}

        log.info(f"Executing SearXNG Search: '{query}'")
        params = {'q': query, 'format': 'json'}
        searxng_url = SEARXNG_BASE_URL.rstrip('/') + "/search"
        headers = {'Accept': 'application/json'}

        try:
            response = requests.get(searxng_url, params=params, headers=headers, timeout=SEARXNG_TIMEOUT)
            response.raise_for_status() # Check for HTTP errors

            try: search_data = response.json()
            except json.JSONDecodeError: log.error(f"Failed SearXNG JSON decode. Status: {response.status_code}. Resp: {response.text[:200]}..."); return {"error": "Failed JSON decode from SearXNG."}

            processed = []
            # Process infoboxes
            if 'infoboxes' in search_data and isinstance(search_data['infoboxes'], list):
                for ib in search_data['infoboxes']:
                    if isinstance(ib, dict): content = ib.get('content'); title = ib.get('infobox_type', 'Infobox'); url = None;
                    if ib.get('urls') and isinstance(ib['urls'], list) and len(ib['urls']) > 0 and isinstance(ib['urls'][0], dict): url = ib['urls'][0].get('url');
                    if content: processed.append({"type": "infobox", "title": title, "snippet": content, "url": url})
            # Process regular results
            if 'results' in search_data and isinstance(search_data['results'], list):
                 for r in search_data['results'][:7]: # Limit results
                     if isinstance(r, dict): title = r.get('title'); url = r.get('url'); snippet = r.get('content');
                     if title and url and snippet: processed.append({"type": "organic", "title": title, "snippet": snippet, "url": url})
            # Process answers
            if 'answers' in search_data and isinstance(search_data['answers'], list):
                 for ans in search_data['answers']:
                      if isinstance(ans, str) and ans.strip(): processed.append({"type": "answer", "title": "Direct Answer", "snippet": ans, "url": None})

            if not processed:
                if 'error' in search_data: log.warning(f"SearXNG reported error: {search_data['error']}"); return {"error": f"SearXNG error: {search_data['error']}"}
                else: log.info("No relevant results found by SearXNG."); return {"results": [], "message": "No relevant results found."}

            log.info(f"SearXNG returned {len(processed)} processed results.")
            return {"results": processed}

        except requests.exceptions.Timeout: log.error(f"SearXNG request timed out ({SEARXNG_TIMEOUT}s) to {searxng_url}"); return {"error": f"Web search timed out."}
        except requests.exceptions.RequestException as e: log.error(f"SearXNG request failed: {e}"); return {"error": f"Web search connection/query failed."}
        except Exception as e: log.exception(f"Unexpected error during SearXNG search: {e}"); return {"error": f"Unexpected web search error."} # Log stack trace
```

