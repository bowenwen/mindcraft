# FILE: README.md
# MindCraft 🧠🤖

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Explore the capabilities of local Large Language Models (LLMs) with this autonomous agent framework designed for research, task management, and extensibility.**

This project provides a foundation for building autonomous agents that can:

*   Reason and plan using local LLMs via **Ollama**.
*   Maintain persistent memory using **ChromaDB** vector storage.
*   Utilize consolidated tools like `web` (search/browse), `memory` (search/write), and `file` (read/write/list) to interact with information and its workspace.
*   Manage a queue of tasks with priorities and dependencies.
*   Generate new tasks based on ongoing work or chat interactions.
*   Interact with users via a **Gradio** web interface.
*   Generate data suitable for fine-tuning models (**QLoRA format**).
*   Employ robust error handling within task steps, including retries for recoverable issues.
*   Summarize completed/failed tasks into memory and optionally prune detailed step memories to manage long-term memory growth.

The agent operates autonomously in the background, processing tasks while providing visibility and control through a web UI.

## ✨ Features

*   **🧠 Intelligent Core:** Leverages local LLMs (configurable via Ollama) for planning, reasoning, reflection, and task generation.
*   **💾 Persistent Vector Memory:** Uses ChromaDB to store and retrieve relevant information (task steps, tool results, reflections, chat history) based on semantic similarity.
*   **🔍 LLM-Powered Memory Re-ranking:** Enhances memory retrieval by using the LLM to re-rank initial candidates for contextual relevance.
*   **🛠️ Extensible Consolidated Tool System:**
    *   **`web` Tool:**
        *   `search` action: Integrated with SearXNG for versatile web searching.
        *   `browse` action: Can fetch and parse content from URLs (HTML, PDF, TXT, JSON).
    *   **`memory` Tool:**
        *   `search` action: Performs semantic search over the agent's memory.
        *   `write` action: Allows the agent to explicitly add content to its memory.
    *   **`file` Tool (Workspace Interaction):**
        *   `read` action: Reads text from files within a secure artifact workspace (`output/artifacts/`).
        *   `write` action: Writes text to files within the workspace (does not overwrite).
        *   `list` action: Lists files and directories within the workspace.
    *   Easily add new tools or actions by modifying tool classes and updating the agent's prompts/parsing.
*   **📋 Task Management:**
    *   Persistent task queue (`output/task_queue.json`).
    *   Supports task priorities.
    *   Supports task dependencies (tasks wait for prerequisites).
*   **🤖 Autonomous Operation:** Runs task processing loop in a background thread.
*   **🌐 Gradio Web Interface:**
    *   **Monitor Tab:** Visualize the agent's current task, step logs, recently accessed memories, and last browsed web content. Control buttons to Start/Pause the autonomous loop. *Updates automatically while the agent is running.*
    *   **Chat Tab:** Interact directly with the agent. Chat history is saved to memory, and interactions can trigger new task generation. Displays recently generated tasks and relevant memories for the chat context.
    *   **Agent State Tab:** View the agent's identity statement, task queues (pending, in-progress, completed, failed), memory summary, and explore task-specific or general memories. *Requires manual refresh using the "Load Agent State" button.*
*   **💡 Reactive Task Generation:** The agent can evaluate chat interactions or its idle state to generate new, relevant tasks (including exploratory ones) with priorities and dependencies.
*   **📊 QLoRA Dataset Generation:** Automatically saves completed task `(description, final_answer)` pairs and qualifying chat interactions to a `.jsonl` file suitable for fine-tuning.
*   **📝 Activity Summaries:** Generates daily summary logs (`output/summary/summary_YYYY-MM-DD.txt`) for high-level tracking.
*   **🔄 Robust Step Error Handling:** Implements configurable retries within task steps for recoverable errors (e.g., tool failures, LLM parsing issues), allowing the agent to attempt recovery before failing the entire task.
*   **🧠 Memory Summarization & Pruning:** Automatically generates LLM-based summaries of completed or failed tasks, storing the essence while optionally deleting the detailed step-by-step memories to manage long-term memory size and relevance (configurable).
*   **📁 Secure Artifact Workspace:** Tools interacting with the filesystem (`file` tool) are restricted to a designated subfolder (`output/artifacts/` by default) to enhance security.

## 🏗️ Architecture Overview

The project is structured into several key Python modules:

*   `app_ui.py`: Main entry point, defines the Gradio interface and orchestrates agent startup/shutdown. Manages UI updates.
*   `agent.py`: Contains the `AutonomousAgent` class, the central orchestrator managing the main loop, state, thinking process (including tool/action selection), error handling/recovery, memory summarization, and interactions between components.
*   `memory.py`: Implements the `AgentMemory` class for interacting with ChromaDB, handling embedding pre-checks, and supporting memory retrieval/deletion/metadata queries. Includes LLM-based re-ranking logic.
*   `task_manager.py`: Defines the `TaskQueue` for managing tasks and their persistence.
*   `data_structures.py`: Defines the `Task` class.
*   `utils.py`: Handles communication with the Ollama API, path sanitization/validation for file tools, and utility functions like relative time formatting.
*   `tools/`: A sub-package for consolidated tools:
    *   `base.py`: Abstract `Tool` base class.
    *   `web_tool.py`: Implements `search` and `browse` actions.
    *   `memory_tool.py`: Implements `search` and `write` actions.
    *   `file_tool.py`: Implements `read`, `write`, and `list` actions for the artifact workspace.
    *   `__init__.py`: Loads and registers the available tool instances (`web`, `memory`, `file`).
*   `config.py`: Loads and stores configuration from the `.env` file (including paths, model names, API keys, retry counts, memory summarization settings, and the `ARTIFACT_FOLDER` path).

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   An running **Ollama** instance ([ollama.com](https://ollama.com/)) accessible from where you run the script.
*   Required Ollama models pulled (e.g., `ollama pull gemma3`, `ollama pull nomic-embed-text`). Check `config.py` for defaults.
*   A running **SearXNG** instance ([docs.searxng.org](https://docs.searxng.org/)) accessible from where you run the script (or use a public one, configured in `.env`). Required for the `web` tool's `search` action.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/bowenwen/mindcraft.git # Or your repo URL
    cd mindcraft # Or your project directory
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
        *   `SEARXNG_BASE_URL`: URL of your SearXNG instance (e.g., `http://localhost:8080`). Needed for `web` tool's `search` action.
        *   *(Optional)* `ARTIFACT_FOLDER`: Path to the secure workspace for the `file` tool (defaults to `./output/artifacts`).
        *   **(Important!)** `ANONYMIZED_TELEMETRY=False`: Add this line to disable ChromaDB's default telemetry and prevent potential connection errors.
        *   *(Optional)* `AGENT_MAX_STEP_RETRIES`: Max retries for a single step before failing task (default: 2).
        *   *(Optional)* `ENABLE_MEMORY_SUMMARIZATION`: Enable task summary generation (default: True).
        *   *(Optional)* `DELETE_MEMORIES_AFTER_SUMMARY`: Delete detailed memories after summary (default: True).
        *   *(Optional)* Set other paths/timeouts defined in `config.py`.

### Running the Application

Run the Gradio application from the root directory of the project:

```bash
python app_ui.py
```

This will start the background agent thread (automatically resume on start) and launch the Gradio web interface. Open the URL provided in your terminal (usually `http://127.0.0.1:7860` or `http://0.0.0.0:7860`).

## How to use

The Gradio interface has three main tabs:

1.  **Agent Monitor:**
    *   **Controls:** Use "Start / Resume" and "Pause" to control the agent's autonomous background task processing. Use "Suggest Task Change" to add a memory hinting the agent should consider wrapping up its current task.
    *   **Agent Status:** Shows the current state (running, paused, idle, etc.) and timestamp.
    *   **Current Task:** Shows the ID, status, and description of the task the agent is currently focused on (or "Idle").
    *   **Last Step Log:** Displays detailed logs generated during the most recent processing step, including the tool, action, parameters used, and retry attempts if applicable.
    *   **Recent Memories:** Shows a summary of the latest memories retrieved by the agent relevant to its current step (may include `task_summary` types).
    *   **Last Web Content:** Displays the text content fetched by the last `web` tool `browse` action execution.
    *   **Last Final Answer:** Displays the final answer provided by the agent for the most recently completed task.
    *   ***This tab updates automatically every few seconds while the agent is running.***

2.  **Chat:**
    *   **Conversation:** Interact directly with the agent using natural language.
    *   **Relevant Memories (Chat):** Shows memories retrieved specifically for the context of the current chat turn.
    *   **Last Generated Task (Chat):** If your chat interaction warrants a background task, its description and ID will appear here.
    *   **Prioritize Task:** If a task was just generated, click this to increase its priority in the queue.
    *   **Inject Info for Task:** Type information in the main message box and click this button to add it to memory for the agent's *current* task context.
    *   Chat history is added to the agent's long-term memory (if the agent is running).
    *   Chat interactions deemed sufficiently complex trigger background task generation.
    *   Interactions leading to task generation are saved to the QLoRA dataset file.

3.  **Agent State:**
    *   **Controls:** Use the **"Load Agent State"** button to refresh all information on this tab. Use the "Refresh General Memories" button specifically for that table.
    *   **Agent Identity Statement:** Displays the agent's current self-defined identity.
    *   **Task Status:** Shows tables for Pending, In Progress, Completed, and Failed tasks with relevant details.
    *   **Memory Explorer:**
        *   Displays a summary count of memories by type (including `agent_explicit_memory_write`).
        *   Allows selecting a completed/failed task ID from a dropdown to view its specific memories.
        *   Shows a table of recent general memories (not tied to specific tasks, like chat or reflections).
    *   ***This tab requires clicking the "Load Agent State" button to see the latest information.***

## ⚙️ Configuration

Key settings are managed via the `.env` file. See `config.py` for all available options and their default values. Most importantly, set your Ollama and SearXNG URLs. Configure `AGENT_MAX_STEP_RETRIES`, `ENABLE_MEMORY_SUMMARIZATION`, `DELETE_MEMORIES_AFTER_SUMMARY`, and `ARTIFACT_FOLDER` as needed.

## 🔧 Extending the Agent

*   **Adding/Modifying Tool Actions:**
    1.  Open the relevant tool file (e.g., `tools/file_tool.py`).
    2.  Add a new private method for the action's logic (e.g., `_run_delete(self, filename)`).
    3.  Update the tool's `description` in `__init__` to include the new action and its parameters.
    4.  Modify the main `run` method to check for the new `action` string in the `parameters` dictionary and call your new private method.
    5.  Update `agent.py`:
        *   Modify `generate_thinking` prompt to teach the LLM about the new action and its required parameters within the `PARAMETERS` JSON.
        *   Adjust the parameter validation logic within `generate_thinking`'s parsing section to handle the new action's specific parameter requirements.
*   **Adding New Tools:**
    1.  Create a new Python file in `tools/` (e.g., `tools/calendar_tool.py`).
    2.  Define a class inheriting from `tools.base.Tool`.
    3.  Implement `__init__` (setting `name` and `description`, mentioning required actions/params).
    4.  Implement the `run` method, which should check `parameters['action']` and call internal logic. Ensure it returns a dictionary, including an `"error"` key on failure.
    5.  Import your new tool class in `tools/__init__.py` and add an instance to `AVAILABLE_TOOLS`.
    6.  Update `agent.py` (`get_available_tools_description`, `generate_thinking` prompt, parameter validation) to incorporate the new tool.
*   **Modifying Prompts:** Core prompts for agent thinking, task generation, reflection, summarization etc., are located within the methods of the `AutonomousAgent` class in `agent.py`. Remember to update the `PARAMETERS` format examples if tool actions change.
*   **Changing Models:** Update the `OLLAMA_CHAT_MODEL` and `OLLAMA_EMBED_MODEL` variables in your `.env` file. Ensure the chosen models are available in your Ollama instance.
*   **Adjusting Error Handling:** Modify `AGENT_MAX_STEP_RETRIES` in `.env` or refine the retry logic in `agent.py:_execute_step`.
*   **Adjusting Memory Strategy:** Toggle `ENABLE_MEMORY_SUMMARIZATION` and `DELETE_MEMORIES_AFTER_SUMMARY` in `.env`. Modify summarization prompts or logic in `agent.py:_summarize_and_prune_task_memories`.

## 🛣️ Future Work / Roadmap

*   [ ] Add more tool actions (e.g., `file` tool `delete`, `append` actions).
*   [ ] Add more tools (e.g., code execution sandbox, specific APIs like calendar).
*   [ ] Enhance task duplicate checking using semantic similarity instead of just string matching during task generation.
*   [ ] Introduce asynchronous processing for tool execution and potentially LLM calls to improve responsiveness.
*   [ ] Implement a more formal state machine for task/agent status.
*   [ ] Add comprehensive unit and integration tests.
*   [ ] Refine QLoRA data generation format and options.
*   [ ] Implement periodic pruning of *old, low-relevance* memories (independent of task summarization).
*   [ ] Integrate Python's `logging` module more deeply for configurable log levels, file output rotation, etc.
*   [ ] Improve Gradio UI feedback (e.g., show retry counts, memory summary status, loading indicators).

## 🙏 Contributing

Contributions are welcome! Please feel free to open an issue to discuss bugs or feature requests, or submit a pull request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.