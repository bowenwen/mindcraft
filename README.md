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
*   **[NEW]** Employ robust error handling within task steps, including retries for recoverable issues.
*   **[NEW]** Summarize completed/failed tasks into memory and optionally prune detailed step memories to manage long-term memory growth.

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
*   **🔄 Robust Step Error Handling:** Implements configurable retries within task steps for recoverable errors (e.g., tool failures, LLM parsing issues), allowing the agent to attempt recovery before failing the entire task.
*   **🧠 Memory Summarization & Pruning:** Automatically generates LLM-based summaries of completed or failed tasks, storing the essence while optionally deleting the detailed step-by-step memories to manage long-term memory size and relevance (configurable).

## 🏗️ Architecture Overview

The project is structured into several key Python modules within the `autonomous_agent` package:

*   `app_ui.py`: Main entry point, defines the Gradio interface and orchestrates agent startup/shutdown.
*   `agent.py`: Contains the `AutonomousAgent` class, the central orchestrator managing the main loop, state, thinking process, error handling/recovery, memory summarization, and interactions between components.
*   `memory.py`: Implements the `AgentMemory` class for interacting with ChromaDB, handling embedding pre-checks, and supporting memory retrieval/deletion/metadata queries.
*   `task_manager.py`: Defines the `TaskQueue` for managing tasks and their persistence.
*   `data_structures.py`: Defines the `Task` class.
*   `llm_utils.py`: Handles communication with the Ollama API.
*   `tools/`: A sub-package for tools:
    *   `base.py`: Abstract `Tool` base class.
    *   `web_search.py`: `WebSearchTool` using SearXNG.
    *   `web_browse.py`: `WebBrowserTool` using Requests, BeautifulSoup, PyMuPDF.
*   `config.py`: Loads and stores configuration from the `.env` file (including new settings for retries and memory summarization).

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
        *   *(Optional)* `AGENT_MAX_STEP_RETRIES`: Max retries for a single step before failing task (default: 2).
        *   *(Optional)* `ENABLE_MEMORY_SUMMARIZATION`: Enable task summary generation (default: True).
        *   *(Optional)* `DELETE_MEMORIES_AFTER_SUMMARY`: Delete detailed memories after summary (default: True).
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
    *   **Last Step Log:** Displays detailed logs generated during the most recent processing step, including retry attempts if applicable.
    *   **Recent Memories:** Shows a summary of the latest memories retrieved by the agent relevant to its current step (may include `task_summary` types).
    *   **Last Web Content:** Displays the text content fetched by the last `web_browse` tool execution.
2.  **Chat:**
    *   **Conversation:** Interact directly with the agent using natural language.
    *   **Relevant Memories (Chat):** Shows memories retrieved specifically for the context of the current chat turn.
    *   **Last Generated Task (Chat):** If your chat interaction warrants a background task, its description will appear here.
    *   Chat history is added to the agent's long-term memory.
    *   Chat interactions deemed sufficiently complex trigger background task generation.
    *   Interactions leading to task generation are saved to the QLoRA dataset file.

## ⚙️ Configuration

Key settings are managed via the `.env` file. See `config.py` for all available options and their default values. Most importantly, set your Ollama and SearXNG URLs. Configure `AGENT_MAX_STEP_RETRIES`, `ENABLE_MEMORY_SUMMARIZATION`, and `DELETE_MEMORIES_AFTER_SUMMARY` as needed.

## 🔧 Extending the Agent

*   **Adding Tools:**
    1.  Create a new Python file in the `tools/` directory (e.g., `tools/my_tool.py`).
    2.  Define a class that inherits from `tools.base.Tool`.
    3.  Implement the `__init__` (setting `name` and `description`) and `run` methods. **Ensure the `run` method returns a dictionary, including an `"error"` key if the tool fails internally.**
    4.  Import your new tool class in `tools/__init__.py` and add an instance to the `AVAILABLE_TOOLS` dictionary.
*   **Modifying Prompts:** Core prompts for agent thinking, task generation, reflection, summarization etc., are located within the methods of the `AutonomousAgent` class in `agent.py`.
*   **Changing Models:** Update the `OLLAMA_CHAT_MODEL` and `OLLAMA_EMBED_MODEL` variables in your `.env` file. Ensure the chosen models are available in your Ollama instance.
*   **Adjusting Error Handling:** Modify `AGENT_MAX_STEP_RETRIES` in `.env` or refine the retry logic in `agent.py:_execute_step`.
*   **Adjusting Memory Strategy:** Toggle `ENABLE_MEMORY_SUMMARIZATION` and `DELETE_MEMORIES_AFTER_SUMMARY` in `.env`. Modify summarization prompts or logic in `agent.py:_summarize_and_prune_task_memories`.

## 🛣️ Future Work / Roadmap

*   [ ] Add more tools (e.g., file system access, code execution sandbox, specific APIs).
*   [ ] Enhance task duplicate checking using semantic similarity instead of just string matching during task generation.
*   [ ] Introduce asynchronous processing for tool execution and potentially LLM calls to improve responsiveness.
*   [ ] Implement a more formal state machine for task/agent status.
*   [ ] Add comprehensive unit and integration tests.
*   [ ] Refine QLoRA data generation format and options.
*   [ ] Implement periodic pruning of *old, low-relevance* memories (independent of task summarization).
*   [ ] Integrate Python's `logging` module more deeply for configurable log levels, file output rotation, etc.
*   [ ] Improve Gradio UI feedback (e.g., show retry counts, memory summary status).

## 🙏 Contributing

Contributions are welcome! Please feel free to open an issue to discuss bugs or feature requests, or submit a pull request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.