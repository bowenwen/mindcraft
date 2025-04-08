# FILE: agent.py
# autonomous_agent/agent.py
import os
import json
import time
import datetime
import uuid
import re
import traceback
import threading  # Keep standard import
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, deque

# --- Project Imports ---
import config
from data_structures import Task
from task_manager import TaskQueue
from memory import AgentMemory, setup_chromadb  # Import setup_chromadb
from tools import load_tools
import prompts

# Use specific helpers from utils
from utils import (
    call_ollama_api,
    format_relative_time,
    check_ollama_status,
    check_searxng_status,
)
import chromadb

# --- Logging Setup ---
import logging

# Agent-specific loggers will inherit from root logger configured in app_ui
log = logging.getLogger("AGENT")  # Use a base name, context added in messages


class AutonomousAgent:
    def __init__(self, agent_id: str, agent_config: Dict[str, Any]):
        self.agent_id = agent_id
        self.agent_name = agent_config.get("name", agent_id)
        self.log_prefix = f"[Agent:{self.agent_id}({self.agent_name})]"
        log.info(f"{self.log_prefix} Initializing AutonomousAgent...")

        # Agent-specific paths
        self.task_queue_path = config.get_agent_task_queue_path(agent_id)
        self.session_state_path = config.get_agent_state_path(agent_id)
        self.qlora_dataset_path = config.get_agent_qlora_dataset_path(agent_id)
        self.memory_db_path = config.get_agent_db_path(agent_id)

        # Agent-specific components
        self.task_queue = TaskQueue(agent_id=self.agent_id)  # Uses agent-specific path

        # Initialize Memory DB
        log.info(f"{self.log_prefix} Setting up ChromaDB Memory...")
        self.memory_collection: Optional[chromadb.Collection] = setup_chromadb(
            agent_id=self.agent_id
        )
        if self.memory_collection is None:
            # Log critical error and raise immediately
            log.critical(
                f"{self.log_prefix} CRITICAL: Memory collection failed to initialize for agent {agent_id}."
            )
            raise ValueError(
                f"Memory collection failed to initialize for agent {agent_id}."
            )
        log.info(f"{self.log_prefix} ChromaDB Memory Collection initialized.")

        # Initialize AgentMemory instance
        log.info(f"{self.log_prefix} Initializing AgentMemory component...")
        try:
            self.memory = AgentMemory(
                collection=self.memory_collection,
                ollama_chat_model=config.OLLAMA_CHAT_MODEL,
                ollama_base_url=config.OLLAMA_BASE_URL,
            )
            # Perform an early check if DB is usable
            db_count = self.memory.collection.count()
            log.info(
                f"{self.log_prefix} AgentMemory component initialized successfully. DB Count: {db_count}"
            )
        except Exception as mem_init_err:
            log.critical(
                f"{self.log_prefix} CRITICAL: Failed to initialize AgentMemory instance: {mem_init_err}",
                exc_info=True,
            )
            raise mem_init_err  # Propagate error

        # Tools are shared
        self.tools = load_tools()
        self.ollama_base_url = config.OLLAMA_BASE_URL
        self.ollama_chat_model = config.OLLAMA_CHAT_MODEL
        self.ollama_embed_model = config.OLLAMA_EMBED_MODEL

        # Agent-specific state
        self.identity_statement: str = agent_config.get(
            "initial_identity", f"Default identity for {agent_id}"
        )
        self.initial_tasks_prompt_key: str = agent_config.get(
            "initial_tasks_prompt", "INITIAL_CREATIVE_TASK_GENERATION_PROMPT_AGENT_01"
        )

        # Threading and control
        self._is_running = threading.Event()
        self._shutdown_request = threading.Event()
        self._agent_thread: Optional[threading.Thread] = None
        # --- USE RLock for reentrant locking ---
        self._state_lock = threading.RLock()

        # Initialize session state with defaults before loading
        self.session_state = {
            "agent_id": self.agent_id,
            "current_task_id": None,
            "current_action_retries": 0,
            "last_checkpoint": None,
            "last_web_browse_content": None,
            "identity_statement": self.identity_statement,
            "user_suggestion_move_on_pending": False,
            "last_action_details": deque(maxlen=config.UI_STEP_HISTORY_LENGTH),
            "tasks_since_last_revision": 0,  # <<<--- ADDED state variable
        }
        log.info(f"{self.log_prefix} Loading session state...")
        self.load_session_state()  # Load agent-specific state

        # UI State (Reflects this agent's status)
        self._ui_update_state: Dict[str, Any] = {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": "paused",  # Default to paused on init
            "log": f"{self.log_prefix} Agent initialized, paused.",
            "current_task_id": self.session_state.get("current_task_id"),
            "current_task_desc": "N/A",
            "current_action_desc": "N/A",
            "current_plan": "N/A",
            "thinking": "(No thinking process yet)",
            "dependent_tasks": [],
            "last_action_type": None,
            "last_tool_results": None,
            "recent_memories": [],
            "last_web_content": self.session_state.get(
                "last_web_browse_content", "(None)"
            ),
            "final_answer": None,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "action_history": list(self.session_state.get("last_action_details", [])),
        }

        self._refresh_ui_task_details()  # Populate task details if loaded

        log.info(f"{self.log_prefix} Agent Initialized Successfully.")

    # _refresh_ui_task_details remains the same
    def _refresh_ui_task_details(self):
        current_task_id = self.session_state.get("current_task_id")
        if current_task_id:
            task = self.task_queue.get_task(current_task_id)
            if task:
                self._ui_update_state["current_task_desc"] = task.description
                plan_status = "(No Plan Yet)"
                action_status = f"Status: {task.status}"
                if task.plan:
                    plan_status = "**Intended Plan (Guidance Only):**\n" + "\n".join(
                        [f"- {s}" for i, s in enumerate(task.plan)]
                    )
                if task.status == "in_progress":
                    action_status = f"Executing actions towards task goal (Attempt {task.reattempt_count + 1})"
                elif task.status == "planning":
                    action_status = f"Planning (Attempt {task.reattempt_count + 1})"
                    plan_status = "(Generating...)"
                self._ui_update_state["current_action_desc"] = action_status
                self._ui_update_state["current_plan"] = plan_status
            else:
                self._ui_update_state["current_task_desc"] = "Task Not Found"
                self._ui_update_state["current_action_desc"] = "N/A"
                self._ui_update_state["current_plan"] = "N/A"
        else:
            self._ui_update_state["current_task_desc"] = "N/A"
            self._ui_update_state["current_action_desc"] = "N/A"
            self._ui_update_state["current_plan"] = "N/A"

    # _revise_identity_statement <<< MODIFIED >>>
    def _revise_identity_statement(self, reason: str):
        log.info(f"{self.log_prefix} Revising identity statement. Reason: {reason}")

        # --- Gather Context ---
        # 1. Memories
        mem_query = f"Recent task summaries, self-reflections, session reflections, errors, key accomplishments, or notable interactions relevant to understanding my evolution, capabilities, and purpose. Focus on the last {config.IDENTITY_REVISION_TASK_INTERVAL} completed/failed tasks."
        log.info(f"{self.log_prefix} Retrieving memories for identity revision...")
        try:
            relevant_memories, _ = self.memory.retrieve_and_rerank_memories(
                query=mem_query,
                task_description="Reflecting on identity",
                context=f"Reason: {reason}",
                identity_statement=self.identity_statement,
                n_results=config.MEMORY_COUNT_IDENTITY_REVISION * 2,
                n_final=config.MEMORY_COUNT_IDENTITY_REVISION,
            )
        except Exception as e:
            log.error(
                f"{self.log_prefix} Failed retrieving memories for identity revision: {e}",
                exc_info=True,
            )
            relevant_memories = []

        memory_context_list = []
        for m in relevant_memories:
            relative_time = format_relative_time(m["metadata"].get("timestamp"))
            mem_type = m["metadata"].get("type", "N/A")
            content_snippet = m["content"][:300]
            memory_context_list.append(
                f"- [Memory - {relative_time}] (Type: {mem_type}):\n  {content_snippet}..."
            )
        memory_context = (
            "\n".join(memory_context_list)
            if memory_context_list
            else "No specific memories selected for this revision."
        )

        # 2. Task Summaries
        log.info(
            f"{self.log_prefix} Retrieving task summaries for identity revision..."
        )
        tasks_structured = self.task_queue.get_all_tasks_structured()
        pending_tasks = tasks_structured.get("pending", [])
        completed_tasks = tasks_structured.get("completed", [])
        failed_tasks = tasks_structured.get("failed", [])

        # Format Pending Tasks Summary (Top 5 by priority)
        pending_summary_list = []
        for task in pending_tasks[:5]:
            pending_summary_list.append(
                f"  - Prio {task.get('Priority', 'N/A')}: {task.get('Description', 'N/A')[:80]}..."
            )
        pending_tasks_summary = (
            "\n".join(pending_summary_list) if pending_summary_list else "  None"
        )

        # Format Completed/Failed Tasks Summary (Top 10 recent)
        comp_fail_summary_list = []
        # Combine and sort by completion/failure time (using completed_at or failed_at)
        all_finished = sorted(
            completed_tasks + failed_tasks,
            key=lambda t: t.get("Completed At")
            or t.get("Failed At", t.get("Created", "0")),
            reverse=True,
        )
        for task in all_finished[:10]:
            status = "COMPLETED" if "Completed At" in task else "FAILED"
            reason_snippet = ""
            if status == "FAILED":
                reason = task.get("Reason", "N/A")
                reason_snippet = f" (Reason: {reason[:50]}...)"
            comp_fail_summary_list.append(
                f"  - [{status}] {task.get('Description', 'N/A')[:80]}...{reason_snippet}"
            )
        completed_failed_tasks_summary = (
            "\n".join(comp_fail_summary_list)
            if comp_fail_summary_list
            else "  None recently finished."
        )

        # --- Call LLM for Revision ---
        prompt = prompts.REVISE_IDENTITY_PROMPT.format(
            identity_statement=self.identity_statement,
            reason=reason,
            memory_context=memory_context,
            pending_tasks_summary=pending_tasks_summary,
            completed_failed_tasks_summary=completed_failed_tasks_summary,
        )
        log.info(
            f"{self.log_prefix} Asking {self.ollama_chat_model} to revise identity statement..."
        )
        revised_statement_text = call_ollama_api(
            prompt, self.ollama_chat_model, self.ollama_base_url, timeout=150
        )

        # --- Process Revision ---
        if (
            revised_statement_text
            and revised_statement_text.strip()
            and len(revised_statement_text) > 20
        ):
            old_statement = self.identity_statement
            with self._state_lock:
                self.identity_statement = revised_statement_text.strip()
                self.session_state["identity_statement"] = self.identity_statement
            log.info(
                f"{self.log_prefix} Identity statement revised:\nOld: {old_statement}\nNew: {self.identity_statement}"
            )
            # Add memory outside lock, check running state
            if self._is_running.is_set() or self._ui_update_state.get("status") in [
                "paused",
                "idle",
            ]:
                try:
                    self.memory.add_memory(
                        f"Identity Statement Updated (Reason: {reason}):\n{self.identity_statement}",
                        {"type": "identity_revision", "reason": reason},
                    )
                except Exception as e:
                    log.error(
                        f"{self.log_prefix} Failed to add identity revision memory: {e}"
                    )
            else:
                log.info(
                    f"{self.log_prefix} Agent not running/paused, skipping memory add for identity revision."
                )
            self.save_session_state()  # Save state including the new identity
        else:
            log.warning(
                f"{self.log_prefix} Failed to get a valid revised identity statement from LLM (Response: '{revised_statement_text}'). Keeping current."
            )
            # Add failure memory if running/paused
            if self._is_running.is_set() or self._ui_update_state.get("status") in [
                "paused",
                "idle",
            ]:
                try:
                    self.memory.add_memory(
                        f"Identity statement revision failed (Reason: {reason}). LLM response insufficient.",
                        {"type": "identity_revision_failed", "reason": reason},
                    )
                except Exception as e:
                    log.error(
                        f"{self.log_prefix} Failed to add identity revision failure memory: {e}"
                    )
            else:
                log.info(
                    f"{self.log_prefix} Agent not running/paused, skipping memory add for failed identity revision."
                )

    # _update_ui_state remains the same
    def _update_ui_state(self, **kwargs):
        # Always acquire lock when updating shared UI state dict
        with self._state_lock:  # RLock handles reentrancy
            self._ui_update_state["timestamp"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
            for key, value in kwargs.items():
                self._ui_update_state[key] = value
            # Ensure action history is updated within the lock
            self._ui_update_state["action_history"] = list(
                self.session_state.get("last_action_details", [])
            )
            # Refresh task details based on potentially updated task ID/state
            self._refresh_ui_task_details()

    # get_ui_update_state remains the same
    def get_ui_update_state(self) -> Dict[str, Any]:
        with self._state_lock:  # Acquire lock for reading shared state (RLock)
            # Always refresh task details before returning
            self._refresh_ui_task_details()
            # Ensure agent ID and name are present
            self._ui_update_state["agent_id"] = self.agent_id
            self._ui_update_state["agent_name"] = self.agent_name
            # Return a copy to prevent modification outside the lock
            return self._ui_update_state.copy()

    # load_session_state uses agent-specific path, logs agent ID <<< MODIFIED >>>
    def load_session_state(self):
        if os.path.exists(self.session_state_path):
            try:
                with open(self.session_state_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if not content or not content.strip():
                    log.info(
                        f"{self.log_prefix} Session state file '{self.session_state_path}' empty."
                    )
                    # Ensure defaults are set even if file is empty
                    self.session_state.setdefault(
                        "identity_statement", self.identity_statement
                    )
                    self.session_state.setdefault(
                        "user_suggestion_move_on_pending", False
                    )
                    self.session_state.setdefault(
                        "last_action_details",
                        deque(maxlen=config.UI_STEP_HISTORY_LENGTH),
                    )
                    self.session_state.setdefault(
                        "tasks_since_last_revision", 0
                    )  # <<< Default
                    return

                loaded = json.loads(content)
                if isinstance(loaded, dict):
                    loaded_agent_id = loaded.get("agent_id")
                    if loaded_agent_id and loaded_agent_id != self.agent_id:
                        log.warning(
                            f"{self.log_prefix} Loaded session state is for a different agent ('{loaded_agent_id}'). Discarding loaded state."
                        )
                        # Keep initial defaults
                        self.session_state.setdefault(
                            "tasks_since_last_revision", 0
                        )  # <<< Default
                        return

                    # Safely update with defaults for missing keys
                    loaded.setdefault("last_web_browse_content", None)
                    loaded.setdefault(
                        "current_action_retries", loaded.pop("current_task_retries", 0)
                    )
                    loaded.setdefault("user_suggestion_move_on_pending", False)
                    loaded.setdefault("tasks_since_last_revision", 0)  # <<< Default
                    loaded.pop("investigation_context", None)  # Remove old key

                    # Load identity, fallback to initial config if missing/empty
                    loaded_identity = loaded.get("identity_statement", "").strip()
                    if not loaded_identity:
                        loaded_identity = (
                            self.identity_statement
                        )  # Use initial from config

                    # Load history, handle old key 'last_step_details'
                    loaded_history = loaded.get(
                        "last_action_details", loaded.pop("last_step_details", None)
                    )
                    if isinstance(loaded_history, list):
                        # Load into deque, ensuring max length
                        loaded["last_action_details"] = deque(
                            loaded_history[-config.UI_STEP_HISTORY_LENGTH :],
                            maxlen=config.UI_STEP_HISTORY_LENGTH,
                        )
                    else:
                        log.warning(
                            f"{self.log_prefix} Invalid or missing history found in state file. Initializing empty."
                        )
                        loaded["last_action_details"] = deque(
                            maxlen=config.UI_STEP_HISTORY_LENGTH
                        )

                    # Update internal state carefully
                    with self._state_lock:  # RLock
                        # Update only the keys present in loaded, keeping defaults otherwise
                        self.session_state.update(loaded)
                        # Ensure instance identity matches loaded/defaulted state
                        self.identity_statement = loaded_identity
                        self.session_state["identity_statement"] = (
                            self.identity_statement
                        )

                    log.info(
                        f"{self.log_prefix} Loaded session state from {self.session_state_path}."
                    )
                    log.info(
                        f"{self.log_prefix}   User Suggestion Pending: {self.session_state.get('user_suggestion_move_on_pending')}"
                    )
                    log.info(
                        f"{self.log_prefix}   Tasks Since Last Revision: {self.session_state.get('tasks_since_last_revision', 0)}"
                    )  # <<< Log loaded value
                    log.info(
                        f"{self.log_prefix}   Loaded {len(self.session_state['last_action_details'])} action history entries."
                    )

                else:
                    log.warning(
                        f"{self.log_prefix} Session state file '{self.session_state_path}' invalid format (not a dict). Using defaults."
                    )
                    # Ensure defaults are set
                    self.session_state.setdefault(
                        "identity_statement", self.identity_statement
                    )
                    self.session_state.setdefault(
                        "user_suggestion_move_on_pending", False
                    )
                    self.session_state.setdefault(
                        "last_action_details",
                        deque(maxlen=config.UI_STEP_HISTORY_LENGTH),
                    )
                    self.session_state.setdefault(
                        "tasks_since_last_revision", 0
                    )  # <<< Default

            except json.JSONDecodeError as e:
                log.warning(
                    f"{self.log_prefix} Failed loading session state JSON '{self.session_state_path}': {e}. Using defaults."
                )
                # Ensure defaults are set
                self.session_state.setdefault(
                    "identity_statement", self.identity_statement
                )
                self.session_state.setdefault("user_suggestion_move_on_pending", False)
                self.session_state.setdefault(
                    "last_action_details", deque(maxlen=config.UI_STEP_HISTORY_LENGTH)
                )
                self.session_state.setdefault(
                    "tasks_since_last_revision", 0
                )  # <<< Default
            except Exception as e:
                log.warning(
                    f"{self.log_prefix} Failed loading session state '{self.session_state_path}': {e}. Using defaults.",
                    exc_info=True,
                )
                # Ensure defaults are set
                self.session_state.setdefault(
                    "identity_statement", self.identity_statement
                )
                self.session_state.setdefault("user_suggestion_move_on_pending", False)
                self.session_state.setdefault(
                    "last_action_details", deque(maxlen=config.UI_STEP_HISTORY_LENGTH)
                )
                self.session_state.setdefault(
                    "tasks_since_last_revision", 0
                )  # <<< Default
        else:
            log.info(
                f"{self.log_prefix} Session state file '{self.session_state_path}' not found. Using defaults."
            )
            # Ensure defaults are set
            self.session_state.setdefault("identity_statement", self.identity_statement)
            self.session_state.setdefault("user_suggestion_move_on_pending", False)
            self.session_state.setdefault(
                "last_action_details", deque(maxlen=config.UI_STEP_HISTORY_LENGTH)
            )
            self.session_state.setdefault("tasks_since_last_revision", 0)  # <<< Default

    # save_session_state uses agent-specific path, logs agent ID <<< MODIFIED >>>
    def save_session_state(self):
        # Save should be relatively quick file IO, but acquire lock for consistency
        with self._state_lock:  # RLock
            try:
                self.session_state["last_checkpoint"] = datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat()
                self.session_state["identity_statement"] = self.identity_statement
                self.session_state["agent_id"] = self.agent_id
                # Ensure tasks_since_last_revision is present (redundant if init/load work, but safe)
                self.session_state.setdefault("tasks_since_last_revision", 0)
                self.session_state.pop(
                    "investigation_context", None
                )  # Clean old key if exists

                # Create a serializable copy
                state_to_save = self.session_state.copy()
                # Convert deque to list for JSON serialization
                state_to_save["last_action_details"] = list(
                    self.session_state["last_action_details"]
                )

                # Perform file write outside the critical section if possible,
                # but here state_to_save needs protection, so keep it inside.
                with open(self.session_state_path, "w", encoding="utf-8") as f:
                    json.dump(state_to_save, f, indent=2, ensure_ascii=False)
                # log.debug(f"{self.log_prefix} Session state saved to '{self.session_state_path}'.") # Reduce noise
            except Exception as e:
                log.error(
                    f"{self.log_prefix} Error saving session state to '{self.session_state_path}': {e}",
                    exc_info=True,
                )

    # handle_user_suggestion_move_on remains the same logic, logs agent ID
    def handle_user_suggestion_move_on(self) -> str:
        feedback = f"{self.log_prefix} Suggestion ignored: Agent not initialized."
        current_task_id = self.session_state.get("current_task_id")
        agent_running = self._is_running.is_set()
        flag_set_successfully = False

        if agent_running:
            if current_task_id:
                task = self.task_queue.get_task(current_task_id)
                if task and task.status in ["planning", "in_progress"]:
                    with self._state_lock:  # Lock needed to modify session_state (RLock)
                        if (
                            self._is_running.is_set()
                        ):  # Double check running status inside lock
                            log.info(
                                f"{self.log_prefix} User suggested moving on from task {current_task_id}. Setting flag."
                            )
                            self.session_state["user_suggestion_move_on_pending"] = True
                            flag_set_successfully = True
                        else:
                            log.info(
                                f"{self.log_prefix} Agent paused just before setting move_on flag for task {current_task_id}. Ignoring suggestion."
                            )
                            feedback = f"Suggestion ignored: Agent paused just before flag could be set. (Agent: {self.agent_id}, Task: {current_task_id})."

                    if flag_set_successfully:
                        feedback = f"Suggestion noted for current task ({current_task_id}) for Agent {self.agent_id}. Agent will consider wrapping up."
                        self.save_session_state()  # Save the updated state
                        # Add memory outside the lock
                        try:
                            self.memory.add_memory(
                                f"User Suggestion: Consider moving on from task {current_task_id}.",
                                {
                                    "type": "user_suggestion_move_on",
                                    "task_id": current_task_id,
                                },
                            )
                        except Exception as e:
                            log.error(
                                f"{self.log_prefix} Error adding move_on suggestion memory after setting flag: {e}"
                            )
                else:
                    log.info(
                        f"{self.log_prefix} User suggested moving on, but task {current_task_id} is not active ({task.status if task else 'Not Found'})."
                    )
                    feedback = f"Suggestion ignored: Task {current_task_id} is not currently being executed (Status: {task.status if task else 'Not Found'})."
            else:
                log.info(
                    f"{self.log_prefix} User suggested task change, but agent is idle."
                )
                feedback = f"{self.log_prefix} Suggestion ignored: Agent is currently idle (no active task)."
        else:  # Agent is paused
            # Check if flag is ALREADY pending from previous state
            is_pending = self.session_state.get(
                "user_suggestion_move_on_pending", False
            )
            log.info(
                f"{self.log_prefix} Agent paused, ignoring suggestion to move on from task {current_task_id or 'N/A'} (flag not set). Pending state: {is_pending}"
            )
            feedback = f"{self.log_prefix} Suggestion noted, but agent is paused. Flag not set. (Task: {current_task_id or 'N/A'}). Current pending: {is_pending}."
        return feedback

    # create_task remains the same
    def create_task(
        self,
        description: str,
        priority: int = 1,
        depends_on: Optional[List[str]] = None,
    ) -> Optional[str]:
        return self.task_queue.add_task(
            Task(description, priority, depends_on=depends_on)
        )

    # get_available_tools_description updates file tool path example
    def get_available_tools_description(self) -> str:
        if not self.tools:
            return "No tools available."
        active_tools_desc = []
        for name, tool in self.tools.items():
            is_active = True
            tool_description = tool.description
            # Check dependencies/configuration for each tool
            if name == "web":
                if not config.SEARXNG_BASE_URL:
                    log.warning(
                        f"{self.log_prefix} Web tool 'search' action disabled: SEARXNG_BASE_URL not set."
                    )
                    tool_description += (
                        "\n  *Note: Search functionality requires SEARXNG_BASE_URL.*"
                    )
                # Check if the *instance* has access to the shared doc DB
                web_tool_instance = self.tools.get("web")
                if web_tool_instance and (
                    not hasattr(web_tool_instance, "doc_archive_collection")
                    or web_tool_instance.doc_archive_collection is None
                ):
                    log.warning(
                        f"{self.log_prefix} Web tool 'browse' with query disabled: Shared Doc Archive DB not available to tool instance."
                    )
                    tool_description += "\n  *Note: Browse with 'query' (focused retrieval) requires functional Document Archive DB.*"

            if name == "memory":
                # Check if the *instance* has a working memory component
                if not self.memory or not self.memory.collection:
                    log.warning(
                        f"{self.log_prefix} Memory tool disabled: Agent's memory component not available."
                    )
                    is_active = False

            if name == "file":
                if not config.SHARED_ARTIFACT_FOLDER:
                    log.warning(
                        f"{self.log_prefix} File tool disabled: SHARED_ARTIFACT_FOLDER not set."
                    )
                    is_active = False
                else:
                    # Update description dynamically if needed (already done in __init__)
                    pass

            if name == "status":
                is_active = True  # Status tool has no external dependencies

            if is_active:
                active_tools_desc.append(
                    f"- Tool Name: **{name}**\n  Description: {tool_description}"
                )
        if not active_tools_desc:
            return "No tools currently active or available (check configuration and initialization)."
        return "\n".join(active_tools_desc)

    # _generate_task_plan remains the same logic, logs agent ID
    def _generate_task_plan(self, task: Task) -> bool:
        log.info(
            f"{self.log_prefix} Generating execution plan for task {task.id[:8]} (Attempt: {task.reattempt_count + 1})..."
        )
        tool_desc = self.get_available_tools_description()
        lessons_learned_context = ""
        if task.reattempt_count > 0:
            log.info(
                f"{self.log_prefix} Retrieving lessons learned for task {task.id[:8]} re-plan..."
            )
            try:
                lesson_memories = self.memory.get_memories_by_metadata(
                    filter_dict={
                        "$and": [{"task_id": task.id}, {"type": "lesson_learned"}]
                    },
                    limit=config.MEMORY_COUNT_PLANNING,
                )
            except Exception as e:
                log.error(f"{self.log_prefix} Failed to retrieve lesson memories: {e}")
                lesson_memories = []

            if lesson_memories:
                lessons = []
                for mem in sorted(
                    lesson_memories,
                    key=lambda m: m.get("metadata", {}).get("timestamp", "0"),
                    reverse=True,
                ):
                    lessons.append(
                        f"- [{format_relative_time(mem['metadata'].get('timestamp'))}] {mem['content']}"
                    )
                lessons_learned_context = (
                    "**Recent Lessons Learned (Consider for this plan!):**\n"
                    + "\n".join(lessons)
                )
                log.info(
                    f"{self.log_prefix} Found {len(lessons)} relevant lessons learned."
                )
            else:
                log.info(
                    f"{self.log_prefix} No specific 'lesson_learned' memories found for this task re-plan."
                )
                lessons_learned_context = "**Note:** This is a re-attempt, but no specific 'lesson_learned' memories were found. Review previous findings carefully."
        else:
            lessons_learned_context = (
                "**Note:** First attempt, no previous lessons learned for this task."
            )
        prompt = prompts.GENERATE_TASK_PLAN_PROMPT.format(
            identity_statement=self.identity_statement,
            task_description=task.description,
            tool_desc=tool_desc,
            max_steps=config.MAX_STEPS_GENERAL_THINKING,
            lessons_learned_context=lessons_learned_context,
        )
        log.info(
            f"{self.log_prefix} Asking {self.ollama_chat_model} to generate plan..."
        )
        llm_response = call_ollama_api(
            prompt, self.ollama_chat_model, self.ollama_base_url, timeout=150
        )
        if not llm_response or not llm_response.strip():
            log.error(
                f"{self.log_prefix} LLM failed to generate plan for task {task.id[:8]}."
            )
            return False
        plan_steps = []
        try:
            # Try parsing numbered list first
            raw_steps = re.findall(r"^\s*\d+\.\s+(.*)", llm_response, re.MULTILINE)
            if not raw_steps:
                # Fallback: Try parsing lines that don't look like preamble/metadata
                lines = llm_response.strip().split("\n")
                potential_steps = [
                    line.strip()
                    for line in lines
                    if line.strip()
                    and not line.strip().startswith(
                        ("```", "**", "Plan:", "Here is", "Note:", "Example:")
                    )
                    and len(line.strip()) > 5  # Basic check for meaningful content
                ]
                # If the first potential step starts with '1.' or '-', assume it's a list
                if potential_steps and (potential_steps[0].startswith(("1.", "-"))):
                    raw_steps = [
                        re.sub(r"^\s*[\d.\-\*]+\s*", "", s) for s in potential_steps
                    ]  # Remove list markers
                elif potential_steps:
                    log.warning(
                        f"{self.log_prefix} Plan parsing used fallback line-by-line method. Response:\n{llm_response}"
                    )
                    raw_steps = (
                        potential_steps  # Keep lines as is if no clear list format
                    )

            if not raw_steps:
                log.error(
                    f"{self.log_prefix} Could not parse any plan steps from LLM response for task {task.id[:8]}:\n{llm_response}"
                )
                return False  # Return False if no steps found after all attempts

            plan_steps = [step.strip() for step in raw_steps if step.strip()]
            if not plan_steps:
                log.error(
                    f"{self.log_prefix} Parsed plan steps list is empty for task {task.id[:8]}:\n{llm_response}"
                )
                return False

            if len(plan_steps) > config.MAX_STEPS_GENERAL_THINKING:
                log.warning(
                    f"{self.log_prefix} LLM generated {len(plan_steps)} steps, exceeding limit {config.MAX_STEPS_GENERAL_THINKING}. Truncating."
                )
                plan_steps = plan_steps[: config.MAX_STEPS_GENERAL_THINKING]

            log.info(
                f"{self.log_prefix} Generated plan for task {task.id[:8]} with {len(plan_steps)} steps."
            )
            for i, step in enumerate(plan_steps):
                if len(step) < 5:
                    log.warning(
                        f"{self.log_prefix} Plan step {i+1} seems very short: '{step}'"
                    )

            task.plan = plan_steps
            if task.reattempt_count > 0:
                task.cumulative_findings += f"\n--- Generated New Plan (Attempt {task.reattempt_count + 1}) ---\n"
            else:
                task.cumulative_findings = "Plan generated. Starting execution.\n"
            task.current_step_index = 0
            return True
        except Exception as e:
            log.exception(
                f"{self.log_prefix} Failed to parse LLM plan response for task {task.id[:8]}: {e}\nResponse:\n{llm_response}"
            )
            task.plan = None
            return False

    # generate_thinking remains the same (includes detailed tool results)
    def generate_thinking(
        self,
        task: Task,
        tool_results: Optional[Dict[str, Any]] = None,
        user_suggested_move_on: bool = False,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        plan_context = "(No plan generated)"
        if task.plan:
            plan_context = "**Intended Plan (Guidance Only):**\n" + "\n".join(
                [f"- {s}" for s in task.plan]
            )
        cumulative_findings = (
            task.cumulative_findings.strip()
            if task.cumulative_findings and task.cumulative_findings.strip()
            else "(No findings yet)"
        )
        tool_desc = self.get_available_tools_description()

        # --- Memory Retrieval ---
        memory_context_str = "No relevant memories selected."
        user_provided_info_str = None
        try:
            memory_query = f"Info relevant to achieving task goal? Task: {task.description}\nConsider overall goal, previous findings, user suggestions, errors, lessons learned, file organization needs."
            if tool_results:
                # Create a concise summary of tool results for the memory query
                tool_name = tool_results.get("tool_name", "Unknown")
                tool_action = tool_results.get("action", "unknown")
                tool_status = tool_results.get("status", "unknown")
                tool_error = tool_results.get("error")
                result_payload = tool_results.get("result", {})
                archive_info = ""
                if isinstance(result_payload, dict):
                    archive_path = result_payload.get("archived_filepath")
                    archive_info = (
                        f" (Archived previous to: {archive_path})"
                        if archive_path
                        else ""
                    )
                # Keep summary very brief for memory query context
                tool_result_summary = (
                    f"Tool: {tool_name}/{tool_action}, Status: {tool_status}"
                )
                if tool_error:
                    tool_result_summary += f", Error: {str(tool_error)[:50]}..."
                elif isinstance(result_payload, dict) and result_payload.get("message"):
                    tool_result_summary += f", Msg: {result_payload['message'][:50]}..."
                tool_result_summary += archive_info

                memory_query += f"\nLast result summary: {tool_result_summary}"

            log.info(f"{self.log_prefix} Retrieving memories for thinking step...")
            relevant_memories, separated_suggestion = (
                self.memory.retrieve_and_rerank_memories(
                    query=memory_query,
                    task_description=f"Working on Task: {task.description}",
                    context=f"Cumulative Findings:\n{cumulative_findings[-500:]}\nIntended Plan:\n{plan_context[:500]}",
                    identity_statement=self.identity_statement,
                    n_results=config.MEMORY_COUNT_GENERAL_THINKING * 2,
                    n_final=config.MEMORY_COUNT_GENERAL_THINKING,
                )
            )

            user_provided_info_content = []
            other_memories_context_list = []
            for mem in relevant_memories:
                meta = mem.get("metadata", {})
                mem_type = meta.get("type", "N/A")
                relative_time = format_relative_time(meta.get("timestamp"))
                dist_str = (
                    f"{mem.get('distance', -1.0):.3f}"
                    if mem.get("distance") is not None
                    else "N/A"
                )
                mem_content = f"[Memory - {relative_time}] (Type: {mem_type}, Dist: {dist_str}, ID: {mem.get('id', 'N/A')[:8]}...):\n{mem['content']}"

                # Separate user-provided info specific to this task
                mem_task_context = meta.get("task_id_context")
                if (
                    mem_type == "user_provided_info_for_task"
                    and mem_task_context == task.id
                ):
                    user_provided_info_content.append(mem_content)
                else:
                    other_memories_context_list.append(mem_content)

            if other_memories_context_list:
                memory_context_str = "\n\n".join(other_memories_context_list)
            if user_provided_info_content:
                user_provided_info_str = "\n---\n".join(user_provided_info_content)

        except Exception as mem_err:
            log.error(
                f"{self.log_prefix} Error retrieving memories for thinking: {mem_err}",
                exc_info=True,
            )
            memory_context_str = "Error retrieving memories."
        # --- End Memory Retrieval ---

        prompt_vars = {
            "identity_statement": self.identity_statement,
            "task_description": task.description,
            "plan_context": plan_context,
            "cumulative_findings": cumulative_findings,
            "tool_desc": tool_desc,
            "memory_context_str": memory_context_str,
        }
        prompt_text = prompts.GENERATE_THINKING_PROMPT_BASE_V2.format(**prompt_vars)

        if user_suggested_move_on:
            prompt_text += (
                f"\n\n**USER SUGGESTION PENDING:** Consider wrapping up this task soon."
            )
        if user_provided_info_str:
            prompt_text += f"\n\n**User Provided Information (Consider for next action):**\n{user_provided_info_str}\n"

        # --- Format last tool results (if any) ---
        if tool_results:
            tool_name = tool_results.get("tool_name", "Unknown")
            tool_action = tool_results.get("action", "unknown")
            tool_status = tool_results.get("status", "unknown")
            tool_error = tool_results.get("error")
            result_payload = tool_results.get("result", {})
            archive_path_info = ""
            if isinstance(result_payload, dict):
                archive_path = result_payload.get("archived_filepath")
                if archive_path:
                    archive_path_info = (
                        f"\nNote: Previous version archived to '{archive_path}'"
                    )

            result_context = (
                f"(Error formatting result for tool: {tool_name}/{tool_action})"
            )
            try:
                # Format based on tool and action
                if tool_status == "failed":
                    result_context = f"Tool Error: {tool_error or 'Unknown failure'}"
                elif tool_name == "web":
                    if tool_action == "search":
                        results_list = result_payload.get("results", [])
                        if results_list:
                            formatted_list = []
                            for i, res in enumerate(results_list):
                                title = res.get("title", "No Title")
                                snippet = res.get("snippet", "...")
                                url = res.get("url", "N/A")
                                formatted_list.append(
                                    f"{i+1}. {title}\n   URL: {url}\n   Snippet: {snippet}"
                                )
                            result_context = "\n".join(formatted_list)
                        else:
                            result_context = (
                                result_payload.get("message") or "No search results."
                            )
                    elif tool_action == "browse":
                        if result_payload.get("query_mode"):
                            snippets = result_payload.get("retrieved_snippets", [])
                            query_used = result_payload.get("query", "N/A")
                            url_browsed = result_payload.get("url", "N/A")
                            if snippets:
                                formatted_list = [
                                    f"Snippet {i+1} (Index: {s.get('chunk_index', 'N/A')}, Dist: {s.get('distance', 'N/A')}) for query '{query_used}':\n{s.get('content', 'N/A')}"
                                    for i, s in enumerate(snippets)
                                ]
                                result_context = (
                                    f"Focused Browse on {url_browsed}:\n"
                                    + "\n---\n".join(formatted_list)
                                )
                            else:
                                result_context = f"Focused Browse on {url_browsed}: No relevant snippets found for query '{query_used}'."
                        else:
                            url_browsed = result_payload.get("url", "N/A")
                            content = result_payload.get("content")
                            # source = result_payload.get("content_source", "internet")
                            truncated = result_payload.get("truncated", False)
                            result_context = f"Browse on {url_browsed}:\n"
                            result_context += (
                                content if content else "(No content extracted)"
                            )
                            if truncated:
                                result_context += "...\n"  # use ... to indicate that content was truncated
                elif tool_name == "memory":
                    if tool_action == "search":
                        memories_list = result_payload.get("retrieved_memories", [])
                        if memories_list:
                            formatted_list = []
                            for mem in memories_list:
                                rank = mem.get("rank", "-")
                                rel_time = mem.get("relative_time", "N/A")
                                mem_type = mem.get("type", "N/A")
                                dist = mem.get("distance", "N/A")
                                snippet = mem.get("content_snippet", "...")
                                formatted_list.append(
                                    f"Rank {rank} ({rel_time}) - Type: {mem_type}, Dist: {dist}\n   Snippet: {snippet}"
                                )
                            result_context = "\n".join(formatted_list)
                        else:
                            result_context = (
                                result_payload.get("message") or "No memories found."
                            )
                    elif tool_action == "write":
                        result_context = result_payload.get(
                            "message", "Memory write completed."
                        )
                elif tool_name == "file":
                    if tool_action == "list":
                        dir_path = result_payload.get("directory_path", ".")
                        files = result_payload.get("files", [])
                        dirs = result_payload.get("directories", [])
                        result_context = f"Directory Listing for '{dir_path}':\nFiles ({len(files)}): {files}\nDirectories ({len(dirs)}): {dirs}"
                    elif tool_action == "read":
                        file_path = result_payload.get("filepath", "N/A")
                        content = result_payload.get("content")
                        truncated = result_payload.get("truncated", False)
                        result_context = f"Read file '{file_path}':\n"
                        result_context += (
                            content if content else "(No content or file empty)"
                        )
                        if truncated:
                            result_context += "...\n"  # use ... to indicate that content was truncated
                    elif tool_action == "write":
                        file_path = result_payload.get("filepath", "N/A")
                        archived = result_payload.get("archived_filepath")
                        result_context = f"Wrote file '{file_path}'." + (
                            f" Archived previous version." if archived else ""
                        )
                elif tool_name == "status":
                    result_context = result_payload.get(
                        "report_content", "(Status report content missing)"
                    )
                else:  # Fallback for unknown tools or other results
                    if isinstance(result_payload, dict) and result_payload.get(
                        "message"
                    ):
                        # Use message if available and specific formatting didn't catch it
                        result_context = result_payload["message"]
                    elif isinstance(result_payload, dict):
                        # Dump other dictionaries nicely
                        result_context = json.dumps(
                            result_payload, indent=2, ensure_ascii=False
                        )
                    else:
                        result_context = str(result_payload)

                # Apply truncation to the formatted result
                max_len = config.MAX_CHARACTERS_TOOL_RESULTS  # Use config limit
                if len(result_context) > max_len:
                    result_context = (
                        result_context[:max_len] + "..."
                    )  # use ... to indicate that content was truncated

                prompt_text += f"\n**Results from Last Action:**\nTool: {tool_name} (Action: {tool_action})\nStatus: {tool_status}\nResult Summary:\n```text\n{result_context}\n```{archive_path_info}\n"

            except Exception as fmt_err:
                log.warning(
                    f"{self.log_prefix} Failed to format detailed tool result for prompt: {fmt_err}",
                    exc_info=True,
                )
                # Fallback to basic status/error info if detailed formatting fails
                error_info = f", Error: {tool_error}" if tool_error else ""
                prompt_text += f"\n**Results from Last Action:**\nTool: {tool_name} (Action: {tool_action})\nStatus: {tool_status}{error_info}\nResult: (Error formatting details)\n"
        else:
            prompt_text += "\n**Results from Last Action:**\nNone.\n"
        # --- End Tool Result Formatting ---

        prompt_text += prompts.GENERATE_THINKING_TASK_NOW_PROMPT_V2.format(
            task_reattempt_count=task.reattempt_count + 1
        )

        log.info(
            f"{self.log_prefix} Asking {self.ollama_chat_model} for next action (Task {task.id[:8]}, Attempt {task.reattempt_count + 1})..."
        )
        llm_response_text = call_ollama_api(
            prompt_text, self.ollama_chat_model, self.ollama_base_url, timeout=180
        )

        if llm_response_text is None:
            log.error(
                f"{self.log_prefix} Failed to get thinking response from Ollama for task {task.id[:8]}."
            )
            return "LLM communication failed.", {
                "type": "error",
                "message": "LLM communication failure (thinking).",
                "subtype": "llm_comm_error",
            }

        # --- Parsing Logic ---
        try:
            action: Dict[str, Any] = {"type": "unknown"}
            raw_thinking = llm_response_text  # Default thinking is full response

            # Find markers robustly
            thinking_marker = "THINKING:"
            action_marker = "NEXT_ACTION:"
            tool_marker = "TOOL:"
            params_marker = "PARAMETERS:"
            answer_marker = "ANSWER:"
            reflections_marker = "REFLECTIONS:"

            # Use findall with MULTILINE and IGNORECASE for robustness
            thinking_match = re.search(
                rf"^{thinking_marker}(.*?)(?=^{action_marker}|^TOOL:|^PARAMETERS:|^ANSWER:|^REFLECTIONS:|\Z)",
                llm_response_text,
                re.MULTILINE | re.IGNORECASE | re.DOTALL,
            )
            action_match = re.search(
                rf"^{action_marker}(.*?)(?=^{tool_marker}|^PARAMETERS:|^ANSWER:|^REFLECTIONS:|\Z)",
                llm_response_text,
                re.MULTILINE | re.IGNORECASE | re.DOTALL,
            )

            if thinking_match:
                raw_thinking = thinking_match.group(1).strip()
            else:
                log.warning(
                    f"{self.log_prefix} 'THINKING:' marker not found or incorrectly placed. Raw response used as thinking."
                )
                # Keep raw_thinking as full response

            if not action_match:
                log.error(
                    f"{self.log_prefix} 'NEXT_ACTION:' marker not found or incorrectly placed."
                )
                # Try to guess action based on keywords if marker missing? Risky.
                # Let's return error for now.
                return raw_thinking, {
                    "type": "error",
                    "message": "Missing or misplaced NEXT_ACTION marker.",
                    "subtype": "parse_error",
                }

            action_type_str = action_match.group(1).strip().lower()

            # Tool Action Parsing
            if "use_tool" in action_type_str:
                action["type"] = "use_tool"
                tool_match = re.search(
                    rf"^{tool_marker}(.*?)(?=^{params_marker}|^ANSWER:|^REFLECTIONS:|\Z)",
                    llm_response_text,
                    re.MULTILINE | re.IGNORECASE | re.DOTALL,
                )
                params_match = re.search(
                    rf"^{params_marker}(.*?)(?=^{answer_marker}|^REFLECTIONS:|\Z)",
                    llm_response_text,
                    re.MULTILINE | re.IGNORECASE | re.DOTALL,
                )

                tool_name = None
                params_json = None

                if tool_match:
                    tool_name = tool_match.group(1).strip().strip("\"'")
                    log.debug(f"{self.log_prefix} Extracted tool name: '{tool_name}'")
                else:
                    # Allow status tool without explicit TOOL marker if PARAMS is also missing/empty
                    if "status" in llm_response_text.lower() and not params_match:
                        tool_name = "status"
                        log.debug(
                            f"{self.log_prefix} Inferred tool name 'status' due to lack of other markers."
                        )
                    else:
                        return raw_thinking, {
                            "type": "error",
                            "message": "Missing or misplaced TOOL marker for use_tool action.",
                            "subtype": "parse_error",
                        }

                if params_match:
                    raw_params = params_match.group(1).strip()
                    # Clean up potential markdown code blocks
                    raw_params = re.sub(
                        r"^```json\s*",
                        "",
                        raw_params,
                        flags=re.MULTILINE | re.IGNORECASE,
                    )
                    raw_params = re.sub(
                        r"\s*```$", "", raw_params, flags=re.MULTILINE | re.IGNORECASE
                    ).strip()

                    log.debug(f"{self.log_prefix} Raw PARAMS string: '{raw_params}'")
                    if not raw_params and tool_name == "status":
                        params_json = {}  # Status tool takes no params
                    elif raw_params:
                        try:
                            params_json = json.loads(raw_params)
                        except json.JSONDecodeError as e1:
                            log.warning(
                                f"{self.log_prefix} Direct JSON parse failed: {e1}. Trying fixes..."
                            )
                            # Apply common fixes (quotes, trailing commas)
                            fixed_str = (
                                raw_params.replace("“", '"')
                                .replace("”", '"')
                                .replace("‘", "'")
                                .replace("’", "'")
                            )
                            fixed_str = re.sub(
                                r",\s*([\}\]])", r"\1", fixed_str
                            )  # Remove trailing commas
                            fixed_str = re.sub(
                                r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", fixed_str
                            )  # Fix bad escapes cautiously

                            try:
                                params_json = json.loads(fixed_str)
                            except json.JSONDecodeError as e2:
                                log.error(
                                    f"{self.log_prefix} JSON parse failed after fixes: {e2}. Raw fixed: '{fixed_str}'"
                                )
                                # Attempt to extract inner JSON object as last resort
                                brace_match = re.search(
                                    r"\{.*\}", raw_params, re.DOTALL
                                )
                                if brace_match:
                                    extracted_str = brace_match.group(0)
                                    log.warning(
                                        f"{self.log_prefix} Trying brace extraction: '{extracted_str}'"
                                    )
                                    try:
                                        fixed_extracted_str = (
                                            extracted_str.replace("“", '"')
                                            .replace("”", '"')
                                            .replace("‘", "'")
                                            .replace("’", "'")
                                        )
                                        fixed_extracted_str = re.sub(
                                            r",\s*([\}\]])", r"\1", fixed_extracted_str
                                        )
                                        fixed_extracted_str = re.sub(
                                            r'(?<!\\)\\(?!["\\/bfnrtu])',
                                            r"\\\\",
                                            fixed_extracted_str,
                                        )
                                        params_json = json.loads(fixed_extracted_str)
                                    except json.JSONDecodeError as e3:
                                        err_msg = f"Invalid JSON in PARAMETERS after all attempts: {e3}. Original: '{raw_params}'"
                                        log.error(f"{self.log_prefix} {err_msg}")
                                        return raw_thinking, {
                                            "type": "error",
                                            "message": err_msg,
                                            "subtype": "parse_error",
                                        }
                                else:
                                    err_msg = f"Invalid JSON in PARAMETERS, no clear object found. Parse error: {e2}. Original: '{raw_params}'"
                                    log.error(f"{self.log_prefix} {err_msg}")
                                    return raw_thinking, {
                                        "type": "error",
                                        "message": err_msg,
                                        "subtype": "parse_error",
                                    }
                    else:  # Empty params block but not status tool
                        return raw_thinking, {
                            "type": "error",
                            "message": "Empty PARAMETERS block for non-status tool.",
                            "subtype": "parse_error",
                        }

                elif tool_name != "status":  # Params marker missing for non-status tool
                    return raw_thinking, {
                        "type": "error",
                        "message": "Missing PARAMETERS marker for use_tool action.",
                        "subtype": "parse_error",
                    }
                else:  # Status tool, no params needed
                    params_json = {}

                # --- Parameter Validation ---
                if (
                    action.get("type") != "error" and params_json is not None
                ):  # Check params_json is not None
                    if not tool_name:
                        action = {
                            "type": "error",
                            "message": "Tool name missing.",
                            "subtype": "parse_error",
                        }
                    elif tool_name not in self.tools:
                        action = {
                            "type": "error",
                            "message": f"Tool '{tool_name}' not available.",
                            "subtype": "invalid_tool",
                        }
                    elif not isinstance(params_json, dict):
                        action = {
                            "type": "error",
                            "message": f"Parsed PARAMETERS not a JSON object.",
                            "subtype": "parse_error",
                        }
                    else:
                        # Specific tool parameter validation (unchanged logic)
                        tool_action = params_json.get(
                            "action"
                        )  # Action within the tool
                        valid_params = True
                        err_msg = ""
                        action_subtype = (
                            "invalid_params"  # Default subtype for param errors
                        )

                        if tool_name == "web":
                            if not tool_action or not isinstance(tool_action, str):
                                err_msg = "Missing/invalid 'action' for web tool."
                                action_subtype = "missing_action"
                            elif tool_action == "search" and (
                                not params_json.get("query")
                                or not isinstance(params_json.get("query"), str)
                            ):
                                err_msg = (
                                    "Missing/invalid 'query' (string) for web search."
                                )
                            elif tool_action == "browse":
                                if not params_json.get("url") or not isinstance(
                                    params_json.get("url"), str
                                ):
                                    err_msg = (
                                        "Missing/invalid 'url' (string) for web browse."
                                    )
                                elif "query" in params_json and (
                                    not isinstance(params_json.get("query"), str)
                                    or not params_json["query"].strip()
                                ):
                                    err_msg = "Invalid 'query' for web browse (must be non-empty string if provided)."
                            elif tool_action not in ["search", "browse"]:
                                err_msg = (
                                    f"Invalid action '{tool_action}' for web tool."
                                )
                                action_subtype = "invalid_action"

                        elif tool_name == "memory":
                            if not tool_action or not isinstance(tool_action, str):
                                err_msg = "Missing/invalid 'action' for memory tool."
                                action_subtype = "missing_action"
                            elif tool_action == "search" and (
                                not params_json.get("query")
                                or not isinstance(params_json.get("query"), str)
                            ):
                                err_msg = "Missing/invalid 'query' (string) for memory search."
                            elif tool_action == "write" and (
                                not params_json.get("content")
                                or not isinstance(params_json.get("content"), str)
                            ):
                                err_msg = "Missing/invalid 'content' (string) for memory write."
                            elif tool_action not in ["search", "write"]:
                                err_msg = (
                                    f"Invalid action '{tool_action}' for memory tool."
                                )
                                action_subtype = "invalid_action"

                        elif tool_name == "file":
                            if not tool_action or not isinstance(tool_action, str):
                                err_msg = "Missing/invalid 'action' for file tool."
                                action_subtype = "missing_action"
                            elif tool_action == "read" and (
                                not params_json.get("filename")
                                or not isinstance(params_json.get("filename"), str)
                            ):
                                err_msg = (
                                    "Missing/invalid 'filename' (string) for file read."
                                )
                            elif tool_action == "write" and (
                                not params_json.get("filename")
                                or not isinstance(params_json.get("filename"), str)
                                or "content" not in params_json
                                or not isinstance(
                                    params_json.get("content"), str
                                )  # Allow empty string content
                            ):
                                err_msg = "Missing/invalid 'filename' (string) or 'content' (string) for file write."
                            elif tool_action == "list" and (
                                "directory" in params_json
                                and not isinstance(params_json.get("directory"), str)
                            ):
                                err_msg = "Invalid 'directory' for file list (must be string if provided)."
                            elif tool_action not in ["read", "write", "list"]:
                                err_msg = (
                                    f"Invalid action '{tool_action}' for file tool."
                                )
                                action_subtype = "invalid_action"

                        elif tool_name == "status":
                            if params_json:  # Should be empty {}
                                err_msg = "Status tool does not accept parameters."
                                action_subtype = "invalid_params"  # Correct subtype

                        else:  # Tool exists but no validation implemented
                            err_msg = (
                                f"Validation not implemented for tool '{tool_name}'."
                            )
                            action_subtype = "internal_error"

                        if err_msg:
                            log.error(
                                f"{self.log_prefix} Parameter Validation Failed: {err_msg} (Tool: {tool_name}, Action: {tool_action}, Params: {params_json})"
                            )
                            action = {
                                "type": "error",
                                "message": err_msg,
                                "subtype": action_subtype,
                            }
                            valid_params = False

                        if valid_params:
                            action["tool"] = tool_name
                            action["parameters"] = params_json
                            log.info(
                                f"{self.log_prefix} Parsed action: Use Tool '{tool_name}'"
                                + (f", Action '{tool_action}'" if tool_action else "")
                            )

            # Final Answer Parsing
            elif "final_answer" in action_type_str:
                action["type"] = "final_answer"
                answer = ""
                reflections = ""
                # Find ANSWER robustly
                answer_match = re.search(
                    rf"^{answer_marker}(.*?)(?=^{reflections_marker}|\Z)",
                    llm_response_text,
                    re.MULTILINE | re.IGNORECASE | re.DOTALL,
                )
                reflections_match = re.search(
                    rf"^{reflections_marker}(.*)",
                    llm_response_text,
                    re.MULTILINE | re.IGNORECASE | re.DOTALL,
                )

                if answer_match:
                    answer = answer_match.group(1).strip()
                else:
                    # Fallback: Take content after NEXT_ACTION if ANSWER missing
                    content_after_action = llm_response_text[
                        action_match.end() :
                    ].strip()
                    # Remove reflections if they exist
                    if reflections_match:
                        content_after_action = content_after_action[
                            : content_after_action.find(reflections_match.group(0))
                        ].strip()
                    if content_after_action:
                        log.warning(
                            f"{self.log_prefix} ANSWER marker missing. Using content after NEXT_ACTION as answer."
                        )
                        answer = content_after_action
                    else:
                        log.error(
                            f"{self.log_prefix} LLM chose final_answer but provided no ANSWER marker or subsequent content."
                        )
                        return raw_thinking, {
                            "type": "error",
                            "message": "Missing ANSWER content for final_answer.",
                            "subtype": "parse_error",
                        }

                if reflections_match:
                    reflections = reflections_match.group(1).strip()

                action["answer"] = answer
                action["reflections"] = reflections

                if not answer:  # Double check after potential fallback
                    return raw_thinking, {
                        "type": "error",
                        "message": "Missing ANSWER content for final_answer.",
                        "subtype": "parse_error",
                    }
                elif (
                    len(answer) < 50
                    and re.search(
                        r"\b(error|fail|cannot|unable|issue)\b", answer, re.IGNORECASE
                    )
                    and not user_suggested_move_on
                ):
                    log.warning(
                        f"{self.log_prefix} LLM 'final_answer' looks like giving up prematurely: '{answer[:100]}...'"
                    )
                    log.info(
                        f"{self.log_prefix} Parsed action: Final Answer (appears weak)."
                    )
                else:
                    log.info(f"{self.log_prefix} Parsed action: Final Answer.")

            # Invalid Action Type
            elif action_type_str:  # If action type was found but not recognized
                action = {
                    "type": "error",
                    "message": f"Invalid NEXT_ACTION specified: '{action_type_str}'. Expected 'use_tool' or 'final_answer'.",
                    "subtype": "parse_error",
                }

            # Check if action remained 'unknown' without a specific parse error
            if action["type"] == "unknown":
                log.error(
                    f"{self.log_prefix} Could not parse a valid action from LLM response: {llm_response_text}"
                )
                action = {
                    "type": "error",
                    "message": "Could not parse action from LLM response.",
                    "subtype": "parse_error",
                }

            return raw_thinking, action

        except Exception as e:
            log.exception(
                f"{self.log_prefix} CRITICAL failure parsing LLM thinking response: {e}\nResponse:\n{llm_response_text}"
            )
            # Return raw thinking (if any) and an internal error
            return raw_thinking or "Error during parsing.", {
                "type": "error",
                "message": f"Internal error parsing LLM response: {e}",
                "subtype": "internal_error",
            }

    # execute_tool remains the same logic, logs agent ID
    def execute_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        if tool_name not in self.tools:
            log.error(
                f"{self.log_prefix} Attempted to execute unknown tool '{tool_name}'"
            )
            return {"error": f"Tool '{tool_name}' not found", "status": "failed"}

        tool = self.tools[tool_name]
        # Determine tool's action from parameters (handle status tool case)
        tool_action = parameters.get("action") if isinstance(parameters, dict) else None
        if tool_name == "status":
            tool_action = "status_report"  # Implicit action for status tool

        log_action_display = tool_action if tool_action else "(implicit)"
        log.info(
            f"{self.log_prefix} Executing tool '{tool_name}', action '{log_action_display}' with params: {parameters}"
        )

        try:
            result: Dict[str, Any] = {}
            start_time = time.time()

            # Execute based on tool name
            if tool_name == "memory":
                # Memory tool needs memory instance and identity
                result = tool.run(
                    parameters,
                    memory_instance=self.memory,
                    identity_statement=self.identity_statement,
                )
            elif tool_name == "status":
                # Status tool needs various pieces of agent state
                ollama_ok, ollama_msg = check_ollama_status()
                searxng_ok, searxng_msg = check_searxng_status()
                ollama_status_str = f"OK" if ollama_ok else f"FAIL ({ollama_msg})"
                searxng_status_str = f"OK" if searxng_ok else f"FAIL ({searxng_msg})"
                task_summary = self.task_queue.get_all_tasks_structured()
                memory_summary = self.memory.get_memory_summary()
                result = tool.run(
                    agent_identity=self.identity_statement,
                    task_queue_summary=task_summary,
                    memory_summary=memory_summary,
                    ollama_status=ollama_status_str,
                    searxng_status=searxng_status_str,
                )
            elif tool_name in ["web", "file"]:
                # Web and File tools only need parameters (paths handled internally)
                result = tool.run(parameters)
            else:
                # Fallback for potentially unknown but loaded tools
                log.warning(
                    f"{self.log_prefix} Executing unknown tool type '{tool_name}' via standard run."
                )
                result = tool.run(parameters)

            duration = time.time() - start_time
            log.info(
                f"{self.log_prefix} Tool '{tool_name}' action '{log_action_display}' finished in {duration:.2f}s."
            )

            # --- State updates after tool execution ---
            # Update last_web_browse_content based on result
            if (
                tool_name == "web"
                and tool_action == "browse"
                and isinstance(result, dict)
            ):
                # Use lock for updating shared session state
                with self._state_lock:  # RLock
                    if result.get("query_mode") is False and "content" in result:
                        self.session_state["last_web_browse_content"] = result.get(
                            "content", "(Empty)"
                        )
                    elif result.get("query_mode") is True:
                        snippets = result.get("retrieved_snippets", [])
                        snippet_count = len(snippets)
                        query_used = parameters.get("query", "N/A")
                        self.session_state["last_web_browse_content"] = (
                            f"(Retrieved {snippet_count} snippets from archived page for query: '{query_used}')"
                        )
                    else:
                        # Handle cases where browse might fail or return unexpected format
                        self.session_state["last_web_browse_content"] = result.get(
                            "message", result.get("error", "(Browse result unclear)")
                        )
            # --- End state updates ---

            # --- Result Processing & Robustness Checks ---
            if not isinstance(result, dict):
                log.warning(
                    f"{self.log_prefix} Result from '{tool_name}' ({log_action_display}) not a dict: {type(result)}. Wrapping."
                )
                return {
                    "tool_name": tool_name,
                    "action": tool_action,
                    "result": {"unexpected_result": str(result)},
                    "status": "completed_malformed_output",
                }

            if result.get("error"):
                log.warning(
                    f"{self.log_prefix} Tool '{tool_name}' ({log_action_display}) reported error: {result['error']}"
                )
                # Return structured error
                return {
                    "tool_name": tool_name,
                    "action": tool_action,
                    "error": result["error"],
                    "status": "failed",  # Explicitly mark as failed
                    "result": result,  # Include original result dict for context
                }

            # Check JSON serializability before returning
            try:
                json.dumps(result)
            except TypeError as json_err:
                log.warning(
                    f"{self.log_prefix} Result from '{tool_name}' ({log_action_display}) not JSON serializable: {json_err}. Sanitizing."
                )
                serializable_result = {}
                for k, v in result.items():
                    try:
                        json.dumps({k: v})  # Test individual item
                        serializable_result[k] = v
                    except TypeError:
                        serializable_result[k] = (
                            f"<Unserializable type: {type(v).__name__}> {str(v)[:100]}..."
                        )
                return {
                    "tool_name": tool_name,
                    "action": tool_action,
                    "result": serializable_result,
                    "status": "completed_unserializable",  # Specific status
                }

            # Prepare final clean return structure
            final_status = result.get(
                "status", "completed"
            )  # Default to completed if status key missing
            # Copy result payload, removing redundant keys
            result_payload = result.copy()
            result_payload.pop("status", None)
            result_payload.pop("error", None)
            # Keep 'action' in payload if returned by tool (e.g., status_report)
            # result_payload.pop("action", None) # Keep action if present

            return {
                "tool_name": tool_name,
                "action": tool_action,  # Use the action determined at the start
                "status": final_status,
                "result": result_payload,
            }

        except Exception as e:
            log.exception(
                f"{self.log_prefix} CRITICAL Error executing tool '{tool_name}' action '{log_action_display}': {e}"
            )
            return {
                "tool_name": tool_name,
                "action": tool_action,
                "error": f"Tool execution raised unexpected exception: {e}",
                "status": "failed",
            }

    # _save_qlora_datapoint uses agent-specific path
    def _save_qlora_datapoint(
        self, source_type: str, instruction: str, input_context: str, output: str
    ):
        if not output or not output.strip():
            log.debug(
                f"{self.log_prefix} Skipping saving QLoRA datapoint: Empty output."
            )
            return
        try:
            datapoint = {
                "instruction": instruction,
                "input": input_context,
                "output": output,
                "source_type": source_type,
                "agent_id": self.agent_id,  # Add agent ID
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            # Append as JSON Lines
            with open(self.qlora_dataset_path, "a", encoding="utf-8") as f:
                json.dump(datapoint, f, ensure_ascii=False)
                f.write("\n")
            # Reduce log noise, change to debug or remove if too verbose
            # log.info(f"{self.log_prefix} QLoRA datapoint saved (Source: {source_type}) to {self.qlora_dataset_path}")
        except Exception as e:
            log.exception(f"{self.log_prefix} Failed to save QLoRA datapoint: {e}")

    # _summarize_and_prune_task_memories remains the same logic, logs agent ID
    def _summarize_and_prune_task_memories(self, task: Task):
        if not config.ENABLE_MEMORY_SUMMARIZATION:
            log.debug(
                f"{self.log_prefix} Memory summarization disabled, skipping for task {task.id[:8]}."
            )
            return

        log.info(
            f"{self.log_prefix} Summarizing memories and findings for {task.status} task {task.id[:8]}..."
        )
        final_summary_text = "No cumulative findings recorded for this task."
        trunc_limit = (
            config.MAX_CHARACTERS_CUMULATIVE_FINDINGS
        )  # Use config for truncation

        if task.cumulative_findings and task.cumulative_findings.strip():
            findings_to_summarize = task.cumulative_findings
            if len(findings_to_summarize) > trunc_limit:
                log.warning(
                    f"{self.log_prefix} Cumulative findings for task {task.id[:8]} exceed limit ({len(findings_to_summarize)} > {trunc_limit}). Truncating for summary prompt."
                )
                # Truncate from the beginning to keep recent info
                findings_to_summarize = (
                    "...\n" + findings_to_summarize[-trunc_limit:]
                )  # use ... to indicate that content was truncated

            summary_prompt = prompts.SUMMARIZE_TASK_PROMPT.format(
                task_status=task.status,
                summary_context=findings_to_summarize,
            )
            log.info(
                f"{self.log_prefix} Asking {self.ollama_chat_model} to summarize task findings {task.id[:8]}..."
            )
            summary_text = call_ollama_api(
                summary_prompt,
                self.ollama_chat_model,
                self.ollama_base_url,
                timeout=150,
            )
            if summary_text and summary_text.strip():
                final_summary_text = f"Task Summary ({task.status}):\n{summary_text}"
                # Update task result/reflections based on summary if they are missing
                if task.status == "completed" and not task.result:
                    log.info(
                        f"{self.log_prefix} Setting task {task.id[:8]} result from final summary."
                    )
                    task.result = {
                        "answer": summary_text,
                        "source": "generated_summary",
                    }
                    self.task_queue.update_task(
                        task.id, status=task.status, result=task.result
                    )  # Only update result
                elif task.status == "failed" and not task.reflections:
                    log.info(
                        f"{self.log_prefix} Setting task {task.id[:8]} reflections from final summary."
                    )
                    task.reflections = f"Failure Summary (Generated): {summary_text}"
                    self.task_queue.update_task(
                        task.id, status=task.status, reflections=task.reflections
                    )  # Only update reflections
            else:
                log.warning(
                    f"{self.log_prefix} Failed to generate summary for task findings {task.id[:8]}. Using truncated raw findings for memory."
                )
                # Use truncated findings if summary fails
                final_summary_text = (
                    f"Task Findings ({task.status}):\n{findings_to_summarize}"
                )
        else:
            log.info(
                f"{self.log_prefix} No cumulative findings to summarize for task {task.id[:8]}."
            )

        # Add the summary (or findings) to memory
        try:
            task_memories = self.memory.get_memories_by_metadata(
                filter_dict={"task_id": task.id}
            )
            summary_metadata = {
                "type": "task_summary",
                "task_id": task.id,
                "original_status": task.status,
                "summarized_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "original_memory_count": len(
                    task_memories
                ),  # Store count before potential prune
            }
            summary_id = self.memory.add_memory(final_summary_text, summary_metadata)
        except Exception as e:
            log.error(
                f"{self.log_prefix} Error retrieving memories or adding summary for task {task.id[:8]}: {e}"
            )
            summary_id = None
            task_memories = []  # Ensure task_memories is defined

        if not summary_id:
            log.error(
                f"{self.log_prefix} Failed to add final summary memory for task {task.id[:8]}. Pruning skipped."
            )
            return  # Cannot prune if summary wasn't added

        log.info(
            f"{self.log_prefix} Summary added to memory (ID: {summary_id[:8]}). Checking for pruning..."
        )

        if not config.DELETE_MEMORIES_AFTER_SUMMARY:
            log.info(
                f"{self.log_prefix} Pruning disabled via config. Keeping original task memories."
            )
            return

        if not task_memories:
            log.info(
                f"{self.log_prefix} No specific action memories found for task {task.id[:8]} to potentially prune."
            )
            return

        log.debug(
            f"{self.log_prefix} Found {len(task_memories)} action-related memories for task {task.id[:8]}."
        )

        # Define types to *keep* associated with the task
        types_to_keep = {
            "task_summary",
            "identity_revision",  # Keep identity revisions even if somehow tagged with task_id
            "session_reflection",  # Keep session reflections
            "user_suggestion_move_on",
            "task_result",  # Keep the final answer memory
            "task_reflection",  # Keep the final reflection memory
            "lesson_learned",  # Keep lessons learned specifically
            "user_provided_info_for_task",  # Keep user input
            "agent_explicit_memory_write",  # Keep memories explicitly written by agent via tool
        }

        memory_ids_to_prune = []
        for mem in task_memories:
            meta = mem.get("metadata", {})
            mem_type = meta.get(
                "type", "memory"
            )  # Default to generic 'memory' if type missing
            mem_id = mem.get("id")

            # Ensure we don't prune the summary we just added
            if mem_id == summary_id:
                continue

            # Prune if the type is not in the keep list
            if mem_type not in types_to_keep:
                if mem_id:
                    memory_ids_to_prune.append(mem_id)
                else:
                    log.warning(
                        f"{self.log_prefix} Found memory with missing ID during pruning check."
                    )

        if memory_ids_to_prune:
            log.info(
                f"{self.log_prefix} Pruning {len(memory_ids_to_prune)} original action memories for task {task.id[:8]}..."
            )
            try:
                deleted = self.memory.delete_memories(memory_ids_to_prune)
                log.info(
                    f"{self.log_prefix} Pruning status for task {task.id[:8]}: {'Success' if deleted else 'Failed'}. Count: {len(memory_ids_to_prune)}"
                )
            except Exception as e:
                log.error(
                    f"{self.log_prefix} Error during memory deletion for task {task.id[:8]}: {e}"
                )
        else:
            log.info(
                f"{self.log_prefix} No original action memories found eligible for pruning for task {task.id[:8]} (kept types: {len(task_memories)})."
            )

    # _reflect_on_error_and_prepare_reattempt remains the same logic, logs agent ID
    def _reflect_on_error_and_prepare_reattempt(
        self,
        task: Task,
        error_message: str,
        error_subtype: str,
        failed_action_context: str,
    ) -> bool:
        log.warning(
            f"{self.log_prefix} Task {task.id[:8]} failed action. Attempting to learn and restart task (Next Attempt {task.reattempt_count + 1}/{config.TASK_MAX_REATTEMPT}). Error: {error_message} ({error_subtype})"
        )
        plan_steps_str = (
            "\n".join([f"- {s}" for s in task.plan]) if task.plan else "N/A"
        )
        prompt = prompts.LESSON_LEARNED_PROMPT_V2.format(
            task_description=task.description,
            plan_context=plan_steps_str,
            failed_action_context=failed_action_context,
            error_message=error_message,
            error_subtype=error_subtype,
            cumulative_findings=task.cumulative_findings[-2000:],  # Limit context
            identity_statement=self.identity_statement,
        )
        log.info(
            f"{self.log_prefix} Asking {self.ollama_chat_model} to generate lesson learned for task {task.id[:8]} error..."
        )
        lesson_learned_text = call_ollama_api(
            prompt, self.ollama_chat_model, self.ollama_base_url, timeout=120
        )
        if not lesson_learned_text or not lesson_learned_text.strip():
            log.warning(
                f"{self.log_prefix} LLM failed to generate a lesson learned. Proceeding with re-attempt anyway."
            )
            lesson_learned_text = (
                f"(LLM failed to generate lesson for error: {error_message})"
            )
        else:
            lesson_learned_text = lesson_learned_text.strip()
            log.info(
                f"{self.log_prefix} Generated Lesson Learned: {lesson_learned_text}"
            )
            # Add lesson to memory
            try:
                self.memory.add_memory(
                    content=f"Lesson Learned (Task '{task.id[:8]}', Attempt {task.reattempt_count+1} Failed Action): {lesson_learned_text}",
                    metadata={
                        "type": "lesson_learned",
                        "task_id": task.id,
                        "failed_action_context": failed_action_context[
                            :200
                        ],  # Limit length
                        "error_subtype": error_subtype,
                        "error_message": error_message[:200],  # Limit length
                    },
                )
            except Exception as e:
                log.error(f"{self.log_prefix} Failed to add lesson learned memory: {e}")

        # Prepare task for reattempt in task queue
        reset_success = self.task_queue.prepare_task_for_reattempt(
            task.id, lesson_learned_text
        )
        if reset_success:
            log.info(
                f"{self.log_prefix} Task {task.id[:8]} successfully prepared for re-attempt {task.reattempt_count + 1} (status: planning)."
            )
            return True
        else:
            log.error(
                f"{self.log_prefix} Failed to reset task {task.id[:8]} state in TaskQueue. Cannot re-attempt."
            )
            return False

    # _perform_action_cycle remains the same logic, logs agent ID
    def _perform_action_cycle(self, task: Task) -> Dict[str, Any]:
        cycle_start_time = time.time()
        action_log = []
        final_answer_text = None
        cycle_status = "processing"  # Possible outcomes: processing, completed, failed, error_retry, error_reattempting_task
        task_status_updated_this_cycle = False  # Track if task queue was updated
        current_retries = self.session_state.get("current_action_retries", 0)

        # Check and consume user suggestion flag within lock
        user_suggested_move_on = False
        with self._state_lock:  # RLock
            if self.session_state.get("user_suggestion_move_on_pending", False):
                user_suggested_move_on = True
                self.session_state["user_suggestion_move_on_pending"] = False
                self.save_session_state()  # Save immediately after consuming flag

        tool_results_for_ui = None  # Store raw tool output for UI
        thinking_to_store = "(No thinking process recorded for this cycle)"
        action_type = "internal"  # Default action type
        action_objective = f"Make progress on task: {task.description[:60]}..."
        cycle_num_display = task.current_step_index + 1  # Cycle number (1-based)

        # Action details for history (populate as cycle progresses)
        action_details_for_history = {
            "task_id": task.id,
            "action_cycle": cycle_num_display,
            "action_objective": action_objective,
            "task_attempt": task.reattempt_count + 1,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "thinking": thinking_to_store,
            "action_type": action_type,
            "action_params": None,
            "result_status": None,
            "result_summary": None,
            "log_snippet": None,
        }

        task.current_step_index += 1  # Increment step index *before* execution

        action_log.append(
            f"--- Task '{task.id[:8]}' | Action Cycle {cycle_num_display} (Retry {current_retries}/{config.AGENT_MAX_STEP_RETRIES}, Task Attempt {task.reattempt_count + 1}/{config.TASK_MAX_REATTEMPT}) ---"
        )
        action_log.append(f"Task Goal: {task.description}")
        if user_suggested_move_on:
            log.info(
                f"{self.log_prefix} User suggestion 'move on' is active for task {task.id[:8]}, cycle {cycle_num_display}."
            )
            action_log.append("[INFO] User suggestion to wrap up considered.")

        log.info(
            f"{self.log_prefix} Executing action cycle {cycle_num_display} for task {task.id[:8]} (Action Retry: {current_retries}, Task Attempt: {task.reattempt_count + 1})"
        )

        # --- Generate Thinking and Action ---
        # Use last cycle's tool results (stored in UI state) as input
        last_tool_results = self._ui_update_state.get("last_tool_results")
        raw_thinking, action = self.generate_thinking(
            task=task,
            tool_results=last_tool_results,
            user_suggested_move_on=user_suggested_move_on,
        )
        action_type = action.get("type", "error")
        action_message = action.get("message", "Unknown error from thinking step")
        action_subtype = action.get("subtype", "unknown_error")
        thinking_to_store = raw_thinking or "Thinking process not extracted."
        action_log.append(f"Thinking:\n{thinking_to_store}")
        action_details_for_history["thinking"] = thinking_to_store
        action_details_for_history["action_type"] = action_type
        action_details_for_history["action_objective"] = (
            f"Cycle {cycle_num_display} thinking towards: {task.description[:60]}..."
        )

        # Store thinking in memory
        try:
            self.memory.add_memory(
                f"Action Cycle {cycle_num_display} Thinking (Action: {action_type}, Subtype: {action_subtype}, Attempt {task.reattempt_count+1}):\n{thinking_to_store}",
                {
                    "type": "task_thinking",
                    "task_id": task.id,
                    "action_cycle": cycle_num_display,
                    "action_type": action_type,
                    "action_subtype": action_subtype,
                    "task_attempt": task.reattempt_count + 1,
                },
            )
        except Exception as e:
            log.error(f"{self.log_prefix} Failed to add thinking memory: {e}")

        cycle_finding = ""  # Accumulate findings text for this cycle

        # --- Handle Action ---
        if action_type == "error":
            # --- Handle LLM Action Error ---
            log.error(
                f"{self.log_prefix} [ACTION ERROR] LLM Action Error (Task {task.id[:8]}, Cycle {cycle_num_display}, Attempt {task.reattempt_count+1}): {action_message} (Subtype: {action_subtype})"
            )
            action_log.append(
                f"[ERROR] Action Error: {action_message} (Type: {action_subtype})"
            )
            cycle_finding = f"\n--- Action Cycle {cycle_num_display} Error (Action Retry {current_retries+1}, Task Attempt {task.reattempt_count+1}): LLM Action Error - {action_message} (Type: {action_subtype}) ---\n"
            task.cumulative_findings += cycle_finding

            # Add error memory
            try:
                self.memory.add_memory(
                    f"Action Cycle {cycle_num_display} Error: {action_message}",
                    {
                        "type": "agent_error",
                        "task_id": task.id,
                        "action_cycle": cycle_num_display,
                        "error_subtype": action_subtype,
                        "llm_response_snippet": thinking_to_store[
                            :200
                        ],  # Use parsed thinking
                        "task_attempt": task.reattempt_count + 1,
                    },
                )
            except Exception as e:
                log.error(f"{self.log_prefix} Failed to add error memory: {e}")

            action_details_for_history["result_status"] = "error"
            action_details_for_history["result_summary"] = (
                f"LLM Action Error: {action_message} ({action_subtype})"
            )
            failed_action_context = f"LLM failed to generate valid action during cycle {cycle_num_display}. Error: {action_message} ({action_subtype}). Thinking: {thinking_to_store[:100]}..."

            # Retry / Reattempt / Fail logic
            if current_retries < config.AGENT_MAX_STEP_RETRIES:
                with self._state_lock:  # RLock
                    self.session_state["current_action_retries"] = current_retries + 1
                log.warning(
                    f"{self.log_prefix} Incrementing action retry count to {current_retries + 1}."
                )
                cycle_status = "error_retry"
            else:  # Max action retries reached
                if task.reattempt_count < config.TASK_MAX_REATTEMPT:
                    if self._reflect_on_error_and_prepare_reattempt(
                        task, action_message, action_subtype, failed_action_context
                    ):
                        cycle_status = "error_reattempting_task"
                        action_log.append(
                            f"[INFO] Max action retries reached. Re-attempting task from beginning (Attempt {task.reattempt_count + 1})."
                        )
                        with self._state_lock:  # RLock
                            self.session_state["current_action_retries"] = 0
                    else:  # Failed to prepare for reattempt (critical task queue error)
                        log.error(
                            f"{self.log_prefix} Failed to prepare task {task.id[:8]} for re-attempt. Failing task permanently."
                        )
                        fail_reason = f"Failed during action cycle {cycle_num_display} Attempt {task.reattempt_count+1}. Max action retries ({config.AGENT_MAX_STEP_RETRIES}) + Failed to reset task state."
                        task.reflections = fail_reason
                        task.cumulative_findings += f"\n--- Task Failed: Max action retries & reset failure on cycle {cycle_num_display}. ---\n"
                        self.task_queue.update_task(
                            task.id, "failed", reflections=fail_reason
                        )
                        task.status = "failed"  # Update local task object status too
                        task_status_updated_this_cycle = True
                        cycle_status = "failed"
                else:  # Max task reattempts also reached
                    log.error(
                        f"{self.log_prefix} Max action retries AND max task reattempts reached for task {task.id[:8]}. Failing permanently."
                    )
                    fail_reason = f"Failed during action cycle {cycle_num_display}. Max action retries ({config.AGENT_MAX_STEP_RETRIES}) and max task reattempts ({config.TASK_MAX_REATTEMPT}) reached. Last error: {action_message}"
                    task.reflections = fail_reason
                    task.cumulative_findings += f"\n--- Task Failed Permanently: Max action and task retries reached on cycle {cycle_num_display}. ---\n"
                    self.task_queue.update_task(
                        task.id, "failed", reflections=fail_reason
                    )
                    task.status = "failed"  # Update local task object
                    task_status_updated_this_cycle = True
                    cycle_status = "failed"

        elif action_type == "use_tool":
            # --- Handle Tool Use ---
            tool_name = action.get("tool")
            params = action.get("parameters")
            # Determine display action name (handle status tool)
            tool_action_param = (
                params.get("action") if isinstance(params, dict) else None
            )
            log_action_display = tool_action_param or (
                "status_report" if tool_name == "status" else "(implicit)"
            )

            log.info(
                f"{self.log_prefix} [ACTION] Cycle {cycle_num_display}: Use Tool '{tool_name}', Action '{log_action_display}' (Attempt {task.reattempt_count+1})"
            )
            action_log.append(
                f"[ACTION] Use Tool: {tool_name}, Action: {log_action_display}, Params: {json.dumps(params, ensure_ascii=False, indent=2)}"
            )
            action_details_for_history["action_params"] = params

            # Execute Tool
            tool_output = self.execute_tool(tool_name, params)
            tool_results_for_ui = (
                tool_output  # Store raw output for next cycle's input via UI state
            )

            # Process Tool Output
            tool_status = tool_output.get("status", "unknown")
            tool_error = tool_output.get("error")
            tool_action_from_result = tool_output.get(
                "action", log_action_display
            )  # Use determined action if available in result
            result_content = tool_output.get("result", {})

            action_details_for_history["result_status"] = tool_status

            # --- Generate Result Summary for History and Findings ---
            result_display_str = "(Error summarizing result)"
            summary_limit = (
                config.MAX_CHARACTER_SUMMARY
            )  # Limit summary length in history/findings
            content_result_limit = (
                config.MAX_TOOL_CONTENT_RESULTS
            )  # Limit content results from tool use, should be less than summary_limit
            try:
                if isinstance(result_content, dict):
                    if "message" in result_content:
                        result_display_str = result_content["message"]
                    elif (
                        tool_name == "file"
                        and tool_action_from_result == "write"
                        and "filepath" in result_content
                    ):
                        archived_path = result_content.get("archived_filepath")
                        result_display_str = (
                            f"Wrote file: {result_content['filepath']}"
                            + (f" (Archived previous)" if archived_path else "")
                        )
                    elif (
                        tool_name == "file"
                        and tool_action_from_result == "list"
                        and "files" in result_content
                    ):
                        result_display_str = f"Listed dir '{result_content.get('directory_path', '?')}': {len(result_content.get('files',[]))} files, {len(result_content.get('directories',[]))} dirs."
                    elif (
                        tool_name == "web"
                        and tool_action_from_result == "search"
                        and "results" in result_content
                    ):
                        result_display_str = (
                            f"Found {len(result_content['results'])} search results."
                        )
                    elif (
                        tool_name == "web"
                        and tool_action_from_result == "browse"
                        and result_content.get("query_mode") is True
                    ):
                        result_display_str = f"Focused browse on '{result_content.get('url','?')}': Found {len(result_content.get('retrieved_snippets',[]))} snippets for query '{result_content.get('query','?')}'."
                    elif (
                        tool_name == "web"
                        and tool_action_from_result == "browse"
                        and "content" in result_content
                    ):
                        trunc_info = (
                            "..." if result_content.get("truncated") else ""
                        )  # use ... to indicate that content was truncated
                        result_display_str = f"Browsed '{result_content.get('url','?')}': Content ({result_content.get('content_source', '?')}) length {len(result_content['content'])} {trunc_info}"
                    elif (
                        tool_name == "memory"
                        and tool_action_from_result == "search"
                        and "retrieved_memories" in result_content
                    ):
                        result_display_str = f"Found {len(result_content['retrieved_memories'])} memories."
                    elif tool_name == "status" and "report_content" in result_content:
                        result_display_str = "Generated status report."
                    elif "content" in result_content:  # Generic content display
                        content_str = str(result_content["content"])
                        result_display_str = content_str[:content_result_limit] + (
                            "..." if len(content_str) > content_result_limit else ""
                        )
                    else:  # Fallback for other dict structures
                        result_display_str = json.dumps(
                            result_content, indent=None, ensure_ascii=False
                        )
                else:  # Non-dict result
                    result_display_str = str(result_content)

                # Apply length limit
                if (
                    len(result_display_str) > content_result_limit
                ):  # Generous limit for internal log/findings
                    result_display_str = (
                        result_display_str[:content_result_limit] + "..."
                    )

            except Exception as summ_e:
                log.warning(
                    f"{self.log_prefix} Error summarizing tool result: {summ_e}"
                )
                result_display_str = "(Error summarizing result)"

            # Create summary specifically for history (shorter limit)
            result_summary_for_history = result_display_str[:summary_limit] + (
                "..." if len(result_display_str) > summary_limit else ""
            )
            if tool_error:  # Prepend error to history summary if tool failed
                result_summary_for_history = (
                    f"Error: {tool_error}\n---\n{result_summary_for_history}"[
                        :summary_limit
                    ]
                    + ("..." if len(result_summary_for_history) > summary_limit else "")
                )
            action_details_for_history["result_summary"] = result_summary_for_history
            # --- End Result Summary ---

            # Add cycle finding
            cycle_finding = f"\n--- Action Cycle {cycle_num_display} (Tool: {tool_name}/{tool_action_from_result}, Attempt {task.reattempt_count+1}) ---\nStatus: {tool_status}\nResult Summary: {result_summary_for_history}\n---\n"
            task.cumulative_findings += cycle_finding

            # Add tool result memory
            try:
                mem_params_summary = json.dumps(params)[
                    : config.MAX_MEMORY_PARAMS_SNIPPETS
                ] + (
                    "..."
                    if len(json.dumps(params)) > config.MAX_MEMORY_PARAMS_SNIPPETS
                    else ""
                )
                self.memory.add_memory(
                    f"Action Cycle {cycle_num_display} Tool Result (Status: {tool_status}, Attempt {task.reattempt_count+1}):\nAction: {tool_name}/{tool_action_from_result}\nParams: {mem_params_summary}\nSummary: {result_summary_for_history}",
                    {
                        "type": "tool_result",
                        "task_id": task.id,
                        "action_cycle": cycle_num_display,
                        "tool_name": tool_name,
                        "action": tool_action_from_result,
                        "result_status": tool_status,
                        "params_snippet": mem_params_summary,
                        "result_summary_snippet": result_summary_for_history[
                            : config.MAX_MEMORY_SUMMARY_SNIPPETS
                        ],  # Add snippet to metadata
                        "task_attempt": task.reattempt_count + 1,
                    },
                )
            except Exception as e:
                log.error(f"{self.log_prefix} Failed to add tool result memory: {e}")

            # Handle Tool Result Status (Retry/Reattempt/Fail)
            if tool_status in [
                "success",
                "completed",
                "completed_malformed_output",
                "completed_unserializable",
                "completed_with_issue",
                "success_archived",
            ]:
                if tool_status not in ["success", "completed", "success_archived"]:
                    log.warning(
                        f"{self.log_prefix} Tool '{tool_name}' action '{tool_action_from_result}' finished cycle {cycle_num_display} with non-ideal status: {tool_status}."
                    )
                else:
                    action_log.append(
                        f"[INFO] Tool '{tool_name}' action '{tool_action_from_result}' completed cycle {cycle_num_display} successfully."
                    )
                cycle_status = "processing"  # Continue processing
                with self._state_lock:  # RLock
                    self.session_state["current_action_retries"] = (
                        0  # Reset retries on success
                    )

            elif tool_status == "failed":
                log.warning(
                    f"{self.log_prefix} Tool '{tool_name}' action '{tool_action_from_result}' failed cycle {cycle_num_display}: {tool_error}"
                )
                action_log.append(
                    f"[ERROR] Tool '{tool_name}' ({tool_action_from_result}) failed: {tool_error}"
                )

                # Add specific error finding
                error_finding = f"\n--- Action Cycle {cycle_num_display} Error (Action Retry {current_retries+1}, Task Attempt {task.reattempt_count+1}): Tool Error - Tool={tool_name}, Action={tool_action_from_result}, Error={tool_error} ---\n"
                task.cumulative_findings += error_finding

                # Add error memory
                try:
                    self.memory.add_memory(
                        f"Action Cycle {cycle_num_display} Tool Error: {tool_error}",
                        {
                            "type": "agent_error",
                            "task_id": task.id,
                            "action_cycle": cycle_num_display,
                            "error_subtype": "tool_execution_error",
                            "tool_name": tool_name,
                            "action": tool_action_from_result,
                            "error_details": str(tool_error)[:200],
                            "task_attempt": task.reattempt_count + 1,
                        },
                    )
                except Exception as e:
                    log.error(f"{self.log_prefix} Failed to add tool error memory: {e}")

                failed_action_context = f"Tool '{tool_name}' (Action: {tool_action_from_result}) failed during cycle {cycle_num_display}. Error: {tool_error}."

                # Retry / Reattempt / Fail logic (same as LLM action error)
                if current_retries < config.AGENT_MAX_STEP_RETRIES:
                    with self._state_lock:  # RLock
                        self.session_state["current_action_retries"] = (
                            current_retries + 1
                        )
                    log.warning(
                        f"{self.log_prefix} Incrementing action retry count to {current_retries + 1}."
                    )
                    cycle_status = "error_retry"
                else:  # Max action retries reached
                    if task.reattempt_count < config.TASK_MAX_REATTEMPT:
                        if self._reflect_on_error_and_prepare_reattempt(
                            task,
                            tool_error or "Tool Failed",
                            "tool_execution_error",
                            failed_action_context,
                        ):
                            cycle_status = "error_reattempting_task"
                            action_log.append(
                                f"[INFO] Max action retries reached. Re-attempting task from beginning (Attempt {task.reattempt_count + 1})."
                            )
                            with self._state_lock:  # RLock
                                self.session_state["current_action_retries"] = 0
                        else:  # Failed to prepare for reattempt
                            log.error(
                                f"{self.log_prefix} Failed to prepare task {task.id[:8]} for re-attempt after tool failure. Failing task permanently."
                            )
                            fail_reason = f"Failed during action cycle {cycle_num_display} Attempt {task.reattempt_count+1} (Tool Error: {tool_error}). Max action retries + Failed to reset task state."
                            task.reflections = fail_reason
                            task.cumulative_findings += f"\n--- Task Failed: Max action retries & reset failure on cycle {cycle_num_display}. ---\n"
                            self.task_queue.update_task(
                                task.id, "failed", reflections=fail_reason
                            )
                            task.status = "failed"
                            task_status_updated_this_cycle = True
                            cycle_status = "failed"
                    else:  # Max task reattempts also reached
                        log.error(
                            f"{self.log_prefix} Max action retries AND max task reattempts reached for task {task.id[:8]} after tool failure. Failing permanently."
                        )
                        fail_reason = f"Failed during action cycle {cycle_num_display}. Max action retries and max task reattempts reached. Last error (Tool: {tool_name}): {tool_error}"
                        task.reflections = fail_reason
                        task.cumulative_findings += f"\n--- Task Failed Permanently: Max action and task retries reached on cycle {cycle_num_display}. ---\n"
                        self.task_queue.update_task(
                            task.id, "failed", reflections=fail_reason
                        )
                        task.status = "failed"
                        task_status_updated_this_cycle = True
                        cycle_status = "failed"
            else:  # Unknown tool status
                log.error(
                    f"{self.log_prefix} Tool '{tool_name}' action '{tool_action_from_result}' returned UNKNOWN status '{tool_status}' in cycle {cycle_num_display}. Failing task."
                )
                action_log.append(
                    f"[CRITICAL] Tool '{tool_name}' ({tool_action_from_result}) unknown status: {tool_status}"
                )
                fail_reason = f"Failed during action cycle {cycle_num_display} Attempt {task.reattempt_count+1} due to unknown tool status '{tool_status}' from tool {tool_name}."
                task.reflections = fail_reason
                task.cumulative_findings += f"\n--- Task Failed: Unknown tool status '{tool_status}' in cycle {cycle_num_display}. ---\n"
                self.task_queue.update_task(task.id, "failed", reflections=fail_reason)
                task.status = "failed"
                task_status_updated_this_cycle = True
                cycle_status = "failed"

        elif action_type == "final_answer":
            # --- Handle Final Answer ---
            log.info(
                f"{self.log_prefix} [ACTION] Cycle {cycle_num_display}: Provide Final Answer (Attempt {task.reattempt_count+1})."
            )
            action_log.append("[ACTION] Provide Final Answer.")
            answer = action.get("answer", "").strip()
            reflections = action.get("reflections", "").strip()

            if not answer:
                log.error(
                    f"{self.log_prefix} LLM chose final_answer but provided empty answer text."
                )
                # Treat this like an LLM action error
                action_message = (
                    "LLM chose final_answer but provided no ANSWER content."
                )
                action_subtype = "parse_error"
                cycle_status = "error_retry"  # Force retry of thinking step
                action_details_for_history["result_status"] = "error"
                action_details_for_history["result_summary"] = action_message
                # Don't modify task state yet, just retry the thinking step
                with self._state_lock:  # RLock
                    self.session_state["current_action_retries"] = (
                        current_retries + 1
                    )  # Use a retry slot

            else:  # Valid answer provided
                final_answer_text = answer  # Store for UI update
                action_details_for_history["result_status"] = "completed"
                summary_limit = 500  # Define limit for history summary
                action_details_for_history["result_summary"] = (
                    f"Final Answer Provided:\n{answer[:summary_limit]}"
                    + ("..." if len(answer) > summary_limit else "")
                )

                log.info(
                    f" FINAL ANSWER (Agent: {self.agent_id}, Task: {task.id[:8]}) "
                    + f"\n{answer}\n"
                )  # Adjusted separator length

                cycle_finding = f"\n--- Action Cycle {cycle_num_display}: Final Answer Provided (Attempt {task.reattempt_count+1}) ---\n{answer[:500]}...\n---\n"
                task.cumulative_findings += cycle_finding

                result_payload = {
                    "answer": answer,
                    "action_cycles": cycle_num_display,
                    "task_attempts": task.reattempt_count + 1,
                }
                task.result = result_payload
                task.reflections = reflections
                task.completed_at = datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat()

                # Update task queue
                self.task_queue.update_task(
                    task.id, "completed", result=result_payload, reflections=reflections
                )
                task.status = "completed"  # Update local object
                task_status_updated_this_cycle = True
                cycle_status = "completed"

                # Save QLoRA data point
                self._save_qlora_datapoint(
                    source_type="task_completion",
                    instruction="Given the overall task and cumulative findings, provide the final answer.",
                    input_context=f"Overall Task: {task.description}\n\nCumulative Findings:\n{task.cumulative_findings}",  # Use full findings
                    output=answer,
                )

                # Add result/reflections to memory
                try:
                    self.memory.add_memory(
                        f"Final Answer (Action Cycle {cycle_num_display}, Attempt {task.reattempt_count+1}):\n{answer}",
                        {
                            "type": "task_result",
                            "task_id": task.id,
                            "final_cycle": cycle_num_display,
                        },
                    )
                    if reflections:
                        self.memory.add_memory(
                            f"Final Reflections (Action Cycle {cycle_num_display}):\n{reflections}",
                            {
                                "type": "task_reflection",
                                "task_id": task.id,
                                "final_cycle": cycle_num_display,
                            },
                        )
                except Exception as e:
                    log.error(
                        f"{self.log_prefix} Failed to add final answer/reflection memory: {e}"
                    )

                # Summarize and prune memories for completed task
                self._summarize_and_prune_task_memories(task)

        # --- Post-Action Cycle Updates ---
        cycle_duration = time.time() - cycle_start_time
        action_log.append(
            f"Action Cycle {cycle_num_display} duration: {cycle_duration:.2f}s. Cycle Outcome: {cycle_status}"
        )

        # Update action history (use lock for deque)
        action_details_for_history["log_snippet"] = "\n".join(
            action_log[-min(5, len(action_log)) :]
        )  # Add log snippet at the end
        with self._state_lock:  # RLock
            if action_log:
                # Append the fully populated details
                self.session_state["last_action_details"].append(
                    action_details_for_history
                )
                # Ensure UI state reflects the updated list immediately (important!)
                self._ui_update_state["action_history"] = list(
                    self.session_state["last_action_details"]
                )
            else:
                log.warning(
                    f"{self.log_prefix} Action log was empty, skipping adding details to history."
                )

        # Update session state based on cycle outcome
        if cycle_status in ["completed", "failed"]:
            log.info(
                f"{self.log_prefix} Task {task.id[:8]} finished permanently with status: {cycle_status}"
            )
            # Fetch the final task object state from the queue for completion handling
            final_task_obj = self.task_queue.get_task(task.id)
            if final_task_obj:
                # Handle completion/failure (increments counter, potentially revises identity)
                self._handle_task_completion_or_failure(final_task_obj)
            else:
                log.error(
                    f"{self.log_prefix} Could not retrieve final task object for {task.id[:8]} after completion/failure."
                )
            # Clear current task from session state
            with self._state_lock:  # RLock
                self.session_state["current_task_id"] = None
                self.session_state["current_action_retries"] = 0
                # User suggestion flag was reset earlier if used

        elif cycle_status == "error_reattempting_task":
            # Task ID remains, retries reset by _reflect_on_error_and_prepare_reattempt
            pass
        elif cycle_status == "error_retry":
            # Task ID remains, retries incremented earlier
            pass
        elif cycle_status == "processing":
            # Task ID remains, retries reset earlier
            pass

        # Save session state changes (retries, current task ID, revision counter etc.)
        self.save_session_state()

        # --- Prepare UI Update Data ---
        # Get potentially updated task status for UI display
        final_task_status_obj = self.task_queue.get_task(task.id)
        final_task_status_for_ui = (
            final_task_status_obj.status if final_task_status_obj else task.status
        )

        # Get dependent tasks based on the *current* task ID in session state
        dependent_tasks_list = []
        current_task_id_for_ui = self.session_state.get("current_task_id")
        if current_task_id_for_ui:
            try:
                dependent_tasks = self.task_queue.get_dependent_tasks(
                    current_task_id_for_ui
                )
                dependent_tasks_list = [
                    {"id": dt.id, "description": dt.description}
                    for dt in dependent_tasks
                ]
            except Exception as e:
                log.error(f"{self.log_prefix} Failed to get dependent tasks: {e}")

        # Get recent memories for UI display (best effort)
        recent_memories_for_ui = []
        try:
            ui_memory_query = f"Memories relevant to outcome '{cycle_status}' of action cycle {cycle_num_display} for task {task.id[:8]} (Attempt {task.reattempt_count+1}). Last Action: {action_type}."
            recent_memories_for_ui, _ = self.memory.retrieve_and_rerank_memories(
                query=ui_memory_query,
                task_description=task.description,
                context=task.cumulative_findings[-1000:],
                identity_statement=self.identity_statement,
                n_final=5,  # Limit for UI display
            )
        except Exception as e:
            log.error(
                f"{self.log_prefix} Failed to get recent memories for UI update: {e}"
            )

        # Update internal UI state dictionary (uses lock)
        self._update_ui_state(
            status=final_task_status_for_ui,  # Use status from potentially updated task obj
            log="\n".join(action_log),  # Full log for this cycle
            thinking=thinking_to_store,
            dependent_tasks=dependent_tasks_list,
            last_action_type=action_type,
            last_tool_results=tool_results_for_ui,  # Store raw output for next cycle
            recent_memories=recent_memories_for_ui,
            # Fetch latest web content from locked session state
            last_web_content=self.session_state.get(
                "last_web_browse_content", "(No recent web browse)"
            ),
            final_answer=final_answer_text,  # Display final answer if generated
        )

        # Return the latest full UI state for this agent
        return self.get_ui_update_state()

    # _handle_task_completion_or_failure <<< MODIFIED >>>
    def _handle_task_completion_or_failure(self, task: Task):
        """Handles post-task actions: increments counter, triggers identity revision if interval met."""
        log.info(
            f"{self.log_prefix} Handling completion/failure for task {task.id[:8]} (Status: {task.status})"
        )

        # Check if agent is running or paused *before* potentially long revision call
        should_attempt_revision = False
        if self._is_running.is_set() or self._ui_update_state.get("status") in [
            "paused",
            "idle",
        ]:
            # Increment task counter and check if revision interval is met
            with self._state_lock:
                tasks_completed = self.session_state.get("tasks_since_last_revision", 0)
                tasks_completed += 1
                self.session_state["tasks_since_last_revision"] = tasks_completed
                log.info(
                    f"{self.log_prefix} Task {task.status}. Tasks since last revision: {tasks_completed}/{config.IDENTITY_REVISION_TASK_INTERVAL}"
                )

                if (
                    tasks_completed >= config.IDENTITY_REVISION_TASK_INTERVAL
                    and config.IDENTITY_REVISION_TASK_INTERVAL > 0
                ):
                    log.info(
                        f"{self.log_prefix} Identity revision interval ({config.IDENTITY_REVISION_TASK_INTERVAL} tasks) reached. Triggering revision."
                    )
                    should_attempt_revision = True
                    self.session_state["tasks_since_last_revision"] = 0  # Reset counter
                else:
                    log.info(
                        f"{self.log_prefix} Identity revision interval not met. Skipping revision."
                    )

            # Save the updated counter state immediately
            self.save_session_state()

            # Attempt revision outside the lock if the flag was set
            if should_attempt_revision:
                try:
                    reason = f"After completing/failing {config.IDENTITY_REVISION_TASK_INTERVAL} tasks (last: {task.status.capitalize()} task {task.id[:8]} - {task.description[:50]}...). Reflecting on recent performance and goals."
                    self._revise_identity_statement(reason)
                except Exception as e:
                    log.error(
                        f"{self.log_prefix} Error during scheduled identity revision after task {task.id[:8]}: {e}",
                        exc_info=True,
                    )
                    # Optionally, try to re-increment counter if revision failed? Risky.
        else:
            log.info(
                f"{self.log_prefix} Agent paused or shutting down, skipping identity revision check after {task.status} task."
            )

    # process_one_step remains the same logic, logs agent ID
    def process_one_step(self) -> Dict[str, Any]:
        log.info(f"{self.log_prefix} Processing one agent cycle...")
        cycle_result_log = [f"{self.log_prefix} Attempting one cycle..."]
        # Get current UI state as default (ensures agent ID/name are present)
        current_ui_state = self.get_ui_update_state()

        try:
            # --- Check ChromaDB Connection ---
            # Added this check earlier in the cycle
            try:
                db_count = self.memory.collection.count()
                log.debug(f"{self.log_prefix} Memory DB check OK (Count: {db_count}).")
            except Exception as db_err:
                log.error(
                    f"{self.log_prefix} CRITICAL: Memory DB connection check failed: {db_err}",
                    exc_info=True,
                )
                # Update UI state to reflect critical error and return
                self._update_ui_state(
                    status="critical_error",
                    log=f"{self.log_prefix} CRITICAL: Memory DB error: {db_err}",
                )
                self.pause_autonomous_loop()  # Pause agent on critical error
                return self.get_ui_update_state()
            # --- End DB Check ---

            task_id_to_process = self.session_state.get("current_task_id")
            task: Optional[Task] = None

            # 1. Check if currently working on a valid task
            if task_id_to_process:
                task = self.task_queue.get_task(task_id_to_process)
                if not task or task.status not in ["planning", "in_progress"]:
                    status_reason = task.status if task else "Not Found"
                    log.warning(
                        f"{self.log_prefix} Task {task_id_to_process} from state invalid/finished (Status: {status_reason}). Resetting session."
                    )
                    cycle_result_log.append(
                        f"Task {task_id_to_process} invalid/finished ({status_reason})."
                    )
                    task = None
                    with self._state_lock:  # RLock
                        self.session_state["current_task_id"] = None
                        self.session_state["current_action_retries"] = 0
                        self.session_state["user_suggestion_move_on_pending"] = False
                    # Update UI state immediately to reflect idle status
                    self._update_ui_state(
                        status="idle",
                        log=f"{self.log_prefix} Idle - Previous task {task_id_to_process} finished.",
                        current_task_id=None,
                        current_task_desc="N/A",
                        current_action_desc="N/A",
                        current_plan="N/A",
                        thinking="(Idle)",
                    )
                    self.save_session_state()  # Save cleared state
                    # Don't immediately generate new tasks here, let next cycle handle idle state

            # 2. If no current valid task, get the next one
            if not task:
                task = self.task_queue.get_next_task()

                if not task:  # No runnable tasks found
                    log.info(f"{self.log_prefix} No runnable tasks found.")
                    cycle_result_log.append("No runnable tasks found.")
                    # Only generate tasks if truly idle (no task ID in session state)
                    if not self.session_state.get("current_task_id"):
                        log.info(
                            f"{self.log_prefix} Agent idle, considering generating new tasks..."
                        )
                        try:
                            first_new_task_desc = self.generate_new_tasks(
                                max_new_tasks=3, trigger_context="idle"
                            )
                            if first_new_task_desc:
                                cycle_result_log.append(
                                    f"Generated idle task: {first_new_task_desc[:60]}..."
                                )
                            else:
                                cycle_result_log.append("No new idle tasks generated.")
                        except Exception as gen_err:
                            log.error(
                                f"{self.log_prefix} Error during idle task generation: {gen_err}",
                                exc_info=True,
                            )
                            cycle_result_log.append(
                                f"Error generating idle tasks: {gen_err}"
                            )

                    # Update UI to show idle status and log message
                    self._update_ui_state(
                        status="idle",
                        log="\n".join(cycle_result_log),
                        current_task_id=None,
                        current_task_desc="N/A",
                        current_action_desc="N/A",
                        current_plan="N/A",
                        thinking="(Idle - No tasks)",
                    )
                    return self.get_ui_update_state()  # End cycle if no tasks

                # Found a new task to start
                log.info(
                    f"{self.log_prefix} Starting task: {task.id[:8]} - '{task.description[:60]}...' (Attempt {task.reattempt_count + 1})"
                )
                cycle_result_log.append(
                    f"Starting task {task.id[:8]}: {task.description[:60]}... (Attempt {task.reattempt_count + 1})"
                )
                self.task_queue.update_task(task.id, "planning")
                task.status = "planning"  # Update local object

                # Update session state
                with self._state_lock:  # RLock
                    self.session_state["current_task_id"] = task.id
                    # Reset action retries for the new task
                    self.session_state["current_action_retries"] = 0
                    self.session_state["user_suggestion_move_on_pending"] = False
                    # Clear plan/findings only if it's the first attempt
                    if task.reattempt_count == 0:
                        task.plan = None
                        task.current_step_index = 0
                        task.cumulative_findings = ""

                self.save_session_state()

                # Update UI state for the new planning task
                self._update_ui_state(
                    status=task.status,
                    log="\n".join(cycle_result_log),
                    thinking="(Starting task planning...)",
                    current_task_id=task.id,  # Ensure UI state has new task ID
                    # _refresh_ui_task_details will be called by _update_ui_state
                )
                # Don't return yet, proceed to planning phase in this cycle

            # 3. Process the task based on its status
            if task.status == "planning":
                log.info(
                    f"{self.log_prefix} Task {task.id[:8]} is planning (Attempt {task.reattempt_count + 1}). Generating plan..."
                )
                cycle_result_log.append(
                    f"Generating plan for task {task.id[:8]} (Attempt {task.reattempt_count + 1})..."
                )
                # Update UI to show planning status before potentially long LLM call
                self._update_ui_state(
                    status="planning", thinking="(Generating plan...)"
                )

                plan_success = self._generate_task_plan(task)

                if plan_success:
                    log.info(
                        f"{self.log_prefix} Plan generated successfully for task {task.id[:8]}. Moving to in_progress."
                    )
                    cycle_result_log.append("Plan generated successfully.")
                    self.task_queue.update_task(task.id, "in_progress")
                    task.status = "in_progress"  # Update local object
                    task.current_step_index = 0  # Ensure step index is reset
                    # No need to save queue here, update_task does it
                    # No need to save session state here, no relevant changes
                    # Update UI state for starting execution
                    self._update_ui_state(
                        status=task.status,
                        log="\n".join(cycle_result_log),
                        thinking="(Plan generated, starting execution...)",
                    )
                    return (
                        self.get_ui_update_state()
                    )  # End cycle after successful planning

                else:  # Plan generation failed
                    log.error(
                        f"{self.log_prefix} Failed to generate plan for task {task.id[:8]} (Attempt {task.reattempt_count + 1})."
                    )
                    cycle_result_log.append("ERROR: Failed to generate plan.")
                    fail_reason = f"Failed during planning phase (Attempt {task.reattempt_count + 1})."

                    # Reattempt planning if attempts remain
                    if task.reattempt_count < config.TASK_MAX_REATTEMPT:
                        log.warning(
                            f"{self.log_prefix} Planning failed, reattempts remain ({task.reattempt_count}/{config.TASK_MAX_REATTEMPT}). Reflecting and retrying planning."
                        )
                        cycle_result_log.append(
                            f"Planning failed. Attempting task re-plan (Attempt {task.reattempt_count + 2})."
                        )
                        # Reflect and prepare task queue object for reattempt
                        if self._reflect_on_error_and_prepare_reattempt(
                            task,
                            "Failed to generate a valid execution plan.",
                            "planning_error",
                            "Planning Phase",
                        ):
                            # Reset action retries in session state for the new planning attempt
                            with self._state_lock:  # RLock
                                self.session_state["current_action_retries"] = 0
                            self.save_session_state()
                            # Update UI state to show re-planning
                            self._update_ui_state(
                                status="planning",  # Remains planning
                                log="\n".join(cycle_result_log),
                                thinking="(Re-planning task after previous planning failure...)",
                            )
                            return (
                                self.get_ui_update_state()
                            )  # End cycle, next cycle will retry planning
                        else:
                            # If reset fails, add to fail reason and proceed to fail task
                            fail_reason += " Additionally failed to reset task state for reattempt."
                            log.error(
                                f"{self.log_prefix} Also failed to reset task {task.id[:8]} state. Failing permanently."
                            )

                    # If no reattempts left or reset failed, fail the task permanently
                    log.error(
                        f"{self.log_prefix} Failing task {task.id[:8]} due to planning failure."
                    )
                    task.reflections = fail_reason
                    self.task_queue.update_task(
                        task.id, "failed", reflections=fail_reason
                    )
                    failed_task_obj = self.task_queue.get_task(
                        task.id
                    )  # Get updated obj
                    if failed_task_obj:
                        self._handle_task_completion_or_failure(failed_task_obj)
                    else:
                        log.error(
                            f"{self.log_prefix} Could not retrieve final task object for {task.id[:8]} after planning failure."
                        )

                    # Clear session state
                    with self._state_lock:  # RLock
                        self.session_state["current_task_id"] = None
                        self.session_state["current_action_retries"] = 0
                        self.session_state["user_suggestion_move_on_pending"] = False
                    self.save_session_state()

                    # Update UI to idle
                    self._update_ui_state(
                        status="idle",
                        log="\n".join(cycle_result_log),
                        current_task_id=None,
                        current_task_desc="N/A",
                        current_action_desc="N/A",
                        current_plan="N/A",
                        thinking="(Idle - Planning Failed)",
                    )
                    return (
                        self.get_ui_update_state()
                    )  # End cycle after planning failure

            elif task.status == "in_progress":
                # Perform one action cycle (handles its own state updates and returns UI state)
                return self._perform_action_cycle(task)

            else:  # Should not happen if logic above is correct
                log.warning(
                    f"{self.log_prefix} Task {task.id[:8]} found but not in expected state (Status: {task.status}). Resetting session."
                )
                cycle_result_log.append(
                    f"Task {task.id[:8]} state unexpected: {task.status}. Resetting session."
                )
                # Reset state and return idle UI
                with self._state_lock:  # RLock
                    self.session_state["current_task_id"] = None
                    self.session_state["current_action_retries"] = 0
                    self.session_state["user_suggestion_move_on_pending"] = False
                self.save_session_state()
                self._update_ui_state(
                    status="idle",
                    log="\n".join(cycle_result_log),
                    current_task_id=None,
                    current_task_desc="N/A",
                    current_action_desc="N/A",
                    current_plan="N/A",
                    thinking="(Idle - Unexpected Task State)",
                )
                return self.get_ui_update_state()

        except Exception as e:
            # --- Critical Error Handling for the entire cycle ---
            log.exception(
                f"{self.log_prefix} CRITICAL Error during process_one_step loop"
            )
            cycle_result_log.append(f"[CRITICAL ERROR]: {traceback.format_exc()}")
            current_task_id_on_error = self.session_state.get("current_task_id")

            # Attempt to fail the current task if there is one
            if current_task_id_on_error:
                log.error(
                    f"{self.log_prefix} Failing task {current_task_id_on_error[:8]} due to critical loop error."
                )
                fail_reason = f"Critical loop error: {e}"
                task_to_fail = self.task_queue.get_task(current_task_id_on_error)
                if task_to_fail:
                    # Append error to reflections if possible
                    existing_reflections = task_to_fail.reflections or ""
                    task_to_fail.reflections = (
                        existing_reflections + f"\n[Loop Error]: {fail_reason}"
                    )
                    self.task_queue.update_task(
                        current_task_id_on_error,
                        "failed",
                        reflections=task_to_fail.reflections,
                    )
                    # Call completion handler for the failed task
                    self._handle_task_completion_or_failure(task_to_fail)
                else:
                    log.error(
                        f"{self.log_prefix} Could not find task {current_task_id_on_error[:8]} to mark as failed after loop error."
                    )

            # Reset session state regardless
            with self._state_lock:  # RLock
                self.session_state["current_task_id"] = None
                self.session_state["current_action_retries"] = 0
                self.session_state["user_suggestion_move_on_pending"] = False
            self.save_session_state()

            # Update UI to critical error state
            self._update_ui_state(
                status="critical_error",
                log="\n".join(cycle_result_log),
                current_task_id=None,
                current_task_desc="N/A",
                current_action_desc="N/A",
                current_plan="N/A",
                thinking="(Critical Error)",
            )
            # Pause the agent's loop on critical error
            self.pause_autonomous_loop()
            return self.get_ui_update_state()

    # generate_new_tasks uses agent-specific initial prompt, logs agent ID
    def generate_new_tasks(
        self,
        max_new_tasks: int = 3,
        last_user_message: Optional[str] = None,
        last_assistant_response: Optional[str] = None,
        trigger_context: str = "unknown",
    ) -> Optional[str]:
        log.info(
            f"{self.log_prefix} --- Attempting to Generate New Tasks (Trigger: {trigger_context}) ---"
        )
        memory_is_empty = False
        memory_count = 0
        try:
            # Verify memory connection before proceeding
            memory_count = self.memory.collection.count()
            memory_is_empty = memory_count == 0
            log.info(f"{self.log_prefix} Memory count for task gen: {memory_count}")
        except Exception as e:
            log.error(
                f"{self.log_prefix} Failed get memory count for task gen: {e}. Assuming not empty.",
                exc_info=True,
            )
            memory_is_empty = False  # Assume not empty on error

        use_initial_creative_prompt = memory_is_empty and trigger_context == "idle"
        max_tasks_to_generate = (
            config.INITIAL_NEW_TASK_N if use_initial_creative_prompt else max_new_tasks
        )

        # Select prompt template
        if use_initial_creative_prompt:
            prompt_template_key = self.initial_tasks_prompt_key
            prompt_template = getattr(prompts, prompt_template_key, None)
            if not prompt_template:
                log.warning(
                    f"{self.log_prefix} Agent-specific initial prompt '{prompt_template_key}' not found. Falling back."
                )
                prompt_template = (
                    prompts.INITIAL_CREATIVE_TASK_GENERATION_PROMPT_AGENT_01
                )  # Generic fallback
        else:
            prompt_template = prompts.GENERATE_NEW_TASKS_PROMPT

        log.info(
            f"{self.log_prefix} Using {'Initial Creative' if use_initial_creative_prompt else 'Standard'} Task Gen Prompt. Max tasks: {max_tasks_to_generate}"
        )

        prompt_vars = {}
        if use_initial_creative_prompt:
            # Minimal context for initial prompt
            prompt_vars = {
                "identity_statement": self.identity_statement,
                "tool_desc": self.get_available_tools_description(),
                "max_new_tasks": max_tasks_to_generate,
                # Add placeholders for keys used in standard prompt but not initial
                "context_query": "N/A (Initial Run)",
                "mem_summary": "None (Initial Run)",
                "active_tasks_summary": "None (Initial Run)",
                "completed_failed_summary": "None (Initial Run)",
                "critical_evaluation_instruction": "N/A (Initial Run - Generate starting tasks)",
            }
        else:  # Standard task generation
            context_source = ""
            mem_query = ""
            critical_evaluation_instruction = ""
            context_query = ""  # Define context_query here

            if (
                trigger_context == "chat"
                and last_user_message
                and last_assistant_response
            ):
                context_source = "last chat interaction"
                context_query = f"Last User: {last_user_message}\nLast Assistant: {last_assistant_response}"
                mem_query = f"Context relevant to last chat: {last_user_message}"
                critical_evaluation_instruction = "\n**Critically Evaluate Need:** Based *specifically* on the **Last Interaction**, is a background task *truly necessary*? Output `[]` if not."
            else:  # Idle or other triggers
                context_source = "general agent state"
                context_query = "General status. Consider logical follow-up/exploration based on completed tasks, idle state, and my identity."
                mem_query = "Recent activities, conclusions, errors, reflections, summaries, identity revisions."
                critical_evaluation_instruction = "\n**Critically Evaluate Need:** Are new tasks genuinely needed for exploration/follow-up, consistent with identity? Output `[]` if not."

            log.info(
                f"{self.log_prefix} Retrieving context for task generation (Source: {context_source})..."
            )

            # Get memories
            try:
                recent_mems, _ = self.memory.retrieve_and_rerank_memories(
                    query=mem_query,
                    task_description="Task Generation Context",
                    context=context_query,
                    identity_statement=self.identity_statement,
                    n_results=config.MEMORY_COUNT_NEW_TASKS * 2,
                    n_final=config.MEMORY_COUNT_NEW_TASKS,
                )
            except Exception as e:
                log.error(
                    f"{self.log_prefix} Failed retrieving memories for task gen: {e}"
                )
                recent_mems = []

            mem_summary_list = []
            for m in recent_mems:
                relative_time = format_relative_time(m["metadata"].get("timestamp"))
                mem_type = m["metadata"].get("type", "mem")
                snippet = m["content"][:150].strip().replace("\n", " ")
                mem_summary_list.append(f"- [{relative_time}] {mem_type}: {snippet}...")
            mem_summary = "\n".join(mem_summary_list) if mem_summary_list else "None"

            # Get task summaries
            existing_tasks_info = [
                {"id": t.id, "description": t.description, "status": t.status}
                for t in self.task_queue.tasks.values()
            ]
            active_tasks_summary = (
                "\n".join(
                    [
                        f"- ID: {t['id'][:8]} (Status: {t['status']}) Desc: {t['description'][:100]}..."
                        for t in existing_tasks_info
                        if t["status"] in ["pending", "planning", "in_progress"]
                    ]
                )
                or "None"
            )
            # Limit completed/failed summary length
            completed_failed_summary = (
                "\n".join(
                    [
                        f"- ID: {t['id'][:8]} (Status: {t['status']}) Desc: {t['description'][:100]}..."
                        for t in sorted(
                            [
                                t
                                for t in existing_tasks_info
                                if t["status"] in ["completed", "failed"]
                            ],
                            key=lambda x: getattr(
                                x, "completed_at", getattr(x, "created_at", "0")
                            ),  # Sort robustly
                            reverse=True,
                        )[
                            :10
                        ]  # Limit to recent 10
                    ]
                )
                or "None"
            )

            prompt_vars = {
                "identity_statement": self.identity_statement,
                "context_query": context_query,
                "mem_summary": mem_summary,
                "active_tasks_summary": active_tasks_summary,
                "completed_failed_summary": completed_failed_summary,
                "critical_evaluation_instruction": critical_evaluation_instruction,
                "max_new_tasks": max_tasks_to_generate,
                # Add placeholders for keys used in initial prompt but not standard
                "tool_desc": (
                    self.get_available_tools_description()
                    if "tool_desc" not in prompt_vars
                    else prompt_vars["tool_desc"]
                ),
            }

        # --- Call LLM for Task Generation ---
        try:
            # Ensure all expected keys are present before formatting
            required_keys = [
                "identity_statement",
                "tool_desc",
                "max_new_tasks",
                "context_query",
                "mem_summary",
                "active_tasks_summary",
                "completed_failed_summary",
                "critical_evaluation_instruction",
            ]
            for key in required_keys:
                if key not in prompt_vars:
                    log.error(
                        f"{self.log_prefix} Missing key '{key}' in prompt_vars for task generation!"
                    )
                    # Provide a default value to avoid crashing format
                    prompt_vars[key] = f"<{key.upper()}_MISSING>"

            prompt = prompt_template.format(**prompt_vars)
        except KeyError as e:
            log.error(
                f"{self.log_prefix} Missing key '{e}' when formatting task generation prompt. Vars: {prompt_vars}"
            )
            return None  # Cannot proceed if prompt formatting fails

        log.info(
            f"{self.log_prefix} Asking {self.ollama_chat_model} to generate up to {max_tasks_to_generate} new tasks..."
        )
        llm_response = call_ollama_api(
            prompt, self.ollama_chat_model, self.ollama_base_url, timeout=180
        )
        if not llm_response:
            log.error(f"{self.log_prefix} LLM failed task gen.")
            return None

        # --- Parse and Add Tasks ---
        first_task_desc_added = None
        new_tasks_added = 0
        try:
            # Robust JSON List Parsing
            json_str = llm_response.strip()
            # Remove potential markdown fences
            json_str = re.sub(
                r"^```json\s*", "", json_str, flags=re.MULTILINE | re.IGNORECASE
            )
            json_str = re.sub(
                r"\s*```$", "", json_str, flags=re.MULTILINE | re.IGNORECASE
            ).strip()

            suggested_tasks = []
            try:
                # Try direct parsing first
                potential_tasks = json.loads(json_str)
                if isinstance(potential_tasks, list):
                    suggested_tasks = potential_tasks
                else:
                    log.warning(
                        f"{self.log_prefix} LLM task gen result was not a list: {type(potential_tasks)}. Trying object extraction."
                    )
                    # If it's a dict with a key containing a list? (Less likely based on prompt)

            except json.JSONDecodeError as e:
                log.warning(
                    f"{self.log_prefix} Direct JSON list parse failed: {e}. Trying extraction..."
                )
                # Fallback: Try to find JSON objects within the text
                try:
                    # Find all {...} patterns, trying to be non-greedy
                    json_objects_match = re.findall(r"(\{.*?\})", json_str, re.DOTALL)
                    if json_objects_match:
                        log.warning(
                            f"{self.log_prefix} Found {len(json_objects_match)} JSON-like objects. Attempting to parse individually."
                        )
                        parsed_objects = []
                        for obj_str in json_objects_match:
                            try:
                                # Try parsing each found object
                                parsed_obj = json.loads(obj_str)
                                if isinstance(parsed_obj, dict):
                                    parsed_objects.append(parsed_obj)
                                else:
                                    log.warning(
                                        f"{self.log_prefix} Extracted object is not a dict: {obj_str}"
                                    )
                            except json.JSONDecodeError:
                                log.warning(
                                    f"{self.log_prefix} Failed to parse extracted object: {obj_str}"
                                )
                        if parsed_objects:
                            suggested_tasks = parsed_objects  # Use the list of successfully parsed dicts
                        else:
                            raise json.JSONDecodeError(
                                "No valid JSON objects found after regex extraction",
                                json_str,
                                0,
                            )
                    else:
                        # Check for common refusal patterns before failing completely
                        if "[]" in json_str or "no new tasks" in json_str.lower():
                            log.info(
                                f"{self.log_prefix} LLM response indicates no new tasks needed."
                            )
                            return None  # Explicit refusal
                        else:
                            raise json.JSONDecodeError(
                                f"JSON list '[]' not found and no JSON objects detected",
                                json_str,
                                0,
                            )

                except json.JSONDecodeError as final_e:
                    log.error(
                        f"{self.log_prefix} Failed JSON parse task gen after all attempts: {final_e}\nLLM Resp:\n{llm_response}\n---"
                    )
                    return None  # Give up if all parsing fails

            # Proceed with validation if suggested_tasks is a list (even if empty)
            if not isinstance(suggested_tasks, list):
                log.warning(
                    f"{self.log_prefix} LLM task gen result was not a list after parsing: {type(suggested_tasks)}"
                )
                return None

            if not suggested_tasks:
                log.info(
                    f"{self.log_prefix} LLM suggested no new tasks (empty list parsed)."
                )
                return None

            log.info(
                f"{self.log_prefix} LLM suggested {len(suggested_tasks)} tasks. Validating..."
            )

            # --- Validation and Adding Logic ---
            current_task_ids_in_batch = set()
            # Get fresh task info for duplicate check
            existing_tasks_info = [
                {"id": t.id, "description": t.description, "status": t.status}
                for t in self.task_queue.tasks.values()
            ]
            active_task_descriptions = {
                t["description"].strip().lower()
                for t in existing_tasks_info
                if t["status"] in ["pending", "planning", "in_progress"]
            }
            valid_existing_task_ids = {t["id"] for t in existing_tasks_info}

            for task_data in suggested_tasks:
                if not isinstance(task_data, dict):
                    log.warning(
                        f"{self.log_prefix} Skipping non-dict item in suggested tasks: {task_data}"
                    )
                    continue

                description = task_data.get("description")
                if (
                    not description
                    or not isinstance(description, str)
                    or not description.strip()
                ):
                    log.warning(
                        f"{self.log_prefix} Skipping task with invalid/missing description: {task_data}"
                    )
                    continue
                description = description.strip()

                # Duplicate check (case-insensitive)
                if description.lower() in active_task_descriptions:
                    log.warning(
                        f"{self.log_prefix} Skipping duplicate active task: '{description[:80]}...'"
                    )
                    continue

                # Priority handling
                priority = task_data.get(
                    "priority", 3 if use_initial_creative_prompt else 5
                )
                try:
                    priority = max(1, min(10, int(priority)))
                except (ValueError, TypeError):
                    priority = (
                        3 if use_initial_creative_prompt else 5
                    )  # Default priority

                # Dependency handling
                dependencies_raw = task_data.get("depends_on")
                validated_dependencies = []
                if isinstance(dependencies_raw, list):
                    for dep_id in dependencies_raw:
                        dep_id_str = str(dep_id).strip()
                        # Check against existing tasks AND tasks added in THIS batch
                        if (
                            dep_id_str in valid_existing_task_ids
                            or dep_id_str in current_task_ids_in_batch
                        ):
                            validated_dependencies.append(dep_id_str)
                        else:
                            log.warning(
                                f"{self.log_prefix} Dependency '{dep_id_str}' for new task '{description[:50]}...' not found in existing tasks or this batch. Ignoring dependency."
                            )
                elif dependencies_raw is not None:
                    log.warning(
                        f"{self.log_prefix} Invalid 'depends_on' format for task '{description[:50]}...': {dependencies_raw}. Expected list or None."
                    )

                # Create and add the task
                new_task = Task(
                    description, priority, depends_on=validated_dependencies or None
                )
                new_task_id = self.task_queue.add_task(new_task)

                if new_task_id:
                    if new_tasks_added == 0:
                        first_task_desc_added = description
                    new_tasks_added += 1
                    # Update sets for duplicate/dependency checks within this batch
                    active_task_descriptions.add(description.lower())
                    valid_existing_task_ids.add(new_task_id)
                    current_task_ids_in_batch.add(new_task_id)
                    # Logging is now handled inside add_task
                else:
                    log.warning(
                        f"{self.log_prefix} Failed to add generated task (likely duplicate or internal error): {description}"
                    )

                if new_tasks_added >= max_tasks_to_generate:
                    log.info(
                        f"{self.log_prefix} Reached max task generation limit ({max_tasks_to_generate})."
                    )
                    break  # Stop processing more suggestions

        except Exception as e:
            log.exception(
                f"{self.log_prefix} Unexpected error during task generation parsing/validation: {e}"
            )
            return None  # Return None on unexpected errors

        log.info(
            f"{self.log_prefix} Finished Task Generation: Added {new_tasks_added} new tasks."
        )
        return first_task_desc_added

    # _autonomous_loop runs for this specific agent instance
    def _autonomous_loop(self, initial_delay: float = 1.0, step_delay: float = 3.0):
        """The main autonomous execution loop running in a separate thread."""
        log.info(f"{self.log_prefix} Background agent loop starting.")
        # Short delay before starting the very first cycle
        time.sleep(initial_delay)

        while not self._shutdown_request.is_set():
            if not self._is_running.is_set():
                # Update status once when pausing, then sleep
                # Lock is acquired inside _update_ui_state
                if self._ui_update_state["status"] != "paused":
                    log.info(f"{self.log_prefix} Agent loop pausing...")
                    # Use the update method which handles locking and refresh
                    self._update_ui_state(
                        status="paused", log=f"{self.log_prefix} Agent loop paused."
                    )
                time.sleep(1)  # Sleep longer while paused
                continue

            # --- Agent is running ---
            log.debug(f"{self.log_prefix} Autonomous loop: Starting cycle...")
            step_start = time.monotonic()
            cycle_outcome_status = "unknown"

            try:
                # Process one step (planning or action execution)
                # This function now handles its own UI state updates internally
                ui_state_after_step = self.process_one_step()
                cycle_outcome_status = ui_state_after_step.get("status", "unknown")
                log.debug(
                    f"{self.log_prefix} Cycle finished. Outcome status: {cycle_outcome_status}"
                )

                # Task completion/failure check for UI auto-switch is handled in app_ui.py

            except Exception as e:
                # Catch unexpected errors within the loop itself
                log.exception(
                    f"{self.log_prefix} CRITICAL UNHANDLED ERROR in autonomous loop iteration."
                )
                self._update_ui_state(
                    status="critical_error",
                    log=f"{self.log_prefix} CRITICAL LOOP ERROR: {e}\n{traceback.format_exc()}",
                )
                # Pause the loop on critical errors to prevent repeated failures
                self._is_running.clear()  # This is safe to call outside lock
                # Longer sleep after critical error before next check
                time.sleep(step_delay * 5)

            # Calculate delay for next step
            step_duration = time.monotonic() - step_start
            # Use a shorter base delay, let tasks themselves take time
            effective_step_delay = config.UI_UPDATE_INTERVAL * 2.0  # Base on UI refresh
            remaining_delay = max(
                0.1, effective_step_delay - step_duration
            )  # Ensure minimum delay

            log.debug(
                f"{self.log_prefix} Cycle took {step_duration:.2f}s. Sleeping {remaining_delay:.2f}s."
            )

            # Sleep only if still running and shutdown not requested
            if self._is_running.is_set() and not self._shutdown_request.is_set():
                time.sleep(remaining_delay)
            else:
                log.debug(
                    f"{self.log_prefix} Agent paused or shutdown requested during delay calculation, skipping sleep."
                )

        # --- End of Loop ---
        log.info(
            f"{self.log_prefix} Background agent loop exiting due to shutdown request."
        )
        # Final UI update on shutdown
        self._update_ui_state(
            status="shutdown", log=f"{self.log_prefix} Agent loop stopped."
        )

    # start/pause/shutdown operate on this specific agent instance
    def start_autonomous_loop(self):
        """Starts or resumes the agent's autonomous execution loop thread."""
        if self._agent_thread and self._agent_thread.is_alive():
            if not self._is_running.is_set():
                log.info(f"{self.log_prefix} Agent loop resuming...")
                self._is_running.set()
                # Update UI state immediately after setting flag
                self._update_ui_state(
                    status="running", log=f"{self.log_prefix} Agent resumed."
                )
            else:
                log.info(f"{self.log_prefix} Agent loop is already running.")
        else:
            log.info(f"{self.log_prefix} Starting new background agent loop thread...")
            self._shutdown_request.clear()  # Ensure shutdown flag is clear
            self._is_running.set()  # Set running flag
            # Update UI state immediately
            self._update_ui_state(
                status="starting",  # Indicate starting status
                log=f"{self.log_prefix} Agent loop starting...",
            )
            # Create and start the thread
            self._agent_thread = threading.Thread(
                target=self._autonomous_loop,
                args=(
                    1.0,  # Initial delay before first cycle
                    config.UI_UPDATE_INTERVAL * 2.0,  # Base step delay
                ),
                daemon=True,  # Allow program exit even if thread hangs
                name=f"AgentLoop-{self.agent_id}",
            )
            self._agent_thread.start()
            log.info(f"{self.log_prefix} Agent loop thread started.")

    def pause_autonomous_loop(self):
        """Pauses the agent's autonomous execution loop."""
        if self._is_running.is_set():
            log.info(f"{self.log_prefix} Pausing agent loop...")
            self._is_running.clear()  # Clear the running flag first
            # Give the loop a moment to recognize the flag change
            time.sleep(0.1)
            # Update UI state after clearing flag
            # No need for explicit lock here, _update_ui_state handles it
            # Only update if not already paused/shutdown/error
            if self._ui_update_state["status"] not in [
                "paused",
                "shutdown",
                "critical_error",
            ]:
                self._update_ui_state(
                    status="paused", log=f"{self.log_prefix} Agent loop paused."
                )
        else:
            log.info(f"{self.log_prefix} Agent loop is already paused or not running.")

    def shutdown(self):
        """Requests shutdown of the agent's loop and waits for the thread to join."""
        log.info(f"{self.log_prefix} Shutdown requested for agent.")
        self._shutdown_request.set()  # Signal loop to stop
        self._is_running.clear()  # Ensure loop stops processing

        thread_to_join = self._agent_thread  # Local reference
        if thread_to_join and thread_to_join.is_alive():
            log.info(
                f"{self.log_prefix} Waiting for agent thread {thread_to_join.name} to join..."
            )
            # Wait for the thread to finish, with a timeout
            thread_to_join.join(timeout=15)  # Reduced timeout
            if thread_to_join.is_alive():
                log.warning(
                    f"{self.log_prefix} Agent thread {thread_to_join.name} did not join cleanly after 15 seconds."
                )
            else:
                log.info(
                    f"{self.log_prefix} Agent thread {thread_to_join.name} joined successfully."
                )
        else:
            log.info(f"{self.log_prefix} Agent thread not running or already joined.")

        # Update final status
        self._update_ui_state(
            status="shutdown", log=f"{self.log_prefix} Agent shut down."
        )
        # Save state one last time on shutdown
        self.save_session_state()
        log.info(f"{self.log_prefix} Agent shutdown complete.")

    # add_self_reflection remains the same logic, logs agent ID
    def add_self_reflection(
        self, reflection: str, reflection_type: str = "self_reflection"
    ):
        if not reflection or not isinstance(reflection, str) or not reflection.strip():
            log.warning(f"{self.log_prefix} Skipping add reflection: Empty content.")
            return None
        # Check running state before adding memory
        if (
            self._is_running.is_set() or self._shutdown_request.is_set()
        ):  # Allow adding during shutdown process too? Maybe not.
            # Let's only add if actively running or paused, not during shutdown
            if self._ui_update_state.get("status") not in [
                "shutdown",
                "critical_error",
            ]:
                log.info(f"{self.log_prefix} Adding {reflection_type} to memory...")
                try:
                    return self.memory.add_memory(reflection, {"type": reflection_type})
                except Exception as e:
                    log.error(
                        f"{self.log_prefix} Failed to add self-reflection memory: {e}"
                    )
                    return None
            else:
                log.info(
                    f"{self.log_prefix} Agent status is {self._ui_update_state.get('status')}, skipping add {reflection_type}."
                )
                return None
        else:  # Not running, not shutting down (i.e., paused or starting)
            log.info(
                f"{self.log_prefix} Agent paused, skipping adding {reflection_type} to memory."
            )
            return None

    # generate_and_add_session_reflection remains the same logic, logs agent ID
    def generate_and_add_session_reflection(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        completed_count: int,
        processed_count: int,
    ):
        duration_minutes = (end - start).total_seconds() / 60
        log.info(f"{self.log_prefix} Retrieving context for session reflection...")
        mem_query = "Summary session activities, task summaries, errors, outcomes, identity statements/revisions."
        try:
            recent_mems, _ = self.memory.retrieve_and_rerank_memories(
                query=mem_query,
                task_description="Session Reflection",
                context="End of work session",
                identity_statement=self.identity_statement,
                n_results=config.MEMORY_COUNT_REFLECTIONS * 2,
                n_final=config.MEMORY_COUNT_REFLECTIONS,
            )
        except Exception as e:
            log.error(
                f"{self.log_prefix} Failed retrieving memories for session reflection: {e}"
            )
            recent_mems = []

        mem_summary_list = []
        for m in recent_mems:
            relative_time = format_relative_time(m["metadata"].get("timestamp"))
            mem_type = m["metadata"].get("type", "mem")
            snippet = m["content"][:100].strip().replace("\n", " ")
            mem_summary_list.append(f"- [{relative_time}] {mem_type}: {snippet}...")
        mem_summary = "\n".join(mem_summary_list) if mem_summary_list else "None"

        prompt_vars = {
            "identity_statement": self.identity_statement,
            "start_iso": start.isoformat(),
            "end_iso": end.isoformat(),
            "duration_minutes": duration_minutes,
            "completed_count": completed_count,
            "processed_count": processed_count,
            "mem_summary": mem_summary,
        }
        prompt = prompts.SESSION_REFLECTION_PROMPT.format(**prompt_vars)
        log.info(
            f"{self.log_prefix} Asking {self.ollama_chat_model} for session reflection..."
        )
        reflection = call_ollama_api(
            prompt, self.ollama_chat_model, self.ollama_base_url, timeout=120
        )
        if reflection and reflection.strip():
            print(
                f"\n--- Session Reflection ({self.agent_id}) ---\n{reflection}\n------"
            )
            # Use the dedicated method to add reflection
            self.add_self_reflection(reflection, "session_reflection")
        else:
            log.warning(f"{self.log_prefix} Failed to generate session reflection.")

    # get_agent_dashboard_state remains the same logic, logs agent ID
    def get_agent_dashboard_state(self) -> Dict[str, Any]:
        log.debug(f"{self.log_prefix} Gathering dashboard state...")
        try:
            # Get task data first
            tasks_structured = self.task_queue.get_all_tasks_structured()

            # Get memory summary (handle potential errors)
            try:
                memory_summary_dict = self.memory.get_memory_summary()
                summary_items = [
                    f"- {m_type}: {count}"
                    for m_type, count in memory_summary_dict.items()
                ]
                memory_summary_str = (
                    (
                        "**Memory Summary (by Type):**\n"
                        + "\n".join(sorted(summary_items))
                    )
                    if summary_items
                    else "No memories found."
                )
            except Exception as mem_err:
                log.error(f"{self.log_prefix} Error getting memory summary: {mem_err}")
                memory_summary_str = f"Error loading memory summary: {mem_err}"

            # Combine completed/failed tasks and sort
            completed_failed_tasks = tasks_structured.get(
                "completed", []
            ) + tasks_structured.get("failed", [])
            completed_failed_tasks.sort(
                key=lambda t: t.get("Completed At")
                or t.get("Failed At", t.get("Created", "0")),
                reverse=True,
            )  # Sort by completion/fail time, fallback to created

            # Combine planning/in_progress
            in_progress_planning_tasks = tasks_structured.get(
                "planning", []
            ) + tasks_structured.get("in_progress", [])
            in_progress_planning_tasks_sorted = sorted(
                in_progress_planning_tasks, key=lambda t: t.get("Created", "")
            )

            return {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "identity_statement": self.identity_statement,
                "pending_tasks": tasks_structured.get("pending", []),
                "in_progress_tasks": in_progress_planning_tasks_sorted,  # Combined list
                "completed_tasks": tasks_structured.get("completed", []),
                "failed_tasks": tasks_structured.get("failed", []),
                "completed_failed_tasks_data": completed_failed_tasks,  # Combined sorted list for dropdown
                "memory_summary": memory_summary_str,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        except Exception as e:
            log.exception(f"{self.log_prefix} Error gathering agent dashboard state")
            return {  # Return structure with error indicators
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "identity_statement": "Error fetching state",
                "pending_tasks": [],
                "in_progress_tasks": [],
                "completed_tasks": [],
                "failed_tasks": [],
                "completed_failed_tasks_data": [],
                "memory_summary": f"Error: {e}",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }

    # get_formatted_memories_for_task remains the same logic, logs agent ID
    def get_formatted_memories_for_task(self, task_id: str) -> List[Dict[str, Any]]:
        if not task_id:
            return []
        log.debug(f"{self.log_prefix} Getting memories for task ID: {task_id[:8]}")
        try:
            memories = self.memory.get_memories_by_metadata(
                filter_dict={"task_id": task_id},
                limit=100,  # Limit results for performance
            )
        except Exception as e:
            log.error(
                f"{self.log_prefix} Failed getting memories for task {task_id}: {e}"
            )
            return [{"Error": f"Failed getting memories: {e}"}]

        formatted = []
        for mem in memories:
            metadata = mem.get("metadata", {})
            content = mem.get("content", "")
            formatted.append(
                {
                    "Timestamp": metadata.get("timestamp", "N/A"),
                    "Type": metadata.get("type", "N/A"),
                    "Content Snippet": (
                        (content[:200] + "..." if len(content) > 200 else content)
                        if content
                        else "N/A"
                    ),
                    "ID": mem.get("id", "N/A"),
                }
            )
        # Sort by timestamp ascending
        return sorted(formatted, key=lambda x: x.get("Timestamp", "0"))

    # get_formatted_general_memories remains the same logic, logs agent ID
    def get_formatted_general_memories(self) -> List[Dict[str, Any]]:
        log.debug(f"{self.log_prefix} Getting general memories...")
        try:
            memories = self.memory.get_general_memories(limit=50)  # Limit results
        except Exception as e:
            log.error(f"{self.log_prefix} Failed getting general memories: {e}")
            return [{"Error": f"Failed getting general memories: {e}"}]

        formatted = []
        for mem in memories:
            metadata = mem.get("metadata", {})
            content = mem.get("content", "")
            formatted.append(
                {
                    "Timestamp": metadata.get("timestamp", "N/A"),
                    "Type": metadata.get("type", "N/A"),
                    "Content Snippet": (
                        (content[:200] + "..." if len(content) > 200 else content)
                        if content
                        else "N/A"
                    ),
                    "ID": mem.get("id", "N/A"),
                }
            )
        # Sort by timestamp descending (most recent first)
        return sorted(formatted, key=lambda x: x.get("Timestamp", "0"), reverse=True)
