# FILE: agent.py
# autonomous_agent/agent.py
import os
import json
import time
import datetime
import uuid
import re
import traceback
import threading
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, deque

# --- Project Imports ---
import config
from data_structures import Task
from task_manager import TaskQueue
from memory import AgentMemory
from tools import load_tools

# Use specific helpers from utils
from utils import (
    call_ollama_api,
    format_relative_time,
    sanitize_and_validate_path,
    list_directory_contents,
    check_ollama_status,
    check_searxng_status,
)
import chromadb
import prompts

# --- Logging Setup ---
import logging

log = logging.getLogger("AGENT")


class AutonomousAgent:
    def __init__(self, memory_collection: Optional[chromadb.Collection] = None):
        log.info("Initializing AutonomousAgent...")
        self.task_queue = TaskQueue()
        if memory_collection is None:
            raise ValueError("Memory collection is required.")
        self.memory = AgentMemory(
            collection=memory_collection,
            ollama_chat_model=config.OLLAMA_CHAT_MODEL,
            ollama_base_url=config.OLLAMA_BASE_URL,
        )
        self.tools = load_tools()
        self.ollama_base_url = config.OLLAMA_BASE_URL
        self.ollama_chat_model = config.OLLAMA_CHAT_MODEL
        self.ollama_embed_model = config.OLLAMA_EMBED_MODEL
        self.session_state_path = config.AGENT_STATE_PATH
        self.qlora_dataset_path = config.QLORA_DATASET_PATH
        self.identity_statement: str = config.INITIAL_IDENTITY_STATEMENT

        self.session_state = {
            "current_task_id": None,
            "current_action_retries": 0,  # Retries for the current *action* attempt
            "last_checkpoint": None,
            "last_web_browse_content": None,
            "identity_statement": self.identity_statement,
            "user_suggestion_move_on_pending": False,
            "last_action_details": deque(maxlen=config.UI_STEP_HISTORY_LENGTH), # Renamed state key
        }
        self.load_session_state()

        self._is_running = threading.Event()
        self._shutdown_request = threading.Event()
        self._agent_thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()

        # --- UI State: Modified labels ---
        self._ui_update_state: Dict[str, Any] = {
            "status": "paused",
            "log": "Agent paused.",
            "current_task_id": self.session_state.get("current_task_id"),
            "current_task_desc": "N/A",
            "current_action_desc": "N/A", # Renamed from current_step_desc
            "current_plan": "N/A", # Intended plan
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
            "action_history": list(self.session_state.get("last_action_details", [])), # Renamed UI key
        }
        # --- End UI State Modification ---

        # Update UI state with initial task details if loaded
        self._refresh_ui_task_details() # Use helper

        if self.identity_statement == config.INITIAL_IDENTITY_STATEMENT:
            log.info("Performing initial identity revision...")
            self._revise_identity_statement("Initialization / Session Start")
            self.save_session_state()
        else:
            log.info("Loaded existing identity statement. Skipping initial revision.")

        log.info("Agent Initialized.")

    def _refresh_ui_task_details(self):
        """Helper to update UI state fields related to the current task."""
        current_task_id = self.session_state.get("current_task_id")
        if current_task_id:
            task = self.task_queue.get_task(current_task_id)
            if task:
                self._ui_update_state["current_task_desc"] = task.description
                plan_status = "(No Plan Yet)"
                action_status = f"Status: {task.status}"
                if task.plan:
                    plan_status = "**Intended Plan (Guidance Only):**\n" + "\n".join(
                        [f"- {s}" for i, s in enumerate(task.plan)] # Simplified display
                    )

                if task.status == "in_progress":
                    action_status = f"Executing actions towards task goal (Attempt {task.reattempt_count + 1})"
                elif task.status == "planning":
                    action_status = f"Planning (Attempt {task.reattempt_count + 1})"
                    plan_status = "(Generating...)"

                self._ui_update_state["current_action_desc"] = action_status # Update renamed field
                self._ui_update_state["current_plan"] = plan_status
            else:
                self._ui_update_state["current_task_desc"] = "Task Not Found"
                self._ui_update_state["current_action_desc"] = "N/A"
                self._ui_update_state["current_plan"] = "N/A"
        else:
            self._ui_update_state["current_task_desc"] = "N/A"
            self._ui_update_state["current_action_desc"] = "N/A"
            self._ui_update_state["current_plan"] = "N/A"

    def _revise_identity_statement(self, reason: str):
        log.info(f"Revising identity statement. Reason: {reason}")
        mem_query = f"Recent task summaries, self-reflections, session reflections, errors, key accomplishments, or notable interactions relevant to understanding my evolution, capabilities, and purpose."
        relevant_memories, _ = self.memory.retrieve_and_rerank_memories(
            query=mem_query,
            task_description="Reflecting on identity",
            context=f"Reason: {reason}",
            identity_statement=self.identity_statement,
            n_results=config.MEMORY_COUNT_IDENTITY_REVISION * 2,
            n_final=config.MEMORY_COUNT_IDENTITY_REVISION,
        )
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
        prompt = prompts.REVISE_IDENTITY_PROMPT.format(
            identity_statement=self.identity_statement,
            reason=reason,
            memory_context=memory_context,
        )
        log.info(f"Asking {self.ollama_chat_model} to revise identity statement...")
        revised_statement_text = call_ollama_api(
            prompt, self.ollama_chat_model, self.ollama_base_url, timeout=150
        )
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
                f"Identity statement revised:\nOld: {old_statement}\nNew: {self.identity_statement}"
            )
            if self._is_running.is_set():
                self.memory.add_memory(
                    f"Identity Statement Updated (Reason: {reason}):\n{self.identity_statement}",
                    {"type": "identity_revision", "reason": reason},
                )
            else:
                log.info("Agent paused, skipping memory add for identity revision.")
            self.save_session_state()
        else:
            log.warning(
                f"Failed to get a valid revised identity statement from LLM (Response: '{revised_statement_text}'). Keeping current."
            )
            if self._is_running.is_set():
                self.memory.add_memory(
                    f"Identity statement revision failed (Reason: {reason}). LLM response insufficient.",
                    {"type": "identity_revision_failed", "reason": reason},
                )
            else:
                log.info(
                    "Agent paused, skipping memory add for failed identity revision."
                )

    def _update_ui_state(self, **kwargs):
        """Updates the internal UI state dictionary."""
        with self._state_lock:
            self._ui_update_state["timestamp"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
            for key, value in kwargs.items():
                self._ui_update_state[key] = value
            # Ensure action history is always updated from session state
            self._ui_update_state["action_history"] = list( # Renamed key
                self.session_state.get("last_action_details", [])
            )
            # Refresh task specific details if needed
            self._refresh_ui_task_details()

    def get_ui_update_state(self) -> Dict[str, Any]:
        with self._state_lock:
            self._refresh_ui_task_details() # Always refresh task details before returning
            return self._ui_update_state.copy()

    def load_session_state(self):
        if os.path.exists(self.session_state_path):
            try:
                with open(self.session_state_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if not content or not content.strip():
                    log.info(f"Session state file '{self.session_state_path}' empty.")
                    return
                loaded = json.loads(content)
                if isinstance(loaded, dict):
                    loaded.setdefault("last_web_browse_content", None)
                    # --- Rename retry counter key ---
                    loaded.setdefault("current_action_retries", loaded.pop("current_task_retries", 0))
                    # --- End Rename ---
                    loaded.setdefault("user_suggestion_move_on_pending", False)
                    loaded.pop("investigation_context", None)
                    loaded_identity = loaded.get(
                        "identity_statement", config.INITIAL_IDENTITY_STATEMENT
                    )
                    if not loaded_identity:
                        loaded_identity = config.INITIAL_IDENTITY_STATEMENT

                    # --- Rename history key ---
                    loaded_history = loaded.get("last_action_details", loaded.get("last_step_details")) # Check old name for compatibility
                    if 'last_step_details' in loaded: del loaded['last_step_details'] # Remove old key if present
                    # --- End Rename ---

                    if isinstance(loaded_history, list):
                        loaded_history = loaded_history[
                            -config.UI_STEP_HISTORY_LENGTH :
                        ]
                        loaded["last_action_details"] = deque(
                            loaded_history, maxlen=config.UI_STEP_HISTORY_LENGTH
                        )
                    else:
                        log.warning(f"Invalid history found. Initializing empty.")
                        loaded["last_action_details"] = deque(
                            maxlen=config.UI_STEP_HISTORY_LENGTH
                        )
                    self.session_state.update(loaded)
                    self.identity_statement = loaded_identity
                    self.session_state["identity_statement"] = self.identity_statement
                    log.info(f"Loaded session state from {self.session_state_path}.")
                    log.info(
                        f"  User Suggestion Pending: {self.session_state.get('user_suggestion_move_on_pending')}"
                    )
                    log.info(
                        f"  Loaded {len(self.session_state['last_action_details'])} action history entries."
                    )
                else:
                    log.warning(
                        f"Session state file '{self.session_state_path}' invalid format."
                    )
            except json.JSONDecodeError as e:
                log.warning(
                    f"Failed loading session state JSON '{self.session_state_path}': {e}."
                )
            except Exception as e:
                log.warning(
                    f"Failed loading session state '{self.session_state_path}': {e}."
                )
        else:
            log.info(
                f"Session state file '{self.session_state_path}' not found. Using defaults."
            )
            self.session_state["identity_statement"] = self.identity_statement
            self.session_state["user_suggestion_move_on_pending"] = False
            self.session_state["last_action_details"] = deque( # Renamed key
                maxlen=config.UI_STEP_HISTORY_LENGTH
            )

    def save_session_state(self):
        try:
            with self._state_lock:
                self.session_state["last_checkpoint"] = datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat()
                self.session_state["identity_statement"] = self.identity_statement
                self.session_state.pop("investigation_context", None)
                state_to_save = self.session_state.copy()
                state_to_save["last_action_details"] = list( # Renamed key
                    self.session_state["last_action_details"]
                )
            with open(self.session_state_path, "w", encoding="utf-8") as f:
                json.dump(state_to_save, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.error(f"Error saving session state to '{self.session_state_path}': {e}")

    # handle_user_suggestion_move_on remains mostly the same, lock scope already reduced
    def handle_user_suggestion_move_on(self) -> str:
        feedback = "Suggestion ignored: Agent not initialized."
        current_task_id = self.session_state.get("current_task_id")
        agent_running = self._is_running.is_set()
        flag_set_successfully = False

        if agent_running:
            if current_task_id:
                task = self.task_queue.get_task(current_task_id)
                # Task status check is simpler now - just needs to be 'in_progress' or 'planning'
                if task and task.status in ["planning", "in_progress"]:
                    with self._state_lock:
                        if self._is_running.is_set():
                            log.info(
                                f"User suggested moving on from task {current_task_id}. Setting flag."
                            )
                            self.session_state["user_suggestion_move_on_pending"] = True
                            flag_set_successfully = True
                        else:
                            log.info(
                                f"Agent paused just before setting move_on flag for task {current_task_id}. Ignoring suggestion."
                            )
                            feedback = f"Suggestion ignored: Agent paused just before flag could be set. (Task: {current_task_id})."

                    if flag_set_successfully:
                        feedback = f"Suggestion noted for current task ({current_task_id}). Agent will consider wrapping up."
                        try:
                            self.save_session_state()
                        except Exception as e:
                            log.error(f"Error saving session state after setting move_on flag: {e}")

                        if self._is_running.is_set():
                            try:
                                self.memory.add_memory(
                                    f"User Suggestion: Consider moving on from task {current_task_id}.",
                                    {
                                        "type": "user_suggestion_move_on",
                                        "task_id": current_task_id,
                                    },
                                )
                            except Exception as e:
                                log.error(f"Error adding move_on suggestion memory after setting flag: {e}")
                        else:
                            log.info("Agent paused after setting move_on flag, skipping memory add.")

                else: # Task not found or not in correct status
                    log.info(f"User suggested moving on, but task {current_task_id} is not active ({task.status if task else 'Not Found'}).")
                    feedback = f"Suggestion ignored: Task {current_task_id} is not currently being executed (Status: {task.status if task else 'Not Found'})."
            else: # No current task
                log.info("User suggested task change, but agent is idle.")
                feedback = "Suggestion ignored: Agent is currently idle (no active task)."
        else: # Agent not running initially
            log.info(f"Agent paused, ignoring suggestion to move on from task {current_task_id or 'N/A'} (flag not set).")
            feedback = f"Suggestion noted, but agent is paused. Flag not set. (Task: {current_task_id or 'N/A'})."

        return feedback

    def create_task(
        self,
        description: str,
        priority: int = 1,
        depends_on: Optional[List[str]] = None,
    ) -> Optional[str]:
        return self.task_queue.add_task(
            Task(description, priority, depends_on=depends_on)
        )

    # get_available_tools_description remains unchanged
    def get_available_tools_description(self) -> str:
        if not self.tools:
            return "No tools available."
        active_tools_desc = []
        for name, tool in self.tools.items():
            is_active = True
            tool_description = tool.description
            if name == "web":
                if not config.SEARXNG_BASE_URL:
                    log.warning(
                        "Web tool 'search' action disabled: SEARXNG_BASE_URL not set."
                    )
                    tool_description += (
                        "\n  *Note: Search functionality requires SEARXNG_BASE_URL.*"
                    )
                if (
                    not config.DOC_ARCHIVE_DB_PATH
                    or not hasattr(self.tools["web"], "doc_archive_collection")
                    or self.tools["web"].doc_archive_collection is None
                ):
                    tool_description += "\n  *Note: Browse with 'query' (focused retrieval) requires Document Archive DB.*"
            if name == "memory" and not self.memory:
                is_active = False
            if name == "file":
                if not config.ARTIFACT_FOLDER:
                    log.warning("File tool disabled: ARTIFACT_FOLDER not set.")
                    is_active = False
                else:
                    # Ensure description matches prompts.py if updated there
                    tool_description = (
                        "Performs file operations within a secure workspace. Use subdirectories for organization of files by project and content category (e.g., 'project_name/content_category/filename.ext'). Requires 'action' parameter. "
                        "Actions: "
                        "'read' (requires 'filename'): Reads text content from a file (use subdirs). Returns content or error with directory listing if not found. "
                        "'write' (requires 'filename', 'content'): Writes text content to a file (use subdirs). Creates directories if needed, by project and content category. **Automatically archives the previous version if the file exists.** Returns success status and paths. "
                        "'list' (optional 'directory'): Lists files/folders in the specified directory (defaults to workspace root). Returns list or error."
                    )
            if name == "status":
                is_active = True
            if is_active:
                active_tools_desc.append(
                    f"- Tool Name: **{name}**\n  Description: {tool_description}"
                )
        if not active_tools_desc:
            return "No tools currently active or available (check configuration)."
        return "\n".join(active_tools_desc)

    # _generate_task_plan remains largely unchanged, plan is still used for context
    def _generate_task_plan(self, task: Task) -> bool:
        """Generates a step-by-step plan for the task using an LLM, considering past lessons."""
        log.info(
            f"Generating execution plan for task {task.id} (Attempt: {task.reattempt_count + 1})..."
        )
        tool_desc = self.get_available_tools_description()

        # Retrieve lessons learned (no change needed here)
        lessons_learned_context = ""
        if task.reattempt_count > 0:
            log.info(f"Retrieving lessons learned for task {task.id} re-plan...")
            lesson_memories = self.memory.get_memories_by_metadata(
                filter_dict={
                    "$and": [{"task_id": task.id}, {"type": "lesson_learned"}]
                },
                limit=config.MEMORY_COUNT_PLANNING,
            )
            if lesson_memories:
                lessons = []
                for mem in sorted(
                    lesson_memories,
                    key=lambda m: m.get("metadata", {}).get("timestamp", "0"),
                    reverse=True,
                ):
                    relative_time = format_relative_time(
                        mem["metadata"].get("timestamp")
                    )
                    lessons.append(f"- [{relative_time}] {mem['content']}")
                lessons_learned_context = (
                    "**Recent Lessons Learned (Consider for this plan!):**\n"
                    + "\n".join(lessons)
                )
                log.info(f"Found {len(lessons)} relevant lessons learned.")
            else:
                log.info("No specific 'lesson_learned' memories found for this task re-plan.")
                lessons_learned_context = "**Note:** This is a re-attempt, but no specific 'lesson_learned' memories were found. Review previous findings carefully."
        else:
            lessons_learned_context = "**Note:** First attempt, no previous lessons learned for this task."

        # Use updated prompt (defined in prompts.py)
        prompt = prompts.GENERATE_TASK_PLAN_PROMPT.format(
            identity_statement=self.identity_statement,
            task_description=task.description,
            tool_desc=tool_desc,
            max_steps=config.MAX_STEPS_GENERAL_THINKING,
            lessons_learned_context=lessons_learned_context,
        )

        log.info(f"Asking {self.ollama_chat_model} to generate plan...")
        llm_response = call_ollama_api(
            prompt, self.ollama_chat_model, self.ollama_base_url, timeout=150
        )

        if not llm_response or not llm_response.strip():
            log.error(f"LLM failed to generate plan for task {task.id}.")
            return False

        # Plan parsing logic remains the same
        plan_steps = []
        try:
            raw_steps = re.findall(r"^\s*\d+\.\s+(.*)", llm_response, re.MULTILINE)
            if not raw_steps:
                lines = llm_response.strip().split("\n")
                raw_steps = [
                    line.strip()
                    for line in lines
                    if line.strip()
                    and not line.strip().startswith(("```", "**", "Example", "Note:", "Plan:", "Here is"))
                ]
                if not raw_steps:
                    raise ValueError("No numbered or simple list found in LLM response.")
                log.warning("Plan parsing used fallback line-by-line method.")

            plan_steps = [step.strip() for step in raw_steps if step.strip()]
            if not plan_steps:
                raise ValueError("Parsed steps list is empty.")

            if len(plan_steps) > config.MAX_STEPS_GENERAL_THINKING:
                log.warning(f"LLM generated {len(plan_steps)} steps, exceeding limit {config.MAX_STEPS_GENERAL_THINKING}. Truncating.")
                plan_steps = plan_steps[: config.MAX_STEPS_GENERAL_THINKING]

            log.info(f"Generated plan for task {task.id} with {len(plan_steps)} steps.")
            for i, step in enumerate(plan_steps):
                if len(step) < 5: log.warning(f"Plan step {i+1} seems very short: '{step}'")

            task.plan = plan_steps
            # Keep existing findings if re-planning, just add marker
            if task.reattempt_count > 0:
                task.cumulative_findings += f"\n--- Generated New Plan (Attempt {task.reattempt_count + 1}) ---\n"
            else:
                task.cumulative_findings = "Plan generated. Starting execution.\n"
            # current_step_index is not relevant for execution anymore, keep at 0 or ignore
            task.current_step_index = 0
            return True

        except Exception as e:
            log.exception(f"Failed to parse LLM plan response for task {task.id}: {e}\nResponse:\n{llm_response}")
            task.plan = None
            return False

    # --- MODIFIED: generate_thinking focuses on overall task, not specific step ---
    def generate_thinking(
        self,
        task: Task,
        tool_results: Optional[Dict[str, Any]] = None,
        user_suggested_move_on: bool = False,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Generates the agent's reasoning and next action based on the overall task goal and current state."""

        # Use the plan for context, but not for step-by-step execution
        plan_context = "(No plan generated)"
        if task.plan:
            plan_context = "**Intended Plan (Guidance Only):**\n" + "\n".join([f"- {s}" for s in task.plan])

        cumulative_findings = (
            task.cumulative_findings if task.cumulative_findings.strip() else "(No findings yet)"
        )

        tool_desc = self.get_available_tools_description()

        # Memory query focuses on the overall task
        memory_query = f"Info relevant to achieving task goal? Task: {task.description}\nConsider overall goal, previous findings, user suggestions, errors, lessons learned, file organization needs."
        if tool_results:
            archived_path = tool_results.get("result", {}).get("archived_filepath")
            archive_info = f" (Archived previous to: {archived_path})" if archived_path else ""
            tool_result_summary = (str(tool_results)[:200] + archive_info) if isinstance(tool_results, dict) else str(tool_results)[:200]
            memory_query += f"\nLast result summary: {tool_result_summary}"

        relevant_memories, separated_suggestion = self.memory.retrieve_and_rerank_memories(
            query=memory_query,
            task_description=f"Working on Task: {task.description}", # Simplified task description
            context=f"Cumulative Findings:\n{cumulative_findings[-500:]}\nIntended Plan:\n{plan_context[:500]}", # Provide plan as context
            identity_statement=self.identity_statement,
            n_results=config.MEMORY_COUNT_GENERAL_THINKING * 2,
            n_final=config.MEMORY_COUNT_GENERAL_THINKING,
        )

        # Format memories (no change needed here)
        user_provided_info_content = []
        other_memories_context_list = []
        for mem in relevant_memories:
            meta = mem.get("metadata", {})
            mem_type = meta.get("type", "N/A")
            relative_time = format_relative_time(meta.get("timestamp"))
            dist_str = f"{mem.get('distance', -1.0):.3f}" if mem.get("distance") is not None else "N/A"
            mem_content = f"[Memory - {relative_time}] (Type: {mem_type}, Dist: {dist_str}, ID: {mem.get('id', 'N/A')}):\n{mem['content']}"
            if mem_type == "user_provided_info_for_task":
                mem_task_context = meta.get("task_id_context")
                if mem_task_context == task.id:
                    user_provided_info_content.append(mem_content)
            else:
                other_memories_context_list.append(mem_content)

        memory_context_str = "\n\n".join(other_memories_context_list) if other_memories_context_list else "No other relevant memories selected."
        user_provided_info_str = "\n---\n".join(user_provided_info_content) if user_provided_info_content else None

        # --- Use MODIFIED thinking prompt (prompts.py) ---
        prompt_vars = {
            "identity_statement": self.identity_statement,
            "task_description": task.description,
            "plan_context": plan_context, # Pass plan context
            "cumulative_findings": cumulative_findings,
            "tool_desc": tool_desc,
            "memory_context_str": memory_context_str,
        }
        prompt_text = prompts.GENERATE_THINKING_PROMPT_BASE_V2.format(**prompt_vars)

        if user_suggested_move_on:
            prompt_text += f"\n\n**USER SUGGESTION PENDING:** Consider wrapping up this task soon."
        if user_provided_info_str:
            prompt_text += f"\n\n**User Provided Information (Consider for next action):**\n{user_provided_info_str}\n"

        if tool_results:
            tool_name = tool_results.get("tool_name", "Unknown")
            tool_action = tool_results.get("action", "unknown")
            result_payload = tool_results.get("result", {})
            archive_path_info = ""
            if isinstance(result_payload, dict):
                archive_path = result_payload.get("archived_filepath")
                if archive_path: archive_path_info = f"\nNote: Previous version archived to '{archive_path}'"
            try:
                result_context = json.dumps(tool_results.get("result", tool_results), indent=2, ensure_ascii=False)
            except TypeError:
                log.warning("Tool result for prompt context was not JSON serializable, using string representation.")
                result_context = str(tool_results.get("result", tool_results))
            prompt_text += f"\n**Results from Last Action:**\nTool: {tool_name} (Action: {tool_action})\nResult:\n```json\n{result_context}\n```{archive_path_info}\n"
        else:
            prompt_text += "\n**Results from Last Action:**\nNone.\n"

        prompt_text += prompts.GENERATE_THINKING_TASK_NOW_PROMPT_V2.format(
            task_reattempt_count=task.reattempt_count + 1,
        )
        # --- End MODIFIED prompt usage ---

        log.info(
            f"Asking {self.ollama_chat_model} for next action (Task {task.id}, Attempt {task.reattempt_count + 1})..."
        )
        llm_response_text = call_ollama_api(
            prompt_text, self.ollama_chat_model, self.ollama_base_url, timeout=180
        )

        if llm_response_text is None:
            log.error(f"Failed to get thinking response from Ollama for task {task.id}.")
            return "LLM communication failed.", {
                "type": "error", "message": "LLM communication failure (thinking).", "subtype": "llm_comm_error",
            }

        # --- Parsing logic (mostly unchanged, but error context adjusted) ---
        try:
            action: Dict[str, Any] = {"type": "unknown"}
            raw_thinking = llm_response_text
            thinking_marker = "THINKING:"
            action_marker = "NEXT_ACTION:"
            tool_marker = "TOOL:"
            params_marker = "PARAMETERS:"
            answer_marker = "ANSWER:"
            reflections_marker = "REFLECTIONS:"
            thinking_start = llm_response_text.find(thinking_marker)
            action_start = llm_response_text.find(action_marker)
            if thinking_start != -1:
                end_think = action_start if action_start > thinking_start else len(llm_response_text)
                raw_thinking = llm_response_text[thinking_start + len(thinking_marker):end_think].strip()
            else:
                log.warning("'THINKING:' marker not found in LLM response.")
                raw_thinking = llm_response_text

            action_type_str = ""
            action_subtype = "unknown_error"
            if action_start != -1:
                end_action = llm_response_text.find("\n", action_start)
                end_action = end_action if end_action != -1 else len(llm_response_text)
                action_type_str = llm_response_text[action_start + len(action_marker):end_action].strip()
            else:
                log.error("'NEXT_ACTION:' marker not found.")
                return raw_thinking, {
                    "type": "error", "message": "Missing NEXT_ACTION marker.", "subtype": "parse_error",
                }

            if "use_tool" in action_type_str:
                action["type"] = "use_tool"
                tool_name = None
                params_json = None
                tool_start = llm_response_text.find(tool_marker, action_start)
                params_start = llm_response_text.find(params_marker, action_start)

                if tool_start != -1:
                    end_tool = llm_response_text.find("\n", tool_start)
                    if params_start > tool_start and (end_tool == -1 or params_start < end_tool):
                        end_tool = params_start
                    end_tool = end_tool if end_tool != -1 else len(llm_response_text)
                    tool_name = llm_response_text[tool_start + len(tool_marker):end_tool].strip().strip('"').strip("'")
                    log.debug(f"Extracted tool name: '{tool_name}'")
                else:
                    return raw_thinking, {"type": "error", "message": "Missing TOOL marker for use_tool action.", "subtype": "parse_error"}

                if params_start != -1:
                    params_str_start = params_start + len(params_marker)
                    end_params = len(llm_response_text)
                    next_marker_pos = -1
                    for marker in [f"\n{answer_marker}", f"\n{reflections_marker}", f"\n{thinking_marker}", f"\n{action_marker}", f"\n{tool_marker}", f"\n{params_marker}"]:
                        pos = llm_response_text.find(marker, params_str_start)
                        if pos != -1 and (next_marker_pos == -1 or pos < next_marker_pos): next_marker_pos = pos
                    if next_marker_pos != -1: end_params = next_marker_pos
                    raw_params = llm_response_text[params_str_start:end_params].strip()
                    raw_params = re.sub(r"^```json\s*", "", raw_params, flags=re.I | re.M)
                    raw_params = re.sub(r"^```\s*", "", raw_params, flags=re.M)
                    raw_params = re.sub(r"\s*```$", "", raw_params, flags=re.M).strip()
                    json_str = raw_params
                    params_json = None
                    log.debug(f"Raw PARAMS string: '{json_str}'")

                    if tool_name == "status" and json_str == "{}": params_json = {}
                    elif json_str:
                        try: params_json = json.loads(json_str)
                        except json.JSONDecodeError as e1:
                            log.warning(f"Direct JSON parse failed: {e1}. Fixing...")
                            fixed_str = json_str.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
                            fixed_str = re.sub(r",\s*([}\]])", r"\1", fixed_str)
                            try: params_json = json.loads(fixed_str)
                            except json.JSONDecodeError as e2:
                                first_brace = raw_params.find("{"); last_brace = raw_params.rfind("}")
                                extracted_str = raw_params[first_brace:last_brace + 1] if first_brace != -1 and last_brace > first_brace else raw_params
                                try:
                                    fixed_extracted_str = extracted_str.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
                                    fixed_extracted_str = re.sub(r",\s*([}\]])", r"\1", fixed_extracted_str)
                                    params_json = json.loads(fixed_extracted_str)
                                except json.JSONDecodeError as e3:
                                    err_msg = f"Invalid JSON in PARAMETERS after all attempts: {e3}. Original: '{raw_params}'"
                                    log.error(err_msg)
                                    return raw_thinking, {"type": "error", "message": err_msg, "subtype": "parse_error"}
                    elif tool_name == "status" and not json_str: params_json = {}
                    else: return raw_thinking, {"type": "error", "message": "Empty PARAMETERS block.", "subtype": "parse_error"}
                elif tool_name != "status": return raw_thinking, {"type": "error", "message": "Missing PARAMETERS marker for use_tool action.", "subtype": "parse_error"}
                else: params_json = {}

                # Parameter Validation (unchanged logic, tool implementation handles internal details)
                if action.get("type") != "error":
                    if not tool_name: action = {"type": "error", "message": "Tool name missing.", "subtype": "parse_error"}
                    elif tool_name not in self.tools: action = {"type": "error", "message": f"Tool '{tool_name}' not available.", "subtype": "invalid_tool"}
                    elif not isinstance(params_json, dict): action = {"type": "error", "message": f"Parsed PARAMETERS not a JSON object.", "subtype": "parse_error"}
                    else:
                        tool_action = params_json.get("action")
                        valid_params = True; err_msg = ""; action_subtype = "invalid_params"
                        if tool_name == "web":
                            if not tool_action or not isinstance(tool_action, str): err_msg = "Missing/invalid 'action' for web tool."; action_subtype = "missing_action"
                            elif tool_action == "search" and ("query" not in params_json or not isinstance(params_json.get("query"), str) or not params_json["query"].strip()): err_msg = "Missing/invalid 'query' for web search."
                            elif tool_action == "browse":
                                if "url" not in params_json or not isinstance(params_json.get("url"), str) or not params_json["url"].strip(): err_msg = "Missing/invalid 'url' for web browse."
                                elif "query" in params_json and (not isinstance(params_json.get("query"), str) or not params_json["query"].strip()): err_msg = "Invalid 'query' for web browse (must be non-empty string if provided)."
                            elif tool_action not in ["search", "browse"]: err_msg = f"Invalid action '{tool_action}' for web tool."; action_subtype = "invalid_action"
                        elif tool_name == "memory":
                            if not tool_action or not isinstance(tool_action, str): err_msg = "Missing/invalid 'action' for memory tool."; action_subtype = "missing_action"
                            elif tool_action == "search" and ("query" not in params_json or not isinstance(params_json.get("query"), str) or not params_json["query"].strip()): err_msg = "Missing/invalid 'query' for memory search."
                            elif tool_action == "write" and ("content" not in params_json or not isinstance(params_json.get("content"), str)): err_msg = "Missing/invalid 'content' (string) for memory write."
                            elif tool_action not in ["search", "write"]: err_msg = f"Invalid action '{tool_action}' for memory tool."; action_subtype = "invalid_action"
                        elif tool_name == "file":
                            if not tool_action or not isinstance(tool_action, str): err_msg = "Missing/invalid 'action' for file tool."; action_subtype = "missing_action"
                            elif tool_action == "read" and ("filename" not in params_json or not isinstance(params_json.get("filename"), str) or not params_json["filename"].strip()): err_msg = "Missing/invalid 'filename' for file read."
                            elif tool_action == "write" and ("filename" not in params_json or not isinstance(params_json.get("filename"), str) or not params_json["filename"].strip() or "content" not in params_json or not isinstance(params_json.get("content"), str)): err_msg = "Missing/invalid 'filename' or 'content' for file write."
                            elif tool_action == "list" and ("directory" in params_json and not isinstance(params_json.get("directory"), str)): err_msg = "Invalid 'directory' for file list."
                            elif tool_action not in ["read", "write", "list"]: err_msg = f"Invalid action '{tool_action}' for file tool."; action_subtype = "invalid_action"
                        elif tool_name == "status":
                            if params_json and params_json != {}: err_msg = "Status tool does not accept parameters."; action_subtype = "invalid_params"
                        else: err_msg = f"Validation not implemented for tool '{tool_name}'."; action_subtype = "internal_error"

                        if err_msg:
                            log.error(f"{err_msg} Tool: {tool_name}, Action: {tool_action}, Params: {params_json}")
                            action = {"type": "error", "message": err_msg, "subtype": action_subtype}
                            valid_params = False
                        if valid_params:
                            action["tool"] = tool_name; action["parameters"] = params_json
                            log.info(f"Parsed action: Use Tool '{tool_name}'" + (f", Action '{tool_action}'" if tool_action else ""))

            elif "final_answer" in action_type_str:
                action["type"] = "final_answer"
                answer = ""
                reflections = ""
                answer_start = llm_response_text.find(answer_marker, action_start)
                reflections_start = llm_response_text.find(reflections_marker, action_start)
                if answer_start != -1:
                    end_answer = reflections_start if reflections_start > answer_start else len(llm_response_text)
                    answer = llm_response_text[answer_start + len(answer_marker):end_answer].strip()
                else:
                    potential_answer_start = llm_response_text.find("\n", action_start) + 1
                    potential_answer_end = reflections_start if reflections_start > potential_answer_start else len(llm_response_text)
                    answer = llm_response_text[potential_answer_start:potential_answer_end].strip() if potential_answer_start > 0 and potential_answer_start < potential_answer_end else ""

                if reflections_start != -1:
                    reflections = llm_response_text[reflections_start + len(reflections_marker):].strip()
                action["answer"] = answer
                action["reflections"] = reflections

                # Simplified check for final answer validity
                if not answer:
                    action = {"type": "error", "message": "LLM chose final_answer but provided no ANSWER.", "subtype": "parse_error"}
                # Removed check based on step index. Agent decides if task is complete.
                elif len(answer) < 50 and re.search(r"\b(error|fail|cannot|unable)\b", answer, re.IGNORECASE) and not user_suggested_move_on:
                    log.warning(f"LLM 'final_answer' looks like giving up: '{answer[:100]}...'")
                    log.info(f"Parsed action: Final Answer (weak).")
                else:
                    log.info(f"Parsed action: Final Answer.")

            elif action_type_str:
                action = {"type": "error", "message": f"Invalid NEXT_ACTION specified: '{action_type_str}'. Expected 'use_tool' or 'final_answer'.", "subtype": "parse_error"}

            if action["type"] == "unknown" and action.get("subtype") != "parse_error":
                action = {"type": "error", "message": "Could not parse action from LLM response.", "subtype": "parse_error"}
            return raw_thinking, action
        except Exception as e:
            log.exception(f"CRITICAL failure parsing LLM thinking response: {e}\nResponse:\n{llm_response_text}")
            return raw_thinking or "Error parsing.", {"type": "error", "message": f"Internal error parsing LLM response: {e}", "subtype": "internal_error"}

    # execute_tool remains unchanged
    def execute_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found", "status": "failed"}
        tool = self.tools[tool_name]
        tool_action = parameters.get("action")
        if tool_name == "status": tool_action = "status_report"
        log_action = tool_action if tool_action else "(implicit)"
        log.info(f"Executing tool '{tool_name}', action '{log_action}' with params: {parameters}")
        try:
            result: Dict[str, Any] = {}
            if tool_name == "memory":
                result = tool.run(parameters, memory_instance=self.memory, identity_statement=self.identity_statement)
            elif tool_name == "status":
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
            elif tool_name in ["web", "file"]: result = tool.run(parameters)
            else: result = tool.run(parameters)

            log.info(f"Tool '{tool_name}' action '{log_action}' finished.")

            # State updates after tool execution
            if tool_name == "web" and tool_action == "browse" and isinstance(result, dict) and result.get("query_mode") is False and "content" in result:
                with self._state_lock: self.session_state["last_web_browse_content"] = result.get("content", "(Empty)")
            elif tool_name == "web" and tool_action == "browse" and isinstance(result, dict) and result.get("query_mode") is True:
                snippets = result.get("retrieved_snippets", []); snippet_count = len(snippets)
                query_used = parameters.get("query", "N/A")
                with self._state_lock: self.session_state["last_web_browse_content"] = f"(Retrieved {snippet_count} snippets from archived page for query: '{query_used}')"

            # Result Processing (Robustness Checks)
            if not isinstance(result, dict):
                log.warning(f"Result from '{tool_name}' ({log_action}) not a dict: {type(result)}")
                return {"tool_name": tool_name, "action": tool_action, "result": {"unexpected_result": str(result)}, "status": "completed_malformed_output"}

            if result.get("error"):
                log.warning(f"Tool '{tool_name}' ({log_action}) reported error: {result['error']}")
                return {"tool_name": tool_name, "action": tool_action, "error": result["error"], "status": "failed", "result": result}

            try: json.dumps(result)
            except TypeError as json_err:
                log.warning(f"Result from '{tool_name}' ({log_action}) not JSON serializable: {json_err}.")
                serializable_result = {}
                for k, v in result.items():
                    try: json.dumps({k: v}); serializable_result[k] = v
                    except TypeError: serializable_result[k] = f"<Unserializable type: {type(v).__name__}> {str(v)[:100]}..."
                return {"tool_name": tool_name, "action": tool_action, "result": serializable_result, "status": "completed_unserializable"}

            # Prepare final return structure
            final_status = result.get("status", "completed")
            result_payload = result.copy()
            result_payload.pop("status", None); result_payload.pop("error", None); result_payload.pop("action", None)

            return {"tool_name": tool_name, "action": tool_action, "status": final_status, "result": result_payload}

        except Exception as e:
            log.exception(f"CRITICAL Error executing tool '{tool_name}' action '{log_action}': {e}")
            return {"tool_name": tool_name, "action": tool_action, "error": f"Tool execution raised unexpected exception: {e}", "status": "failed"}

    # _save_qlora_datapoint remains unchanged
    def _save_qlora_datapoint(
        self, source_type: str, instruction: str, input_context: str, output: str
    ):
        if not output: return
        try:
            datapoint = {"instruction": instruction, "input": input_context, "output": output, "source_type": source_type}
            with open(self.qlora_dataset_path, "a", encoding="utf-8") as f:
                json.dump(datapoint, f, ensure_ascii=False); f.write("\n")
            log.info(f"QLoRA datapoint saved (Source: {source_type})")
        except Exception as e: log.exception(f"Failed to save QLoRA datapoint: {e}")

    # _summarize_and_prune_task_memories remains unchanged
    def _summarize_and_prune_task_memories(self, task: Task):
        if not config.ENABLE_MEMORY_SUMMARIZATION: return
        log.info(f"Summarizing memories and findings for {task.status} task {task.id}...")
        final_summary_text = "No cumulative findings recorded."
        if task.cumulative_findings and task.cumulative_findings.strip():
            trunc_limit = config.CONTEXT_TRUNCATION_LIMIT
            summary_prompt = prompts.SUMMARIZE_TASK_PROMPT.format(task_status=task.status, summary_context=task.cumulative_findings[:trunc_limit])
            log.info(f"Asking {self.ollama_chat_model} to summarize task findings {task.id}...")
            summary_text = call_ollama_api(summary_prompt, self.ollama_chat_model, self.ollama_base_url, timeout=150)
            if summary_text and summary_text.strip():
                final_summary_text = f"Task Summary ({task.status}):\n{summary_text}"
                if task.status == "completed" and not task.result:
                    log.info(f"Setting task {task.id} result from final summary.")
                    task.result = {"answer": summary_text, "steps_taken": -1} # Indicate steps not tracked
                    self.task_queue.update_task(task.id, status=task.status, result=task.result, reflections=task.reflections)
                elif task.status == "failed" and not task.reflections:
                    log.info(f"Setting task {task.id} reflections from final summary.")
                    task.reflections = f"Failure Summary: {summary_text}"
                    self.task_queue.update_task(task.id, status=task.status, result=task.result, reflections=task.reflections)
            else:
                log.warning(f"Failed to generate summary for task findings {task.id}. Using raw findings.")
                final_summary_text = f"Task Findings ({task.status}):\n{task.cumulative_findings[:1000]}..."
        else: log.info(f"No cumulative findings to summarize for task {task.id}.")

        task_memories = self.memory.get_memories_by_metadata(filter_dict={"task_id": task.id})
        summary_metadata = {
            "type": "task_summary", "task_id": task.id, "original_status": task.status,
            "summarized_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "original_memory_count": len(task_memories),
        }
        summary_id = self.memory.add_memory(final_summary_text, summary_metadata)
        if not task_memories:
            log.info(f"No specific action memories found for task {task.id} to potentially prune.")
            return
        log.debug(f"Found {len(task_memories)} action-related memories for task {task.id}.")

        if summary_id and config.DELETE_MEMORIES_AFTER_SUMMARY:
            memory_ids_to_prune = []
            types_to_keep = {"task_summary", "identity_revision", "session_reflection", "user_suggestion_move_on", "task_result", "task_reflection", "lesson_learned", "user_provided_info_for_task"}
            for mem in task_memories:
                meta = mem.get("metadata", {}); mem_type = meta.get("type", "memory")
                if mem_type not in types_to_keep: memory_ids_to_prune.append(mem["id"])
            if memory_ids_to_prune:
                ids_to_delete_final = [mid for mid in memory_ids_to_prune if mid != summary_id]
                if ids_to_delete_final:
                    log.info(f"Summary added ({summary_id}). Pruning {len(ids_to_delete_final)} original action memories for task {task.id}...")
                    deleted = self.memory.delete_memories(ids_to_delete_final)
                    log.info(f"Pruning status for task {task.id}: {deleted}")
                else: log.info(f"No memories eligible for pruning found for task {task.id} after filtering.")
            else: log.info(f"No original action memories found eligible for pruning for task {task.id}.")
        elif summary_id: log.info(f"Summary added ({summary_id}). Deletion of action memories disabled or no memories to prune.")
        else: log.error(f"Failed to add final summary memory for task {task.id}. Pruning skipped.")

    # --- MODIFIED: _reflect_on_error_and_prepare_reattempt uses action context ---
    def _reflect_on_error_and_prepare_reattempt(
        self,
        task: Task,
        error_message: str,
        error_subtype: str,
        failed_action_context: str, # Changed from step index/objective
    ) -> bool:
        """Generates a lesson learned from an error and resets the task for re-planning."""
        log.warning(
            f"Task {task.id} failed action after max retries. Attempting to learn and restart task (Attempt {task.reattempt_count + 1}/{config.TASK_MAX_REATTEMPT})."
        )

        # Plan context is still useful for reflection
        plan_steps_str = "\n".join([f"- {s}" for s in task.plan]) if task.plan else "N/A"

        prompt = prompts.LESSON_LEARNED_PROMPT_V2.format( # Use updated prompt
            task_description=task.description,
            plan_context=plan_steps_str,
            failed_action_context=failed_action_context, # Pass action context
            error_message=error_message,
            error_subtype=error_subtype,
            cumulative_findings=task.cumulative_findings[-2000:],
            identity_statement=self.identity_statement,
        )

        log.info(f"Asking {self.ollama_chat_model} to generate lesson learned for task {task.id} error...")
        lesson_learned_text = call_ollama_api(
            prompt, self.ollama_chat_model, self.ollama_base_url, timeout=120
        )

        if not lesson_learned_text or not lesson_learned_text.strip():
            log.warning("LLM failed to generate a lesson learned. Proceeding with re-attempt anyway.")
            lesson_learned_text = "(LLM failed to generate lesson)"
        else:
            lesson_learned_text = lesson_learned_text.strip()
            log.info(f"Generated Lesson Learned: {lesson_learned_text}")
            self.memory.add_memory(
                content=f"Lesson Learned (Task '{task.id}', Attempt {task.reattempt_count+1} Failed Action): {lesson_learned_text}",
                metadata={
                    "type": "lesson_learned", "task_id": task.id,
                    "failed_action_context": failed_action_context, # Store action context
                    "error_subtype": error_subtype, "error_message": error_message[:200],
                },
            )

        # Reset the task state using the TaskManager method (sets status to planning)
        reset_success = self.task_queue.prepare_task_for_reattempt(task.id, lesson_learned_text)

        if reset_success:
            log.info(f"Task {task.id} successfully prepared for re-attempt {task.reattempt_count + 1} (status: planning).")
            return True
        else:
            log.error(f"Failed to reset task {task.id} state in TaskQueue. Cannot re-attempt.")
            return False

    # --- MODIFIED: _execute_step becomes _perform_action_cycle ---
    def _perform_action_cycle(self, task: Task) -> Dict[str, Any]:
        """Performs one cycle of thinking, acting, and reflecting for the given task."""
        cycle_start_time = time.time()
        action_log = []
        final_answer_text = None
        cycle_status = "processing"  # Status of this *cycle* execution attempt
        task_status_updated_this_cycle = False
        current_retries = self.session_state.get("current_action_retries", 0) # Renamed
        user_suggested_move_on = self.session_state.get("user_suggestion_move_on_pending", False)

        tool_results_for_ui = None
        thinking_to_store = "(No thinking process recorded for this cycle)"
        action_type = "internal" # Default action type
        action_objective = f"Make progress on task: {task.description[:60]}..." # Action objective is broader now

        # Initialize details for history
        action_details_for_history = {
            "task_id": task.id,
            "action_cycle": task.current_step_index + 1, # Use step index as a cycle counter
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

        # --- Start of the Action Cycle ---
        task.current_step_index += 1 # Increment cycle counter
        cycle_num_display = task.current_step_index

        if user_suggested_move_on:
            log.info(f"User suggestion 'move on' is pending for task {task.id}, cycle {cycle_num_display}.")
            with self._state_lock: self.session_state["user_suggestion_move_on_pending"] = False
            self.save_session_state()

        action_log.append(
            f"--- Task '{task.id}' | Action Cycle {cycle_num_display} (Retry {current_retries}/{config.AGENT_MAX_STEP_RETRIES}, Task Attempt {task.reattempt_count + 1}/{config.TASK_MAX_REATTEMPT}) ---"
        )
        action_log.append(f"Task Goal: {task.description}")
        if user_suggested_move_on: action_log.append("[INFO] User suggestion to wrap up considered.")

        log.info(f"Executing action cycle {cycle_num_display} for task {task.id} (Action Retry: {current_retries}, Task Attempt: {task.reattempt_count + 1})")

        # Generate thinking and next action based on overall task state
        raw_thinking, action = self.generate_thinking(
            task=task,
            tool_results=self._ui_update_state.get("last_tool_results"),
            user_suggested_move_on=user_suggested_move_on,
        )
        action_type = action.get("type", "error")
        action_message = action.get("message", "Unknown error")
        action_subtype = action.get("subtype", "unknown_error")
        thinking_to_store = raw_thinking or "Thinking process not extracted."
        action_log.append(f"Thinking:\n{thinking_to_store}")

        action_details_for_history["thinking"] = thinking_to_store
        action_details_for_history["action_type"] = action_type
        action_details_for_history["action_objective"] = f"Cycle {cycle_num_display} thinking towards: {task.description[:60]}..."

        # Log thinking to memory
        self.memory.add_memory(
            f"Action Cycle {cycle_num_display} Thinking (Action: {action_type}, Subtype: {action_subtype}, Attempt {task.reattempt_count+1}):\n{thinking_to_store}",
            {
                "type": "task_thinking", "task_id": task.id, "action_cycle": cycle_num_display,
                "action_type": action_type, "action_subtype": action_subtype,
                "task_attempt": task.reattempt_count + 1,
            },
        )

        cycle_finding = ""

        # --- Process Action or Error ---
        if action_type == "error":
            log.error(f"[ACTION ERROR] LLM Action Error (Task {task.id}, Cycle {cycle_num_display}, Attempt {task.reattempt_count+1}): {action_message} (Subtype: {action_subtype})")
            action_log.append(f"[ERROR] Action Error: {action_message} (Type: {action_subtype})")
            cycle_finding = f"\n--- Action Cycle {cycle_num_display} Error (Action Retry {current_retries+1}, Task Attempt {task.reattempt_count+1}): LLM Action Error - {action_message} (Type: {action_subtype}) ---\n"
            task.cumulative_findings += cycle_finding
            self.memory.add_memory(
                f"Action Cycle {cycle_num_display} Error: {action_message}",
                {"type": "agent_error", "task_id": task.id, "action_cycle": cycle_num_display,
                 "error_subtype": action_subtype, "llm_response_snippet": raw_thinking[:200],
                 "task_attempt": task.reattempt_count + 1},
            )
            action_details_for_history["result_status"] = "error"
            action_details_for_history["result_summary"] = f"LLM Action Error: {action_message} ({action_subtype})"

            # Handle Action Retries / Task Re-attempts for LLM Error
            failed_action_context = f"LLM failed to generate valid action during cycle {cycle_num_display}. Error: {action_message} ({action_subtype})."
            if current_retries < config.AGENT_MAX_STEP_RETRIES:
                self.session_state["current_action_retries"] = current_retries + 1
                log.warning(f"Incrementing action retry count to {current_retries + 1}.")
                cycle_status = "error_retry"
            else:
                if task.reattempt_count < config.TASK_MAX_REATTEMPT:
                    if self._reflect_on_error_and_prepare_reattempt(task, action_message, action_subtype, failed_action_context):
                        cycle_status = "error_reattempting_task"
                        action_log.append(f"[INFO] Max action retries reached. Re-attempting task from beginning (Attempt {task.reattempt_count + 1}).")
                        self.session_state["current_action_retries"] = 0 # Reset action retries for new task attempt
                    else:
                        log.error(f"Failed to prepare task {task.id} for re-attempt. Failing task permanently.")
                        fail_reason = f"Failed during action cycle {cycle_num_display} Attempt {task.reattempt_count+1}. Max action retries ({config.AGENT_MAX_STEP_RETRIES}) + Failed to reset task state."
                        task.reflections = fail_reason
                        task.cumulative_findings += f"\n--- Task Failed: Max action retries & reset failure on cycle {cycle_num_display}. ---\n"
                        self.task_queue.update_task(task.id, "failed", reflections=fail_reason)
                        task.status = "failed"; task_status_updated_this_cycle = True
                        cycle_status = "failed"
                else:
                    log.error(f"Max action retries AND max task reattempts reached for task {task.id}. Failing permanently.")
                    fail_reason = f"Failed during action cycle {cycle_num_display}. Max action retries ({config.AGENT_MAX_STEP_RETRIES}) and max task reattempts ({config.TASK_MAX_REATTEMPT}) reached. Last error: {action_message}"
                    task.reflections = fail_reason
                    task.cumulative_findings += f"\n--- Task Failed Permanently: Max action and task retries reached on cycle {cycle_num_display}. ---\n"
                    self.task_queue.update_task(task.id, "failed", reflections=fail_reason)
                    task.status = "failed"; task_status_updated_this_cycle = True
                    cycle_status = "failed"

        elif action_type == "use_tool":
            tool_name = action.get("tool"); params = action.get("parameters")
            tool_action = params.get("action") if isinstance(params, dict) else None
            if tool_name == "status": tool_action = "status_report"
            log_action_display = tool_action or "(implicit)"
            log.info(f"[ACTION] Cycle {cycle_num_display}: Use Tool '{tool_name}', Action '{log_action_display}' (Attempt {task.reattempt_count+1})")
            action_log.append(f"[ACTION] Use Tool: {tool_name}, Action: {log_action_display}, Params: {json.dumps(params, ensure_ascii=False, indent=2)}")
            action_details_for_history["action_params"] = params

            tool_output = self.execute_tool(tool_name, params)
            tool_results_for_ui = tool_output
            tool_status = tool_output.get("status", "unknown"); tool_error = tool_output.get("error")
            tool_action_from_result = tool_output.get("action", tool_action)
            result_content = tool_output.get("result", {})

            action_details_for_history["result_status"] = tool_status

            # Generate Result Summary (unchanged logic)
            result_display_str = "(Error summarizing result)"
            summary_limit = 500
            try:
                if isinstance(result_content, dict):
                    if "message" in result_content: result_display_str = result_content["message"]
                    elif tool_name == "file" and tool_action_from_result == "write" and "filepath" in result_content:
                        archived_path = result_content.get("archived_filepath")
                        result_display_str = f"Wrote file: {result_content['filepath']}" + (f" (Archived previous: {archived_path})" if archived_path else "")
                    elif tool_name == "file" and tool_action_from_result == "list" and "files" in result_content:
                        result_display_str = f"Listed dir '{result_content.get('directory_path', '?')}': {len(result_content.get('files',[]))} files, {len(result_content.get('directories',[]))} dirs."
                    elif tool_name == "web" and tool_action_from_result == "search" and "results" in result_content: result_display_str = f"Found {len(result_content['results'])} search results."
                    elif tool_name == "web" and tool_action_from_result == "browse" and "query_mode" in result_content and result_content["query_mode"] is True: result_display_str = f"Focused browse on '{result_content.get('url','?')}': Found {len(result_content.get('retrieved_snippets',[]))} snippets for query '{result_content.get('query','?')}'."
                    elif tool_name == "web" and tool_action_from_result == "browse" and "content" in result_content:
                        trunc_info = "(truncated)" if result_content.get("truncated") else ""; result_display_str = f"Browsed '{result_content.get('url','?')}': Content ({result_content.get('content_source', '?')}) length {len(result_content['content'])} {trunc_info}"
                    elif tool_name == "memory" and tool_action_from_result == "search" and "retrieved_memories" in result_content: result_display_str = f"Found {len(result_content['retrieved_memories'])} memories."
                    elif tool_name == "status" and "report_content" in result_content: result_display_str = "Generated status report."
                    elif "content" in result_content: content_str = str(result_content["content"]); result_display_str = content_str[:300] + ("..." if len(content_str) > 300 else "")
                    else: result_display_str = json.dumps(result_content, indent=None, ensure_ascii=False)
                else: result_display_str = str(result_content)
                if len(result_display_str) > 1000: result_display_str = result_display_str[:1000] + "..."
            except Exception as summ_e: log.warning(f"Error summarizing tool result: {summ_e}"); result_display_str = "(Error summarizing result)"
            result_summary_for_history = result_display_str[:summary_limit] + ("..." if len(result_display_str) > summary_limit else "")
            if tool_error: result_summary_for_history = f"Error: {tool_error}\n---\n{result_summary_for_history}"[:summary_limit] + ("..." if len(result_summary_for_history) > summary_limit else "")
            action_details_for_history["result_summary"] = result_summary_for_history
            # End Summary Generation

            cycle_finding = f"\n--- Action Cycle {cycle_num_display} (Tool: {tool_name}/{tool_action_from_result or 'action'}, Attempt {task.reattempt_count+1}) ---\nStatus: {tool_status}\nResult Summary: {result_summary_for_history}\n---\n"
            task.cumulative_findings += cycle_finding

            # Add memory record of the tool execution
            mem_params = json.dumps(params)[:500] + ("..." if len(json.dumps(params)) > 500 else "")
            mem_summary = result_summary_for_history
            self.memory.add_memory(
                f"Action Cycle {cycle_num_display} Tool Result (Status: {tool_status}, Attempt {task.reattempt_count+1}):\nAction: {tool_name}/{tool_action_from_result}\nParams: {mem_params}\nSummary: {mem_summary}",
                {"type": "tool_result", "task_id": task.id, "action_cycle": cycle_num_display, "tool_name": tool_name,
                 "action": tool_action_from_result, "result_status": tool_status, "params_snippet": mem_params,
                 "task_attempt": task.reattempt_count + 1},
            )

            # Handle Tool Result Status
            if tool_status in ["success", "completed", "completed_malformed_output", "completed_unserializable", "completed_with_issue", "success_archived"]:
                if tool_status not in ["success", "completed", "success_archived"]: log.warning(f"Tool '{tool_name}' action '{tool_action_from_result}' finished cycle {cycle_num_display} with non-ideal status: {tool_status}.")
                else: action_log.append(f"[INFO] Tool '{tool_name}' action '{tool_action_from_result}' completed cycle {cycle_num_display} successfully.")
                cycle_status = "processing" # Ready for next cycle
                self.session_state["current_action_retries"] = 0 # Reset action retries on success
            elif tool_status == "failed":
                log.warning(f"Tool '{tool_name}' action '{tool_action_from_result}' failed cycle {cycle_num_display}: {tool_error}")
                action_log.append(f"[ERROR] Tool '{tool_name}' ({tool_action_from_result}) failed: {tool_error}")
                error_finding = f"\n--- Action Cycle {cycle_num_display} Error (Action Retry {current_retries+1}, Task Attempt {task.reattempt_count+1}): Tool Error - Tool={tool_name}, Action={tool_action_from_result}, Error={tool_error} ---\n"
                task.cumulative_findings += error_finding
                self.memory.add_memory(
                    f"Action Cycle {cycle_num_display} Tool Error: {tool_error}",
                    {"type": "agent_error", "task_id": task.id, "action_cycle": cycle_num_display,
                     "error_subtype": "tool_execution_error", "tool_name": tool_name, "action": tool_action_from_result,
                     "error_details": str(tool_error)[:200], "task_attempt": task.reattempt_count + 1},
                )
                # Handle Action Retries / Task Re-attempts for Tool Failure
                failed_action_context = f"Tool '{tool_name}' (Action: {tool_action_from_result}) failed during cycle {cycle_num_display}. Error: {tool_error}."
                if current_retries < config.AGENT_MAX_STEP_RETRIES:
                    self.session_state["current_action_retries"] = current_retries + 1
                    log.warning(f"Incrementing action retry count to {current_retries + 1}.")
                    cycle_status = "error_retry"
                else:
                    if task.reattempt_count < config.TASK_MAX_REATTEMPT:
                        if self._reflect_on_error_and_prepare_reattempt(task, tool_error or "Tool Failed", "tool_execution_error", failed_action_context):
                            cycle_status = "error_reattempting_task"
                            action_log.append(f"[INFO] Max action retries reached. Re-attempting task from beginning (Attempt {task.reattempt_count + 1}).")
                            self.session_state["current_action_retries"] = 0
                        else:
                            log.error(f"Failed to prepare task {task.id} for re-attempt. Failing task permanently.")
                            fail_reason = f"Failed during action cycle {cycle_num_display} Attempt {task.reattempt_count+1}. Max action retries + Failed to reset task state."
                            task.reflections = fail_reason
                            task.cumulative_findings += f"\n--- Task Failed: Max action retries & reset failure on cycle {cycle_num_display}. ---\n"
                            self.task_queue.update_task(task.id, "failed", reflections=fail_reason)
                            task.status = "failed"; task_status_updated_this_cycle = True
                            cycle_status = "failed"
                    else:
                        log.error(f"Max action retries AND max task reattempts reached for task {task.id}. Failing permanently.")
                        fail_reason = f"Failed during action cycle {cycle_num_display}. Max action retries and max task reattempts reached. Last error: {tool_error}"
                        task.reflections = fail_reason
                        task.cumulative_findings += f"\n--- Task Failed Permanently: Max action and task retries reached on cycle {cycle_num_display}. ---\n"
                        self.task_queue.update_task(task.id, "failed", reflections=fail_reason)
                        task.status = "failed"; task_status_updated_this_cycle = True
                        cycle_status = "failed"
            else: # Unknown tool status
                log.error(f"Tool '{tool_name}' action '{tool_action_from_result}' returned UNKNOWN status '{tool_status}' in cycle {cycle_num_display}. Failing task.")
                action_log.append(f"[CRITICAL] Tool '{tool_name}' ({tool_action_from_result}) unknown status: {tool_status}")
                fail_reason = f"Failed during action cycle {cycle_num_display} Attempt {task.reattempt_count+1} due to unknown tool status '{tool_status}'."
                task.reflections = fail_reason
                task.cumulative_findings += f"\n--- Task Failed: Unknown tool status '{tool_status}' in cycle {cycle_num_display}. ---\n"
                self.task_queue.update_task(task.id, "failed", reflections=fail_reason)
                task.status = "failed"; task_status_updated_this_cycle = True
                cycle_status = "failed"

        elif action_type == "final_answer":
            log.info(f"[ACTION] Cycle {cycle_num_display}: Provide Final Answer (Attempt {task.reattempt_count+1}).")
            action_log.append("[ACTION] Provide Final Answer.")
            answer = action.get("answer", "").strip(); reflections = action.get("reflections", "").strip()
            final_answer_text = answer

            action_details_for_history["result_status"] = "completed"
            action_details_for_history["result_summary"] = f"Final Answer Provided:\n{answer[:500]}" + ("..." if len(answer) > 500 else "")

            print("\n" + "="*15 + f" FINAL ANSWER (Task {task.id}) " + "="*15 + f"\n{answer}\n" + "="*(34 + len(str(task.id))) + "\n")
            cycle_finding = f"\n--- Action Cycle {cycle_num_display}: Final Answer Provided (Attempt {task.reattempt_count+1}) ---\n{answer[:500]}...\n---\n"
            task.cumulative_findings += cycle_finding
            result_payload = {"answer": answer, "action_cycles": cycle_num_display, "task_attempts": task.reattempt_count + 1}
            task.result = result_payload; task.reflections = reflections
            task.completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            # Don't increment step index here, cycle already incremented
            self.task_queue.update_task(task.id, "completed", result=result_payload, reflections=reflections)
            task.status = "completed"; task_status_updated_this_cycle = True
            cycle_status = "completed"

            self._save_qlora_datapoint(
                source_type="task_completion",
                instruction="Given the overall task and cumulative findings, provide the final answer.",
                input_context=f"Overall Task: {task.description}\n\nCumulative Findings:\n{task.cumulative_findings}",
                output=answer,
            )
            self.memory.add_memory(
                f"Final Answer (Action Cycle {cycle_num_display}, Attempt {task.reattempt_count+1}):\n{answer}",
                {"type": "task_result", "task_id": task.id, "final_cycle": cycle_num_display},
            )
            if reflections:
                self.memory.add_memory(
                    f"Final Reflections (Action Cycle {cycle_num_display}):\n{reflections}",
                    {"type": "task_reflection", "task_id": task.id, "final_cycle": cycle_num_display},
                )

            # Summarize after completion
            self._summarize_and_prune_task_memories(task)

        # --- Update State after action cycle ---
        cycle_duration = time.time() - cycle_start_time
        action_log.append(f"Action Cycle {cycle_num_display} duration: {cycle_duration:.2f}s. Cycle Outcome: {cycle_status}")

        # Update action history
        action_details_for_history["log_snippet"] = "\n".join(action_log[-min(5, len(action_log)):])
        with self._state_lock:
            if action_log:
                self.session_state["last_action_details"].append(action_details_for_history)
                self._ui_update_state["action_history"] = list(self.session_state["last_action_details"])
            else: log.warning("Action log was empty, skipping adding details to history.")

        # Clear session state only if task is *permanently* completed or failed
        if cycle_status in ["completed", "failed"]:
            log.info(f"Task {task.id} finished permanently with status: {cycle_status}")
            final_task_obj = self.task_queue.get_task(task.id)
            if final_task_obj: self._handle_task_completion_or_failure(final_task_obj)
            else: log.error(f"Could not retrieve final task object for {task.id}.")
            self.session_state["current_task_id"] = None
            self.session_state["current_action_retries"] = 0 # Renamed
            # User suggestion flag already reset
        elif cycle_status == "error_reattempting_task":
            # Task ID remains, action retries reset earlier by _reflect...
            pass
        elif cycle_status == "error_retry":
            # Task ID remains, action retries incremented earlier
            pass
        elif cycle_status == "processing":
            # Task ID remains, action retries reset earlier
            pass

        self.save_session_state()

        # --- Prepare UI Update Data ---
        final_task_status_obj = self.task_queue.get_task(task.id)
        final_task_status_for_ui = final_task_status_obj.status if final_task_status_obj else task.status
        dependent_tasks_list = []
        current_task_id_for_ui = self.session_state.get("current_task_id")

        self._refresh_ui_task_details() # Updates internal _ui_update_state

        if current_task_id_for_ui:
            dependent_tasks = self.task_queue.get_dependent_tasks(current_task_id_for_ui)
            dependent_tasks_list = [{"id": dt.id, "description": dt.description} for dt in dependent_tasks]

        # Query memories relevant to the *outcome* of the cycle for UI display
        ui_memory_query = f"Memories relevant to outcome '{cycle_status}' of action cycle {cycle_num_display} for task {task.id} (Attempt {task.reattempt_count+1}). Last Action: {action_type}."
        recent_memories_for_ui, _ = self.memory.retrieve_and_rerank_memories(
            query=ui_memory_query, task_description=task.description,
            context=task.cumulative_findings[-1000:], identity_statement=self.identity_statement,
            n_final=5, # Limit for UI
        )

        # Update internal UI state dictionary
        self._update_ui_state(
            status=final_task_status_for_ui, log="\n".join(action_log),
            thinking=thinking_to_store, dependent_tasks=dependent_tasks_list,
            last_action_type=action_type, last_tool_results=tool_results_for_ui,
            recent_memories=recent_memories_for_ui,
            last_web_content=self.session_state.get("last_web_browse_content", "(No recent web browse)"),
            final_answer=final_answer_text,
            # action_history is already updated via session_state and _update_ui_state
        )

        # Return the latest full UI state
        return self.get_ui_update_state()
    # --- END MODIFIED _perform_action_cycle ---

    # _handle_task_completion_or_failure remains unchanged
    def _handle_task_completion_or_failure(self, task: Task):
        log.info(f"Handling completion/failure for task {task.id} (Status: {task.status})")
        if self._is_running.is_set():
            reason = f"{task.status.capitalize()} Task: {task.description[:50]}..."
            self._revise_identity_statement(reason)
        else:
            log.info(f"Agent paused, skipping identity revision after {task.status} task.")

    # --- MODIFIED: process_one_step uses new execution cycle ---
    def process_one_step(self) -> Dict[str, Any]:
        """Processes one planning or execution cycle for the current task."""
        log.info("Processing one agent cycle...")
        cycle_result_log = ["Attempting one cycle..."]
        default_state = self.get_ui_update_state()
        default_state.update({"status": "idle", "log": "Idle.", "current_task_id": None})
        self._update_ui_state(**default_state)

        try:
            task_id_to_process = self.session_state.get("current_task_id")
            task: Optional[Task] = None

            # --- 1. Check if already working on a task ---
            if task_id_to_process:
                task = self.task_queue.get_task(task_id_to_process)
                if not task or task.status not in ["planning", "in_progress", "pending"]:
                    log.warning(f"Task {task_id_to_process} from state invalid/finished (Status: {task.status if task else 'Not Found'}). Resetting session.")
                    cycle_result_log.append(f"Task {task_id_to_process} invalid/finished.")
                    task = None
                    self.session_state["current_task_id"] = None
                    self.session_state["current_action_retries"] = 0 # Renamed
                    with self._state_lock: self.session_state["user_suggestion_move_on_pending"] = False
                    self._update_ui_state(**default_state)
                    self.save_session_state()

            # --- 2. If no valid current task, get the next one ---
            if not task:
                while True:
                    task = self.task_queue.get_next_task()
                    if not task:
                        log.info("No runnable tasks found.")
                        cycle_result_log.append("No runnable tasks found.")
                        if not self.session_state.get("current_task_id"):
                            log.info("Agent idle, considering generating tasks...")
                            generated_desc = self.generate_new_tasks(max_new_tasks=3, trigger_context="idle")
                            if generated_desc: cycle_result_log.append(f"Generated idle task: {generated_desc[:60]}...")
                            else: cycle_result_log.append("No new idle tasks generated.")
                        state_to_update = default_state.copy()
                        state_to_update["log"] = "\n".join(cycle_result_log)
                        self._update_ui_state(**state_to_update)
                        return self.get_ui_update_state()
                    break # Found a valid task

                # Start the valid task
                log.info(f"Starting task: {task.id} - '{task.description[:60]}...' (Attempt {task.reattempt_count + 1})")
                cycle_result_log.append(f"Starting task {task.id}: {task.description[:60]}... (Attempt {task.reattempt_count + 1})")
                self.task_queue.update_task(task.id, "planning")
                task.status = "planning"
                self.session_state["current_task_id"] = task.id
                if task.reattempt_count == 0:
                    task.plan = None; task.current_step_index = 0; task.cumulative_findings = "" # Reset step index/cycle counter

                self.session_state["current_action_retries"] = 0 # Reset ACTION retries
                with self._state_lock: self.session_state["user_suggestion_move_on_pending"] = False
                self.save_session_state()
                self._refresh_ui_task_details()
                self._update_ui_state(status=task.status, log="\n".join(cycle_result_log), thinking="(Generating plan...)")
                # Proceed to planning phase below

            # --- 3. Handle task based on its status ---
            if task and task.status == "planning":
                log.info(f"Task {task.id} is in planning state (Attempt {task.reattempt_count + 1}). Generating plan...")
                cycle_result_log.append(f"Generating plan for task {task.id} (Attempt {task.reattempt_count + 1})...")
                plan_success = self._generate_task_plan(task)

                if plan_success:
                    log.info(f"Plan generated successfully for task {task.id}. Moving to in_progress.")
                    cycle_result_log.append("Plan generated successfully.")
                    self.task_queue.update_task(task.id, "in_progress")
                    task.status = "in_progress"
                    task.current_step_index = 0 # Reset cycle counter before execution starts
                    self.task_queue.save_queue()
                    self.session_state["current_action_retries"] = 0 # Reset action retries
                    self.save_session_state()
                    self._refresh_ui_task_details()
                    self._update_ui_state(status=task.status, log="\n".join(cycle_result_log), thinking="(Plan generated, starting execution...)")
                    return self.get_ui_update_state()
                else: # Plan generation failed
                    log.error(f"Failed to generate plan for task {task.id} (Attempt {task.reattempt_count + 1}).")
                    cycle_result_log.append("ERROR: Failed to generate plan.")
                    fail_reason = f"Failed during planning phase (Attempt {task.reattempt_count + 1})."
                    if task.reattempt_count < config.TASK_MAX_REATTEMPT:
                        log.warning(f"Planning failed, but task reattempts remain ({task.reattempt_count + 1}/{config.TASK_MAX_REATTEMPT}). Reflecting and retrying planning.")
                        cycle_result_log.append(f"Planning failed. Attempting task re-plan (Attempt {task.reattempt_count + 2}).")
                        # Generate lesson learned about planning failure and reset state
                        if self._reflect_on_error_and_prepare_reattempt(
                            task, "Failed to generate a valid execution plan.", "planning_error", "Planning Phase", # Use general context
                        ):
                            # State already reset by _reflect... method, status is 'planning'
                            self.session_state["current_action_retries"] = 0 # Reset action retries
                            self.save_session_state()
                            self._refresh_ui_task_details()
                            self._update_ui_state(status="planning", log="\n".join(cycle_result_log), thinking="(Re-planning task after previous planning failure...)")
                            return self.get_ui_update_state() # Next loop will attempt planning again
                        else:
                            fail_reason += " Additionally failed to reset task state for reattempt."
                            log.error(f"Also failed to reset task {task.id} state. Failing permanently.")
                            # Fall through to permanent failure logic

                    # If planning fails AND max task reattempts reached OR reset fails
                    task.reflections = fail_reason
                    self.task_queue.update_task(task.id, "failed", reflections=fail_reason)
                    failed_task_obj = self.task_queue.get_task(task.id)
                    if failed_task_obj: self._handle_task_completion_or_failure(failed_task_obj)
                    else: log.error(f"Could not retrieve final task object for {task.id} after planning failure.")

                    self.session_state["current_task_id"] = None
                    self.session_state["current_action_retries"] = 0 # Renamed
                    with self._state_lock: self.session_state["user_suggestion_move_on_pending"] = False
                    self.save_session_state()
                    state_to_update = default_state.copy()
                    state_to_update["log"] = "\n".join(cycle_result_log); state_to_update["status"] = "idle"
                    self._update_ui_state(**state_to_update)
                    return self.get_ui_update_state()

            elif task and task.status == "in_progress":
                # Task is planned and ready for execution cycles
                # Call the renamed execution cycle function
                cycle_result = self._perform_action_cycle(task)
                # This function now handles internal state updates and returns the full UI state
                return cycle_result

            elif task: # Task exists but status is unexpected
                log.warning(f"Task {task.id} found but not in expected state (Status: {task.status}). Resetting session.")
                cycle_result_log.append(f"Task {task.id} state unexpected: {task.status}. Resetting session.")
                self.session_state["current_task_id"] = None
                self.session_state["current_action_retries"] = 0 # Renamed
                with self._state_lock: self.session_state["user_suggestion_move_on_pending"] = False
                self.save_session_state()
                state_to_update = default_state.copy()
                state_to_update["log"] = "\n".join(cycle_result_log)
                self._update_ui_state(**state_to_update)
                return self.get_ui_update_state()
            else: # Should be unreachable
                log.error("Task became None unexpectedly during cycle processing.")
                cycle_result_log.append("[CRITICAL ERROR] Task object lost.")
                self._update_ui_state(status="error", log="\n".join(cycle_result_log))
                return self.get_ui_update_state()

        except Exception as e:
            log.exception("CRITICAL Error during process_one_step (cycle) loop")
            cycle_result_log.append(f"[CRITICAL ERROR]: {traceback.format_exc()}")
            current_task_id = self.session_state.get("current_task_id")
            if current_task_id:
                log.error(f"Failing task {current_task_id} due to critical loop error.")
                fail_reason = f"Critical loop error: {e}"
                task_to_fail = self.task_queue.get_task(current_task_id)
                if task_to_fail:
                    task_to_fail.reflections = (task_to_fail.reflections or "") + f"\nLoop Error: {fail_reason}"
                    self.task_queue.update_task(current_task_id, "failed", reflections=task_to_fail.reflections)
                    self._handle_task_completion_or_failure(task_to_fail)
                else: log.error(f"Could not find task {current_task_id} to mark as failed after loop error.")

                self.session_state["current_task_id"] = None
                self.session_state["current_action_retries"] = 0 # Renamed
                with self._state_lock: self.session_state["user_suggestion_move_on_pending"] = False
                self.save_session_state()

            state_to_update = default_state.copy()
            state_to_update["status"] = "critical_error"; state_to_update["log"] = "\n".join(cycle_result_log)
            self._update_ui_state(**state_to_update)
            return self.get_ui_update_state()
    # --- END MODIFIED process_one_step ---

    # generate_new_tasks remains unchanged
    def generate_new_tasks(
        self, max_new_tasks: int = 3, last_user_message: Optional[str] = None,
        last_assistant_response: Optional[str] = None, trigger_context: str = "unknown",
    ) -> Optional[str]:
        log.info(f"\n--- Attempting to Generate New Tasks (Trigger: {trigger_context}) ---")
        memory_is_empty = False; memory_count = 0
        try: memory_count = self.memory.collection.count(); memory_is_empty = memory_count == 0
        except Exception as e: log.error(f"Failed get memory count: {e}. Assuming not empty."); memory_is_empty = False
        use_initial_creative_prompt = memory_is_empty and trigger_context == "idle"
        max_tasks_to_generate = config.INITIAL_NEW_TASK_N if use_initial_creative_prompt else max_new_tasks
        prompt_template = prompts.INITIAL_CREATIVE_TASK_GENERATION_PROMPT if use_initial_creative_prompt else prompts.GENERATE_NEW_TASKS_PROMPT
        log.info(f"Using {'Initial Creative' if use_initial_creative_prompt else 'Standard'} Task Gen Prompt. Max tasks: {max_tasks_to_generate}")
        prompt_vars = {}
        if use_initial_creative_prompt:
            prompt_vars = {"identity_statement": self.identity_statement, "tool_desc": self.get_available_tools_description(), "max_new_tasks": max_tasks_to_generate}
            context_query = "Initial run with empty memory. Generate creative starting tasks."
        else:
            context_source = ""; context_query = ""; mem_query = ""; critical_evaluation_instruction = ""
            if trigger_context == "chat" and last_user_message and last_assistant_response:
                context_source = "last chat interaction"
                context_query = f"Last User: {last_user_message}\nLast Assistant: {last_assistant_response}"
                mem_query = f"Context relevant to last chat: {last_user_message}"
                critical_evaluation_instruction = "\n**Critically Evaluate Need:** Based *specifically* on the **Last Interaction**, is a background task *truly necessary*? Output `[]` if not."
            else:
                context_source = "general agent state"
                context_query = "General status. Consider logical follow-up/exploration based on completed tasks, idle state, and my identity."
                mem_query = "Recent activities, conclusions, errors, reflections, summaries, identity revisions."
                critical_evaluation_instruction = "\n**Critically Evaluate Need:** Are new tasks genuinely needed for exploration/follow-up, consistent with identity? Output `[]` if not."
            log.info(f"Retrieving context for task generation (Source: {context_source})...")
            recent_mems, _ = self.memory.retrieve_and_rerank_memories(
                query=mem_query, task_description="Task Generation Context", context=context_query,
                identity_statement=self.identity_statement, n_results=config.MEMORY_COUNT_NEW_TASKS * 2, n_final=config.MEMORY_COUNT_NEW_TASKS,
            )
            mem_summary_list = []
            for m in recent_mems:
                relative_time = format_relative_time(m["metadata"].get("timestamp")); mem_type = m["metadata"].get("type", "mem")
                snippet = m["content"][:150].strip().replace("\n", " "); mem_summary_list.append(f"- [{relative_time}] {mem_type}: {snippet}...")
            mem_summary = "\n".join(mem_summary_list) if mem_summary_list else "None"
            existing_tasks_info = [{"id": t.id, "description": t.description, "status": t.status} for t in self.task_queue.tasks.values()]
            active_tasks_summary = "\n".join([f"- ID: {t['id']} (Status: {t['status']}) Desc: {t['description'][:100]}..." for t in existing_tasks_info if t["status"] in ["pending", "planning", "in_progress"]]) or "None"
            completed_failed_summary = "\n".join([f"- ID: {t['id']} (Status: {t['status']}) Desc: {t['description'][:100]}..." for t in existing_tasks_info if t["status"] in ["completed", "failed"]])[-1000:]
            prompt_vars = {
                "identity_statement": self.identity_statement, "context_query": context_query, "mem_summary": mem_summary,
                "active_tasks_summary": active_tasks_summary, "completed_failed_summary": completed_failed_summary,
                "critical_evaluation_instruction": critical_evaluation_instruction, "max_new_tasks": max_tasks_to_generate,
            }
        if use_initial_creative_prompt:
            prompt_vars.setdefault("context_query", "N/A (Initial Run)"); prompt_vars.setdefault("mem_summary", "None (Initial Run)")
            prompt_vars.setdefault("active_tasks_summary", "None (Initial Run)"); prompt_vars.setdefault("completed_failed_summary", "None (Initial Run)")
            prompt_vars.setdefault("critical_evaluation_instruction", "N/A (Initial Run)")

        prompt = prompt_template.format(**prompt_vars)
        log.info(f"Asking {self.ollama_chat_model} to generate up to {max_tasks_to_generate} new tasks...")
        llm_response = call_ollama_api(prompt, self.ollama_chat_model, self.ollama_base_url, timeout=180)
        if not llm_response: log.error("LLM failed task gen."); return None

        first_task_desc_added = None; new_tasks_added = 0
        try:
            llm_response = re.sub(r"^```json\s*", "", llm_response, flags=re.I | re.M)
            llm_response = re.sub(r"\s*```$", "", llm_response, flags=re.M).strip()
            list_start = llm_response.find("["); list_end = llm_response.rfind("]")
            if list_start == -1 or list_end == -1 or list_end < list_start:
                if "no new tasks" in llm_response.lower() or llm_response.strip() == "[]": log.info("LLM explicitly stated no new tasks needed."); return None
                else:
                    json_objects = re.findall(r'\{\s*".*?\}\s*', llm_response, re.DOTALL)
                    if json_objects: log.warning("LLM response for tasks was not a valid JSON list, but found JSON objects. Attempting to parse."); json_str = f"[{','.join(json_objects)}]"
                    else: raise json.JSONDecodeError(f"JSON list '[]' not found and no JSON objects detected: {llm_response[:300]}...", llm_response, 0)
            else: json_str = llm_response[list_start : list_end + 1]
            suggested_tasks = json.loads(json_str)
            if not isinstance(suggested_tasks, list): log.warning(f"LLM task gen result was not a list: {type(suggested_tasks)}"); return None
            if not suggested_tasks: log.info("LLM suggested no new tasks (empty list parsed)."); return None
            log.info(f"LLM suggested {len(suggested_tasks)} tasks. Validating...")
            current_task_ids_in_batch = set()
            existing_tasks_info = [{"id": t.id, "description": t.description, "status": t.status} for t in self.task_queue.tasks.values()]
            active_task_descriptions = {t["description"].strip().lower() for t in existing_tasks_info if t["status"] in ["pending", "planning", "in_progress"]}
            valid_existing_task_ids = set(t["id"] for t in existing_tasks_info)
            for task_data in suggested_tasks:
                if not isinstance(task_data, dict): log.warning(f"Skipping non-dict item in suggested tasks: {task_data}"); continue
                description = task_data.get("description")
                if not description or not isinstance(description, str) or not description.strip(): log.warning(f"Skipping task with invalid/missing description: {task_data}"); continue
                description = description.strip()
                if description.lower() in active_task_descriptions: log.warning(f"Skipping duplicate active task: '{description[:80]}...'"); continue
                priority = task_data.get("priority", 3 if use_initial_creative_prompt else 5)
                try: priority = max(1, min(10, int(priority)))
                except (ValueError, TypeError): priority = 3 if use_initial_creative_prompt else 5
                dependencies_raw = task_data.get("depends_on"); validated_dependencies = []
                if isinstance(dependencies_raw, list):
                    for dep_id in dependencies_raw:
                        dep_id_str = str(dep_id).strip()
                        if dep_id_str in valid_existing_task_ids or dep_id_str in current_task_ids_in_batch: validated_dependencies.append(dep_id_str)
                        else: log.warning(f"Dependency '{dep_id_str}' for new task '{description[:50]}...' not found in existing tasks or this batch. Ignoring dependency.")
                elif dependencies_raw is not None: log.warning(f"Invalid 'depends_on' format for task '{description[:50]}...': {dependencies_raw}. Expected list.")
                new_task = Task(description, priority, depends_on=validated_dependencies or None)
                new_task_id = self.task_queue.add_task(new_task)
                if new_task_id:
                    if new_tasks_added == 0: first_task_desc_added = description
                    new_tasks_added += 1
                    active_task_descriptions.add(description.lower()); valid_existing_task_ids.add(new_task_id); current_task_ids_in_batch.add(new_task_id)
                    log.info(f"Added Task {new_task_id}: '{description[:60]}...' (Prio: {priority}, Depends: {validated_dependencies})")
                if new_tasks_added >= max_tasks_to_generate: log.info(f"Reached max task generation limit ({max_tasks_to_generate})."); break
        except json.JSONDecodeError as e: log.error(f"Failed JSON parse task gen: {e}\nLLM Resp:\n{llm_response}\n---"); return None
        except Exception as e: log.exception(f"Unexpected error during task generation parsing/validation: {e}"); return None
        log.info(f"Finished Task Generation: Added {new_tasks_added} new tasks.")
        return first_task_desc_added

    # _autonomous_loop remains unchanged, calls the modified process_one_step
    def _autonomous_loop(self, initial_delay: float = 2.0, step_delay: float = 5.0):
        log.info("Background agent loop starting.")
        time.sleep(initial_delay)
        while not self._shutdown_request.is_set():
            if not self._is_running.is_set():
                with self._state_lock:
                    if self._ui_update_state["status"] != "paused":
                        log.info("Agent loop pausing...")
                        self._ui_update_state["status"] = "paused"
                        self._ui_update_state["log"] = "Agent loop paused."
                time.sleep(1)
                continue

            log.debug("Autonomous loop: Processing cycle...") # Renamed log message
            step_start = time.monotonic()
            try:
                _ = self.process_one_step() # Calls the modified function
            except Exception as e:
                log.exception(f"CRITICAL UNHANDLED ERROR in autonomous loop iteration.")
                self._update_ui_state(status="critical_error", log=f"CRITICAL LOOP ERROR: {e}")
                time.sleep(step_delay * 5)

            step_duration = time.monotonic() - step_start
            remaining_delay = max(0, step_delay - step_duration)
            log.debug(f"Cycle took {step_duration:.2f}s. Sleeping {remaining_delay:.2f}s.") # Renamed log message
            if self._is_running.is_set(): time.sleep(remaining_delay)
            else: log.debug("Agent paused during delay calculation, skipping sleep.")
        log.info("Background agent loop shutting down.")

    # start/pause/shutdown/add_reflection/generate_session_reflection remain unchanged
    def start_autonomous_loop(self):
        if self._agent_thread and self._agent_thread.is_alive():
            if not self._is_running.is_set():
                log.info("Agent loop resuming...")
                self._is_running.set()
                self._update_ui_state(status="running", log="Agent resumed.")
            else: log.info("Agent loop is already running.")
        else:
            log.info("Starting new background agent loop...")
            self._shutdown_request.clear(); self._is_running.set()
            self._update_ui_state(status="running", log="Agent started.")
            self._agent_thread = threading.Thread(
                target=self._autonomous_loop,
                args=(2.0, config.UI_UPDATE_INTERVAL * 2.0,),
                daemon=True,
            )
            self._agent_thread.start()

    def pause_autonomous_loop(self):
        if self._is_running.is_set():
            log.info("Pausing agent loop...")
            self._is_running.clear()
        else: log.info("Agent loop is already paused.")

    def shutdown(self):
        log.info("Shutdown requested.")
        self._shutdown_request.set(); self._is_running.clear()
        if self._agent_thread and self._agent_thread.is_alive():
            log.info("Waiting for agent thread join...")
            self._agent_thread.join(timeout=30)
            if self._agent_thread.is_alive(): log.warning("Agent thread didn't join cleanly.")
            else: log.info("Agent thread joined.")
        else: log.info("Agent thread not running/already joined.")
        self._update_ui_state(status="shutdown", log="Agent shut down.")
        self.save_session_state()
        log.info("Shutdown complete.")

    def add_self_reflection(self, reflection: str, reflection_type: str = "self_reflection"):
        if not reflection or not isinstance(reflection, str) or not reflection.strip(): return None
        if self._is_running.is_set():
            log.info(f"Adding {reflection_type} to memory...")
            return self.memory.add_memory(reflection, {"type": reflection_type})
        else: log.info(f"Agent paused, skipping adding {reflection_type} to memory."); return None

    def generate_and_add_session_reflection(
        self, start: datetime.datetime, end: datetime.datetime,
        completed_count: int, processed_count: int
    ):
        duration_minutes = (end - start).total_seconds() / 60
        log.info("Retrieving context for session reflection...")
        mem_query = "Summary session activities, task summaries, errors, outcomes, identity statements/revisions."
        recent_mems, _ = self.memory.retrieve_and_rerank_memories(
            query=mem_query, task_description="Session Reflection", context="End of work session",
            identity_statement=self.identity_statement, n_results=config.MEMORY_COUNT_REFLECTIONS * 2, n_final=config.MEMORY_COUNT_REFLECTIONS,
        )
        mem_summary_list = []
        for m in recent_mems:
            relative_time = format_relative_time(m["metadata"].get("timestamp")); mem_type = m["metadata"].get("type", "mem")
            snippet = m["content"][:100].strip().replace("\n", " "); mem_summary_list.append(f"- [{relative_time}] {mem_type}: {snippet}...")
        mem_summary = "\n".join(mem_summary_list) if mem_summary_list else "None"
        prompt_vars = {
            "identity_statement": self.identity_statement, "start_iso": start.isoformat(), "end_iso": end.isoformat(),
            "duration_minutes": duration_minutes, "completed_count": completed_count, "processed_count": processed_count,
            "mem_summary": mem_summary,
        }
        prompt = prompts.SESSION_REFLECTION_PROMPT.format(**prompt_vars)
        log.info(f"Asking {self.ollama_chat_model} for session reflection...")
        reflection = call_ollama_api(prompt, self.ollama_chat_model, self.ollama_base_url, timeout=120)
        if reflection and reflection.strip():
            print(f"\n--- Session Reflection ---\n{reflection}\n------")
            self.add_self_reflection(reflection, "session_reflection")
        else: log.warning("Failed to generate session reflection.")

    # get_agent_dashboard_state remains unchanged
    def get_agent_dashboard_state(self) -> Dict[str, Any]:
        log.debug("Gathering dashboard state...")
        try:
            tasks_structured = self.task_queue.get_all_tasks_structured()
            memory_summary = self.memory.get_memory_summary()
            completed_failed_tasks = tasks_structured.get("completed", []) + tasks_structured.get("failed", [])
            completed_failed_tasks.sort(key=lambda t: t.get("Completed At") or t.get("Failed At", "0"), reverse=True)
            summary_items = [f"- {m_type}: {count}" for m_type, count in memory_summary.items()]
            memory_summary_str = "**Memory Summary (by Type):**\n" + "\n".join(sorted(summary_items)) if summary_items else "No memories found."
            in_progress_planning_tasks = tasks_structured.get("planning", []) + tasks_structured.get("in_progress", [])
            in_progress_planning_tasks_sorted = sorted(in_progress_planning_tasks, key=lambda t: t.get("Created", ""))
            return {
                "identity_statement": self.identity_statement, "pending_tasks": tasks_structured.get("pending", []),
                "in_progress_tasks": in_progress_planning_tasks_sorted, "completed_tasks": tasks_structured.get("completed", []),
                "failed_tasks": tasks_structured.get("failed", []), "completed_failed_tasks_data": completed_failed_tasks,
                "memory_summary": memory_summary_str, "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        except Exception as e:
            log.exception("Error gathering agent dashboard state")
            return {
                "identity_statement": "Error fetching state", "pending_tasks": [], "in_progress_tasks": [],
                "completed_tasks": [], "failed_tasks": [], "completed_failed_tasks_data": [],
                "memory_summary": f"Error: {e}", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }

    # get_formatted_memories_for_task remains unchanged
    def get_formatted_memories_for_task(self, task_id: str) -> List[Dict[str, Any]]:
        if not task_id: return []
        log.debug(f"Getting memories for task ID: {task_id}")
        memories = self.memory.get_memories_by_metadata(filter_dict={"task_id": task_id}, limit=100)
        formatted = []
        for mem in memories:
            metadata = mem.get("metadata", {}); formatted.append({
                "Timestamp": metadata.get("timestamp", "N/A"), "Type": metadata.get("type", "N/A"),
                "Content Snippet": (mem.get("content", "")[:200] + "..." if mem.get("content") else "N/A"),
                "ID": mem.get("id", "N/A"),
            })
        return sorted(formatted, key=lambda x: x.get("Timestamp", "0"))

    # get_formatted_general_memories remains unchanged
    def get_formatted_general_memories(self) -> List[Dict[str, Any]]:
        log.debug("Getting general memories...")
        memories = self.memory.get_general_memories(limit=50)
        formatted = []
        for mem in memories:
            metadata = mem.get("metadata", {}); formatted.append({
                "Timestamp": metadata.get("timestamp", "N/A"), "Type": metadata.get("type", "N/A"),
                "Content Snippet": (mem.get("content", "")[:200] + "..." if mem.get("content") else "N/A"),
                "ID": mem.get("id", "N/A"),
            })
        return sorted(formatted, key=lambda x: x.get("Timestamp", "0"), reverse=True)
