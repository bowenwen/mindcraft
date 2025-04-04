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
            "investigation_context": "",
            "current_task_retries": 0,
            "last_checkpoint": None,
            "last_web_browse_content": None,
            "identity_statement": self.identity_statement,
            "user_suggestion_move_on_pending": False,
            "last_step_details": deque(maxlen=config.UI_STEP_HISTORY_LENGTH),
        }
        self.load_session_state()

        self._is_running = threading.Event()
        self._shutdown_request = threading.Event()
        self._agent_thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()

        self._ui_update_state: Dict[str, Any] = {
            "status": "paused",
            "log": "Agent paused.",
            "current_task_id": self.session_state.get("current_task_id"),
            "current_task_desc": "N/A",
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
            "step_history": list(self.session_state.get("last_step_details", [])),
        }

        if self.session_state.get("current_task_id"):
            task = self.task_queue.get_task(self.session_state["current_task_id"])
            if task:
                self._ui_update_state["current_task_desc"] = task.description

        if self.identity_statement == config.INITIAL_IDENTITY_STATEMENT:
            log.info("Performing initial identity revision...")
            self._revise_identity_statement("Initialization / Session Start")
            self.save_session_state()
        else:
            log.info("Loaded existing identity statement. Skipping initial revision.")

        log.info("Agent Initialized.")

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
            self._ui_update_state["step_history"] = list(
                self.session_state.get("last_step_details", [])
            )

    def get_ui_update_state(self) -> Dict[str, Any]:
        with self._state_lock:
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
                    loaded.setdefault("current_task_retries", 0)
                    loaded.setdefault("user_suggestion_move_on_pending", False)
                    loaded_identity = loaded.get(
                        "identity_statement", config.INITIAL_IDENTITY_STATEMENT
                    )
                    if not loaded_identity:
                        loaded_identity = config.INITIAL_IDENTITY_STATEMENT

                    loaded_history = loaded.get("last_step_details")
                    if isinstance(loaded_history, list):
                        loaded_history = loaded_history[
                            -config.UI_STEP_HISTORY_LENGTH :
                        ]
                        loaded["last_step_details"] = deque(
                            loaded_history, maxlen=config.UI_STEP_HISTORY_LENGTH
                        )
                    else:
                        log.warning(
                            f"Invalid or missing 'last_step_details' in session state. Initializing empty history."
                        )
                        loaded["last_step_details"] = deque(
                            maxlen=config.UI_STEP_HISTORY_LENGTH
                        )

                    self.session_state.update(loaded)
                    self.identity_statement = loaded_identity
                    self.session_state["identity_statement"] = self.identity_statement

                    log.info(f"Loaded session state from {self.session_state_path}.")
                    log.info(f"  Loaded Identity: {self.identity_statement[:100]}...")
                    log.info(
                        f"  User Suggestion Pending: {self.session_state.get('user_suggestion_move_on_pending')}"
                    )
                    log.info(
                        f"  Loaded {len(self.session_state['last_step_details'])} step history entries."
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
            self.session_state["last_step_details"] = deque(
                maxlen=config.UI_STEP_HISTORY_LENGTH
            )

    def save_session_state(self):
        try:
            with self._state_lock:
                self.session_state["last_checkpoint"] = datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat()
                self.session_state["identity_statement"] = self.identity_statement
                state_to_save = self.session_state.copy()
                state_to_save["last_step_details"] = list(
                    self.session_state["last_step_details"]
                )
            with open(self.session_state_path, "w", encoding="utf-8") as f:
                json.dump(state_to_save, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.error(f"Error saving session state to '{self.session_state_path}': {e}")

    def handle_user_suggestion_move_on(self) -> str:
        feedback = "Suggestion ignored: Agent not initialized."
        current_task_id = self.session_state.get("current_task_id")
        with self._state_lock:
            if self._is_running.is_set():
                if current_task_id:
                    log.info(
                        f"User suggested moving on from task {current_task_id}. Setting flag."
                    )
                    self.session_state["user_suggestion_move_on_pending"] = True
                    feedback = f"Suggestion noted for current task ({current_task_id}). Agent will consider wrapping up."
                    self.save_session_state()
                    # Optional: Add memory
                    # self.memory.add_memory(f"User Suggestion: Consider moving on from task {current_task_id}.", {"type": "user_suggestion_move_on", "task_id": current_task_id})

                else:
                    log.info("User suggested task change, but agent is idle.")
                    feedback = (
                        "Suggestion ignored: Agent is currently idle (no active task)."
                    )
            else:
                log.info(
                    f"Agent paused, ignoring suggestion to move on from task {current_task_id or 'N/A'} (flag not set)."
                )
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

    def get_available_tools_description(self) -> str:
        """Generates a description string for the available tools and their actions."""
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
                    tool_description += "\n  *Note: Search functionality requires SEARXNG_BASE_URL to be configured.*"
                # Note: We don't disable the tool if the doc archive DB fails,
                # as basic browse (without query) should still work.
                # We could add a note if self.tools['web'].doc_archive_collection is None.

            if name == "memory" and not self.memory:
                is_active = False

            if name == "file" and not config.ARTIFACT_FOLDER:
                log.warning("File tool disabled: ARTIFACT_FOLDER not set.")
                is_active = False

            if name == "status":
                is_active = True

            if is_active:
                active_tools_desc.append(
                    f"- Tool Name: **{name}**\n  Description: {tool_description}"
                )

        if not active_tools_desc:
            return "No tools currently active or available (check configuration)."
        return "\n".join(active_tools_desc)

    def generate_thinking(
        self,
        task_description: str,
        context: str = "",
        tool_results: Optional[Dict[str, Any]] = None,
        user_suggested_move_on: bool = False,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        tool_desc = self.get_available_tools_description()
        memory_query = f"Info relevant to next step? Goal: {context[-500:]}\nTask: {task_description}\nConsider recent chat, user suggestions, errors, or provided info if relevant."
        if tool_results:
            memory_query += f"\nLast result summary: {str(tool_results)[:200]}"

        relevant_memories, separated_suggestion = (
            self.memory.retrieve_and_rerank_memories(
                query=memory_query,
                task_description=task_description,
                context=context,
                identity_statement=self.identity_statement,
                n_results=config.MEMORY_COUNT_GENERAL_THINKING * 2,
                n_final=config.MEMORY_COUNT_GENERAL_THINKING,
            )
        )

        user_provided_info_content = []
        other_memories_context_list = []
        for mem in relevant_memories:
            mem_type = mem.get("metadata", {}).get("type")
            relative_time = format_relative_time(mem["metadata"].get("timestamp"))
            dist_str = (
                f"{mem.get('distance', -1.0):.3f}"
                if mem.get("distance") is not None
                else "N/A"
            )
            mem_content = f"[Memory - {relative_time}] (Type: {mem_type}, Dist: {dist_str}, ID: {mem.get('id', 'N/A')}):\n{mem['content']}"
            if mem_type == "user_provided_info_for_task":
                user_provided_info_content.append(mem_content)
            else:
                other_memories_context_list.append(mem_content)

        memory_context_str = (
            "\n\n".join(other_memories_context_list)
            if other_memories_context_list
            else "No other relevant memories selected."
        )
        user_provided_info_str = (
            "\n---\n".join(user_provided_info_content)
            if user_provided_info_content
            else None
        )

        prompt_vars = {
            "identity_statement": self.identity_statement,
            "task_description": task_description,
            "tool_desc": tool_desc,
            "memory_context_str": memory_context_str,
        }
        prompt_text = prompts.GENERATE_THINKING_PROMPT_BASE.format(**prompt_vars)

        if user_suggested_move_on:
            prompt_text += f"\n\n**USER SUGGESTION PENDING:** The user suggested considering wrapping up this task soon."
        if user_provided_info_str:
            prompt_text += f"\n\n**User Provided Information (Consider for current step):**\n{user_provided_info_str}\n"

        prompt_text += f"""
**Current Investigation Context (History - IMPORTANT: May contain previous errors):**
{context if context else "First step for this task."}\n"""

        if tool_results:
            tool_name = tool_results.get("tool_name", "Unknown")
            tool_action = tool_results.get("action", "unknown")
            prompt_text += f"\n**Results from Last Action:**\nTool: {tool_name} (Action: {tool_action})\nResult:\n```json\n{json.dumps(tool_results.get('result', tool_results), indent=2, ensure_ascii=False)}\n```\n"  # Show full result dict if 'result' key missing
        else:
            prompt_text += "\n**Results from Last Action:**\nNone.\n"

        prompt_text += """
**Your Task Now:**
1.  **Analyze:** Review ALL provided information (Identity, Task, Tools/Actions, Memories - including recency/novelty, User Provided Info, Context, Last Result).
2.  **Reason:** Determine the single best action *right now* to make progress towards the **Overall Task**, consistent with your identity.
3.  **User Input Handling:**
    *   If **User Provided Information** exists, incorporate it into your plan for the *current* step if relevant.
    *   If **USER SUGGESTION PENDING** exists: Acknowledge it in your thinking. *Strongly consider* concluding the task in this step or the next using `final_answer` (e.g., summarizing findings so far) **unless** critical sub-steps are still absolutely required to meet the core task objective. Do not stop abruptly if essential work remains unfinished.
4.  **CRITICAL: Handle Previous Errors:** If the **Context** mentions an error from a previous step, focus on a **RECOVERY STRATEGY** (e.g., retry tool/action, different tool/action, fix params, refine search, use memory tool search action) before considering giving up or moving on based *only* on the error.
5.  **Choose Action:** Decide between `use_tool` or `final_answer`. Use `final_answer` only when the *original* task goal is fully met, or if reasonably wrapping up based on sufficient progress and the user suggestion.
**Output Format (Strict JSON-like structure):**
THINKING:
<Your reasoning process. Explain analysis, memory consideration (recency/novelty), how you're handling user suggestion/info (if any), your plan (especially recovery or conclusion strategy), and alignment with identity/task. Specify the chosen tool AND action (if applicable). For 'web browse', state if you are using the 'query' parameter for focused retrieval.>
NEXT_ACTION: <"use_tool" or "final_answer">
If NEXT_ACTION is "use_tool":
TOOL: <name_of_tool_from_list (e.g., web, memory, file, status)>
PARAMETERS: <{{ "action": "specific_action_for_tool", "param1": "value1", ... }} or {{}}> <-- MUST INCLUDE 'action' for web/memory/file tools! **Use empty dict {{}} for 'status' tool.**
If NEXT_ACTION is "final_answer":
ANSWER: <Your complete answer, or a summary if concluding early based on suggestion.>
REFLECTIONS: <Optional thoughts.>
**Critical Formatting Reminders:** Start *immediately* with "THINKING:". Structure must be exact. `PARAMETERS` must be valid JSON. For web/memory/file tools, `PARAMETERS` **must include the "action" key** corresponding to the tool's capabilities (e.g., "search", "browse", "read", "write", "list", "status_report"). **For the 'status' tool, use an empty JSON object `{}` for PARAMETERS.** Tool name must be one of the available tools. For `web` tool `browse` action, you can optionally include a `"query"` parameter to search within the page content.
"""

        log.info(
            f"Asking {self.ollama_chat_model} for next action (new tool structure)..."
        )
        llm_response_text = call_ollama_api(
            prompt_text, self.ollama_chat_model, self.ollama_base_url, timeout=180
        )
        if llm_response_text is None:
            log.error("Failed to get thinking response from Ollama.")
            return "LLM communication failed.", {
                "type": "error",
                "message": "LLM communication failure (thinking).",
                "subtype": "llm_comm_error",
            }

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
                end_think = (
                    action_start
                    if action_start > thinking_start
                    else len(llm_response_text)
                )
                raw_thinking = llm_response_text[
                    thinking_start + len(thinking_marker) : end_think
                ].strip()
            else:
                log.warning("'THINKING:' marker not found in LLM response.")
                raw_thinking = llm_response_text
            action_type_str = ""
            action_subtype = "unknown_error"
            if action_start != -1:
                end_action = llm_response_text.find("\n", action_start)
                end_action = end_action if end_action != -1 else len(llm_response_text)
                action_type_str = llm_response_text[
                    action_start + len(action_marker) : end_action
                ].strip()
            else:
                log.error("'NEXT_ACTION:' marker not found.")
                action = {
                    "type": "error",
                    "message": "Missing NEXT_ACTION marker.",
                    "subtype": "parse_error",
                }
                return raw_thinking, action

            if "use_tool" in action_type_str:
                action["type"] = "use_tool"
                tool_name = None
                params_json = None
                tool_start = llm_response_text.find(tool_marker, action_start)
                params_start = llm_response_text.find(params_marker, action_start)

                if tool_start != -1:
                    end_tool = llm_response_text.find("\n", tool_start)
                    if params_start > tool_start and (
                        end_tool == -1 or params_start < end_tool
                    ):
                        end_tool = params_start
                    end_tool = end_tool if end_tool != -1 else len(llm_response_text)
                    tool_name = (
                        llm_response_text[tool_start + len(tool_marker) : end_tool]
                        .strip()
                        .strip('"')
                        .strip("'")
                    )
                    log.debug(f"Extracted tool name: '{tool_name}'")
                else:
                    action = {
                        "type": "error",
                        "message": "Missing TOOL marker for use_tool action.",
                        "subtype": "parse_error",
                    }
                    return raw_thinking, action

                if params_start != -1:
                    params_str_start = params_start + len(params_marker)
                    end_params = len(llm_response_text)
                    next_marker_pos = -1
                    for marker in [
                        f"\n{answer_marker}",
                        f"\n{reflections_marker}",
                        f"\n{thinking_marker}",
                        f"\n{action_marker}",
                        f"\n{tool_marker}",
                        f"\n{params_marker}",
                    ]:
                        pos = llm_response_text.find(marker, params_str_start)
                        if pos != -1 and (
                            next_marker_pos == -1 or pos < next_marker_pos
                        ):
                            next_marker_pos = pos
                    if next_marker_pos != -1:
                        end_params = next_marker_pos
                    raw_params = llm_response_text[params_str_start:end_params].strip()
                    raw_params = re.sub(
                        r"^```json\s*", "", raw_params, flags=re.I | re.M
                    )
                    raw_params = re.sub(r"^```\s*", "", raw_params, flags=re.M)
                    raw_params = re.sub(r"\s*```$", "", raw_params, flags=re.M)
                    raw_params = raw_params.strip()
                    json_str = raw_params
                    params_json = None
                    log.debug(f"Raw PARAMS string: '{json_str}'")

                    if tool_name == "status" and json_str == "{}":
                        params_json = {}
                        log.debug("PARAMS parsed as empty dict {} for status tool.")
                    elif json_str:
                        try:
                            params_json = json.loads(json_str)
                            log.debug(f"PARAMS parsed directly: {params_json}")
                        except json.JSONDecodeError as e1:
                            log.warning(f"Direct JSON parse failed: {e1}. Fixing...")
                            fixed_str = (
                                json_str.replace("“", '"')
                                .replace("”", '"')
                                .replace("‘", "'")
                                .replace("’", "'")
                            )
                            fixed_str = re.sub(r",\s*([}\]])", r"\1", fixed_str)
                            log.debug(f"Attempting parse after fixes: '{fixed_str}'")
                            try:
                                params_json = json.loads(fixed_str)
                                log.debug(f"PARAMS parsed after fixes: {params_json}")
                            except json.JSONDecodeError as e2:
                                first_brace = raw_params.find("{")
                                last_brace = raw_params.rfind("}")
                                extracted_str = (
                                    raw_params[first_brace : last_brace + 1]
                                    if first_brace != -1 and last_brace > first_brace
                                    else raw_params
                                )
                                log.debug(f"Extracting braces: '{extracted_str}'")
                                try:
                                    fixed_extracted_str = (
                                        extracted_str.replace("“", '"')
                                        .replace("”", '"')
                                        .replace("‘", "'")
                                        .replace("’", "'")
                                    )
                                    fixed_extracted_str = re.sub(
                                        r",\s*([}\]])", r"\1", fixed_extracted_str
                                    )
                                    log.debug(
                                        f"Trying extracted+fixed: '{fixed_extracted_str}'"
                                    )
                                    params_json = json.loads(fixed_extracted_str)
                                    log.debug(
                                        f"PARAMS parsed after extraction+fixes: {params_json}"
                                    )
                                except json.JSONDecodeError as e3:
                                    err_msg = f"Invalid JSON in PARAMETERS after all attempts: {e3}. Original: '{raw_params}'"
                                    log.error(err_msg)
                                    action = {
                                        "type": "error",
                                        "message": err_msg,
                                        "subtype": "parse_error",
                                    }
                                    return raw_thinking, action
                    elif tool_name == "status" and not json_str:
                        params_json = {}
                        log.debug(
                            "PARAMS interpreted as empty dict {} for status tool (no params provided)."
                        )
                    else:
                        action = {
                            "type": "error",
                            "message": "Empty PARAMETERS block.",
                            "subtype": "parse_error",
                        }
                        return raw_thinking, action
                elif tool_name != "status":
                    action = {
                        "type": "error",
                        "message": "Missing PARAMETERS marker for use_tool action.",
                        "subtype": "parse_error",
                    }
                    return raw_thinking, action
                else:
                    params_json = {}
                    log.debug(
                        "No PARAMETERS marker found for status tool, assuming empty params {}."
                    )

                # Validate Tool, Action, and Required Parameters
                if action.get("type") != "error":
                    if not tool_name:
                        action = {
                            "type": "error",
                            "message": "Internal Error: Tool name missing.",
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
                            "message": f"Parsed PARAMETERS not a JSON object: {type(params_json)}",
                            "subtype": "parse_error",
                        }
                    else:
                        tool_action = params_json.get("action")
                        valid_params = True
                        err_msg = ""
                        action_subtype = "invalid_params"

                        # --- Parameter validation based on tool and action ---
                        if tool_name == "web":
                            if not tool_action or not isinstance(tool_action, str):
                                err_msg = "Missing or invalid 'action' key within PARAMETERS for web tool."
                                action_subtype = "missing_action"
                            elif tool_action == "search" and (
                                "query" not in params_json
                                or not isinstance(params_json.get("query"), str)
                                or not params_json["query"].strip()
                            ):
                                err_msg = "Missing/invalid 'query' for web search."
                            elif tool_action == "browse":
                                if (
                                    "url" not in params_json
                                    or not isinstance(params_json.get("url"), str)
                                    or not params_json["url"].strip()
                                ):
                                    err_msg = "Missing/invalid 'url' for web browse."
                                # --- MODIFICATION: Allow optional 'query' for browse ---
                                elif "query" in params_json and (
                                    not isinstance(params_json.get("query"), str)
                                    or not params_json["query"].strip()
                                ):
                                    err_msg = "Invalid 'query' for web browse (must be non-empty string if provided)."
                                # --- END MODIFICATION ---
                            elif tool_action not in ["search", "browse"]:
                                err_msg = (
                                    f"Invalid action '{tool_action}' for web tool."
                                )
                                action_subtype = "invalid_action"
                        # ... (memory, file, status validation unchanged) ...
                        elif tool_name == "memory":
                            if not tool_action or not isinstance(tool_action, str):
                                err_msg = "Missing or invalid 'action' key within PARAMETERS for memory tool."
                                action_subtype = "missing_action"
                            elif tool_action == "search" and (
                                "query" not in params_json
                                or not isinstance(params_json.get("query"), str)
                                or not params_json["query"].strip()
                            ):
                                err_msg = "Missing/invalid 'query' for memory search."
                            elif tool_action == "write" and (
                                "content" not in params_json
                                or not isinstance(params_json.get("content"), str)
                            ):
                                err_msg = "Missing/invalid 'content' (must be string) for memory write."
                            elif tool_action not in ["search", "write"]:
                                err_msg = (
                                    f"Invalid action '{tool_action}' for memory tool."
                                )
                                action_subtype = "invalid_action"
                        elif tool_name == "file":
                            if not tool_action or not isinstance(tool_action, str):
                                err_msg = "Missing or invalid 'action' key within PARAMETERS for file tool."
                                action_subtype = "missing_action"
                            elif tool_action == "read" and (
                                "filename" not in params_json
                                or not isinstance(params_json.get("filename"), str)
                                or not params_json["filename"].strip()
                            ):
                                err_msg = "Missing/invalid 'filename' for file read."
                            elif tool_action == "write" and (
                                "filename" not in params_json
                                or not isinstance(params_json.get("filename"), str)
                                or not params_json["filename"].strip()
                                or "content" not in params_json
                                or not isinstance(params_json.get("content"), str)
                            ):
                                err_msg = "Missing/invalid 'filename' or 'content' for file write."
                            elif tool_action == "list" and (
                                "directory" in params_json
                                and not isinstance(params_json.get("directory"), str)
                            ):
                                err_msg = "Invalid 'directory' (must be string if provided) for file list."
                            elif tool_action not in ["read", "write", "list"]:
                                err_msg = (
                                    f"Invalid action '{tool_action}' for file tool."
                                )
                                action_subtype = "invalid_action"
                        elif tool_name == "status":
                            if params_json and params_json != {}:
                                err_msg = "Status tool does not accept any parameters."
                                action_subtype = "invalid_params"
                        else:
                            err_msg = f"Parameter validation not implemented for tool '{tool_name}'."
                            action_subtype = "internal_error"

                        if err_msg:
                            log.error(
                                f"{err_msg} Tool: {tool_name}, Action: {tool_action}, Params: {params_json}"
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
                                f"Parsed action: Use Tool '{tool_name}'"
                                + (f", Action '{tool_action}'" if tool_action else "")
                            )
            # --- End parameter validation update ---

            elif "final_answer" in action_type_str:
                # --- Final Answer parsing unchanged ---
                action["type"] = "final_answer"
                answer = ""
                reflections = ""
                answer_start = llm_response_text.find(answer_marker, action_start)
                reflections_start = llm_response_text.find(
                    reflections_marker, action_start
                )
                if answer_start != -1:
                    end_answer = (
                        reflections_start
                        if reflections_start > answer_start
                        else len(llm_response_text)
                    )
                    answer = llm_response_text[
                        answer_start + len(answer_marker) : end_answer
                    ].strip()
                else:
                    potential_answer_start = (
                        llm_response_text.find("\n", action_start) + 1
                    )
                    potential_answer_end = (
                        reflections_start
                        if reflections_start > potential_answer_start
                        else len(llm_response_text)
                    )
                    answer = (
                        llm_response_text[
                            potential_answer_start:potential_answer_end
                        ].strip()
                        if potential_answer_start > 0
                        and potential_answer_start < potential_answer_end
                        else ""
                    )
                if reflections_start != -1:
                    reflections = llm_response_text[
                        reflections_start + len(reflections_marker) :
                    ].strip()
                action["answer"] = answer
                action["reflections"] = reflections
                if not answer:
                    action = {
                        "type": "error",
                        "message": "LLM chose final_answer but provided no ANSWER.",
                        "subtype": "parse_error",
                    }
                elif (
                    len(answer) < 100
                    and re.search(
                        r"\b(error|fail|cannot|unable)\b", answer, re.IGNORECASE
                    )
                    and not user_suggested_move_on
                ):
                    log.warning(
                        f"LLM 'final_answer' looks like giving up without user suggestion: '{answer[:100]}...'"
                    )
                    action = {
                        "type": "error",
                        "message": f"LLM gave up instead of recovering: {answer}",
                        "subtype": "give_up_error",
                    }
                else:
                    log.info(f"Parsed action: Final Answer.")
            elif action_type_str:
                action = {
                    "type": "error",
                    "message": f"Invalid NEXT_ACTION: '{action_type_str}'",
                    "subtype": "parse_error",
                }
            if (
                action["type"] == "unknown"
                and action.get("subtype", "") != "parse_error"
            ):
                action = {
                    "type": "error",
                    "message": "Could not parse action.",
                    "subtype": "parse_error",
                }
            return raw_thinking, action
        except Exception as e:
            log.exception(
                f"CRITICAL failure parsing LLM thinking response: {e}\nResponse:\n{llm_response_text}"
            )
            return raw_thinking or "Error parsing.", {
                "type": "error",
                "message": f"Internal error parsing LLM response: {e}",
                "subtype": "internal_error",
            }

    def execute_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        # ... (method implementation unchanged, relies on tool's run method) ...
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found", "status": "failed"}
        tool = self.tools[tool_name]
        tool_action = parameters.get("action")

        if tool_name == "status":
            tool_action = "status_report"
        log_action = tool_action if tool_action else "(implicit)"
        log.info(
            f"Executing tool '{tool_name}', action '{log_action}' with params: {parameters}"
        )

        try:
            result: Dict[str, Any] = {}
            if tool_name == "memory":
                result = tool.run(
                    parameters,
                    memory_instance=self.memory,
                    identity_statement=self.identity_statement,
                )
            elif tool_name == "status":
                log.debug("Gathering state for status tool...")
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
            elif tool_name in [
                "web",
                "file",
            ]:  # Web tool execution remains the same call signature
                result = tool.run(parameters)
            else:
                result = tool.run(parameters)

            log.info(f"Tool '{tool_name}' action '{log_action}' finished.")

            # Store last browse content specifically if it was a non-query browse
            if (
                tool_name == "web"
                and tool_action == "browse"
                and isinstance(result, dict)
                and result.get("query_mode") is False  # Only store if not query mode
                and "content" in result
            ):
                with self._state_lock:
                    self.session_state["last_web_browse_content"] = result.get(
                        "content", "(Empty)"
                    )
            elif (
                tool_name == "web"
                and tool_action == "browse"
                and isinstance(result, dict)
                and result.get("query_mode") is True
            ):
                # Clear or update last browse content if query mode was used?
                # Maybe set it to indicate snippets were retrieved.
                with self._state_lock:
                    self.session_state["last_web_browse_content"] = (
                        f"(Retrieved snippets for query: {parameters.get('query', 'N/A')})"
                    )

            if not isinstance(result, dict):
                log.warning(
                    f"Result from '{tool_name}' ({log_action}) not a dict: {type(result)}"
                )
                return {
                    "tool_name": tool_name,
                    "action": tool_action,
                    "result": {"unexpected_result": str(result)},
                    "status": "completed_malformed_output",
                }

            if result.get("error"):
                log.warning(
                    f"Tool '{tool_name}' ({log_action}) reported error: {result['error']}"
                )
                return {
                    "tool_name": tool_name,
                    "action": tool_action,
                    "error": result["error"],
                    "status": "failed",
                    "result": result,  # Include original result for context
                }

            try:
                json.dumps(result)
            except TypeError as json_err:
                log.warning(
                    f"Result from '{tool_name}' ({log_action}) not JSON serializable: {json_err}."
                )
                serializable_result = {}
                for k, v in result.items():
                    try:
                        json.dumps({k: v})
                        serializable_result[k] = v
                    except TypeError:
                        serializable_result[k] = (
                            f"<Unserializable type: {type(v).__name__}> {str(v)[:100]}..."
                        )
                return {
                    "tool_name": tool_name,
                    "action": tool_action,
                    "result": serializable_result,
                    "status": "completed_unserializable",
                }

            # If no error reported by tool, assume success
            # Use status from tool result if provided, otherwise default to 'completed'
            final_status = result.get("status", "completed")
            final_result = {
                "tool_name": tool_name,
                "action": tool_action,
                "status": final_status,  # Use status from result
            }
            final_result.update(result)
            final_result.pop("error", None)
            # Also remove 'status' key if it was merged in, as we handle it separately
            if "status" in result:
                final_result.pop("status", None)

            # Wrap the tool's output under a 'result' key for consistency in the UI state
            return {
                "tool_name": tool_name,
                "action": tool_action,
                "status": final_status,
                "result": final_result,  # Nest the tool output here
            }

        except Exception as e:
            log.exception(
                f"CRITICAL Error executing tool '{tool_name}' action '{log_action}': {e}"
            )
            return {
                "tool_name": tool_name,
                "action": tool_action,
                "error": f"Tool execution raised exception: {e}",
                "status": "failed",
            }

    # ... (rest of agent.py methods like _save_qlora_datapoint, _summarize_and_prune_task_memories, _execute_step, etc. are unchanged but benefit from the updated tool output structure where applicable) ...

    def _save_qlora_datapoint(
        self, source_type: str, instruction: str, input_context: str, output: str
    ):
        # ... (unchanged) ...
        if not output:
            return
        try:
            datapoint = {
                "instruction": instruction,
                "input": input_context,
                "output": output,
                "source_type": source_type,
            }
            with open(self.qlora_dataset_path, "a", encoding="utf-8") as f:
                json.dump(datapoint, f, ensure_ascii=False)
                f.write("\n")
            log.info(f"QLoRA datapoint saved (Source: {source_type})")
        except Exception as e:
            log.exception(f"Failed to save QLoRA datapoint: {e}")

    def _summarize_and_prune_task_memories(self, task: Task):
        # ... (unchanged) ...
        if not config.ENABLE_MEMORY_SUMMARIZATION:
            return
        log.info(f"Summarizing memories for {task.status} task {task.id}...")
        task_memories = self.memory.get_memories_by_metadata(
            filter_dict={"task_id": task.id}
        )
        if not task_memories:
            return
        log.debug(f"Found {len(task_memories)} memories for task {task.id}.")
        summary_context = f"Task: {task.description}\nStatus: {task.status}\nResult: {str(task.result)[:500]}\nReflections: {task.reflections}\n--- Log (with relative time) ---\n"
        try:
            task_memories.sort(key=lambda m: m.get("metadata", {}).get("timestamp", ""))
        except:
            pass
        mem_details = []
        memory_ids_to_prune = []
        for mem in task_memories:
            meta = mem.get("metadata", {})
            mem_type = meta.get("type", "memory")
            if mem_type in [
                "task_summary",
                "identity_revision",
                "session_reflection",
                "user_suggestion_move_on",
            ]:
                continue
            timestamp_str = meta.get("timestamp", "N/A")
            content = mem.get("content", "")
            relative_time = format_relative_time(timestamp_str)
            tool_action_str = ""
            if (
                mem_type == "tool_result"
                and meta.get("tool_name")
                and meta.get("action")
            ):
                tool_action_str = (
                    f" (Tool: {meta.get('tool_name')}, Action: {meta.get('action')})"
                )
            mem_details.append(
                f"[{relative_time} - {mem_type}{tool_action_str}]\n{content}\n-------"
            )
            memory_ids_to_prune.append(mem["id"])

        summary_context += "\n".join(mem_details) + "\n--- End Log ---"
        trunc_limit = max(0, config.CONTEXT_TRUNCATION_LIMIT // 2)
        summary_prompt = prompts.SUMMARIZE_TASK_PROMPT.format(
            task_status=task.status,
            summary_context=summary_context[:trunc_limit],
            context_truncation_limit=trunc_limit,
        )
        log.info(f"Asking {self.ollama_chat_model} to summarize task {task.id}...")
        summary_text = call_ollama_api(
            summary_prompt, self.ollama_chat_model, self.ollama_base_url, timeout=150
        )
        if summary_text and summary_text.strip():
            summary_metadata = {
                "type": "task_summary",
                "task_id": task.id,
                "original_status": task.status,
                "summarized_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "original_memory_count": len(memory_ids_to_prune),
            }
            summary_id = self.memory.add_memory(
                f"Task Summary ({task.status}):\n{summary_text}", summary_metadata
            )
            if summary_id and config.DELETE_MEMORIES_AFTER_SUMMARY:
                log.info(
                    f"Summary added ({summary_id}). Pruning {len(memory_ids_to_prune)} original memories for task {task.id}..."
                )
                ids_to_delete_final = [
                    mid for mid in memory_ids_to_prune if mid != summary_id
                ]
                if ids_to_delete_final:
                    deleted = self.memory.delete_memories(ids_to_delete_final)
                    log.info(f"Pruning status for task {task.id}: {deleted}")
                else:
                    log.info(
                        f"No original memories to prune for task {task.id} after summary creation."
                    )
            elif summary_id:
                log.info(f"Summary added ({summary_id}). Deletion disabled.")
            else:
                log.error(f"Failed to add summary memory for task {task.id}.")
        else:
            log.warning(f"Failed to generate summary for task {task.id}.")

    def _execute_step(self, task: Task) -> Dict[str, Any]:
        # ... (Step 1: Generate Thinking - unchanged) ...
        step_start_time = time.time()
        step_log = []
        final_answer_text = None
        step_status = "processing"
        task_status_updated_this_step = False
        current_context = self.session_state.get("investigation_context", "")
        step_num = self.session_state.get(f"task_{task.id}_step", 0) + 1
        current_retries = self.session_state.get("current_task_retries", 0)
        user_suggested_move_on = self.session_state.get(
            "user_suggestion_move_on_pending", False
        )
        if user_suggested_move_on:
            log.info(f"User suggestion 'move on' is pending for task {task.id}.")
            with self._state_lock:
                self.session_state["user_suggestion_move_on_pending"] = False
        step_log.append(
            f"--- Task '{task.id}' | Step {step_num} (Retry {current_retries}/{config.AGENT_MAX_STEP_RETRIES}) ---"
        )
        if user_suggested_move_on:
            step_log.append(
                "[INFO] User suggestion to wrap up is being considered this step."
            )
        log.info(
            f"Executing step {step_num} for task {task.id} (Retry: {current_retries})"
        )

        raw_thinking, action = self.generate_thinking(
            task.description,
            current_context,
            None,
            user_suggested_move_on=user_suggested_move_on,
        )
        action_type = action.get("type", "error")
        action_message = action.get("message", "Unknown error")
        action_subtype = action.get("subtype", "unknown_error")
        thinking_to_store = (
            raw_thinking if raw_thinking else "Thinking process not extracted."
        )
        step_log.append(f"Thinking:\n{thinking_to_store}")
        self.memory.add_memory(
            f"Step {step_num} Thinking (Action: {action_type}, Subtype: {action_subtype}):\n{thinking_to_store}",
            {
                "type": "task_thinking",
                "task_id": task.id,
                "step": step_num,
                "action_type": action_type,
                "action_subtype": action_subtype,
            },
        )
        step_details_for_history = {
            "task_id": task.id,
            "step": step_num,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "thinking": thinking_to_store,
            "action_type": action_type,
            "action_params": None,
            "result_status": None,
            "result_summary": None,
            "log_snippet": None,
        }
        tool_results_for_ui = None

        # ... (Step 2: Process Action or Error - largely unchanged, handle error status) ...
        if action_type == "error":
            log.error(
                f"[STEP ERROR] LLM Action Error: {action_message} (Subtype: {action_subtype})"
            )
            step_log.append(
                f"[ERROR] Action Error: {action_message} (Type: {action_subtype})"
            )
            error_context = f"\n--- Step {step_num} Error (Retry {current_retries+1}/{config.AGENT_MAX_STEP_RETRIES}): {action_message} (Type: {action_subtype}) ---"
            current_context += error_context
            self.memory.add_memory(
                f"Step {step_num} Error: {action_message}",
                {
                    "type": "agent_error",
                    "task_id": task.id,
                    "step": step_num,
                    "error_subtype": action_subtype,
                    "llm_response_snippet": raw_thinking[:200],
                },
            )
            step_details_for_history["result_status"] = "error"
            step_details_for_history["result_summary"] = (
                f"LLM Action Error: {action_message} ({action_subtype})"
            )
            if current_retries < config.AGENT_MAX_STEP_RETRIES:
                self.session_state["current_task_retries"] = current_retries + 1
                log.warning(f"Incrementing retry count to {current_retries + 1}.")
                step_status = "error_retry"
            else:
                log.error(f"Max retries reached for task {task.id}. Failing.")
                fail_reason = f"Failed after {step_num} steps. Max retries on LLM action error: {action_message}"
                self.task_queue.update_task(task.id, "failed", reflections=fail_reason)
                task_status_updated_this_step = True
                step_status = "failed"
                failed_task_obj = self.task_queue.get_task(task.id)
                if failed_task_obj:
                    self._handle_task_completion_or_failure(failed_task_obj)
                self.session_state["current_task_id"] = None
                self.session_state["investigation_context"] = ""
                self.session_state["current_task_retries"] = 0

        elif action_type == "use_tool":
            tool_name = action.get("tool")
            params = action.get("parameters")
            tool_action = params.get("action") if params else None
            if tool_name == "status":
                tool_action = "status_report"
            log.info(
                f"[ACTION] Use Tool '{tool_name}', Action '{tool_action if tool_action else '(implicit)'}'"
            )
            step_log.append(
                f"[ACTION] Use Tool: {tool_name}, Action: {tool_action if tool_action else '(implicit)'}, Params: {json.dumps(params, ensure_ascii=False, indent=2)}"
            )
            step_details_for_history["action_params"] = params

            # --- MODIFICATION: Use updated execute_tool output structure ---
            tool_output = self.execute_tool(
                tool_name, params
            )  # This now returns a structured dict
            tool_results_for_ui = tool_output  # Store the whole dict for UI display
            tool_status = tool_output.get("status", "unknown")
            tool_error = tool_output.get("error")
            tool_action_from_result = tool_output.get(
                "action", tool_action
            )  # Use action from result if available
            # The actual results are nested under the 'result' key now
            result_content = tool_output.get("result", {})
            step_details_for_history["result_status"] = tool_status
            # --- End MODIFICATION ---

            # Log tool result to context and memory (use result_content)
            result_display_str = "(No result field)"
            try:
                result_display_str = json.dumps(
                    result_content, indent=2, ensure_ascii=False
                )
            except Exception:
                result_display_str = str(result_content)

            summary_limit = 500
            result_summary_for_history = result_display_str[:summary_limit] + (
                "..." if len(result_display_str) > summary_limit else ""
            )
            if tool_error:
                result_summary_for_history = (
                    f"Error: {tool_error}\n---\n{result_summary_for_history}"
                )
                result_summary_for_history = result_summary_for_history[
                    :summary_limit
                ] + ("..." if len(result_summary_for_history) > summary_limit else "")
            step_details_for_history["result_summary"] = result_summary_for_history

            context_display_limit = 1000
            result_display_context = result_display_str[:context_display_limit] + (
                "...(truncated)"
                if len(result_display_str) > context_display_limit
                else ""
            )

            step_context_update = f"\n--- Step {step_num} ---\nAction: Use Tool '{tool_name}', Action '{tool_action_from_result if tool_action_from_result else '(implicit)'}'\nParams: {json.dumps(params, ensure_ascii=False)}\nResult Status: {tool_status}\nResult:\n```json\n{result_display_context}\n```\n"  # Assume JSON for context log
            current_context += step_context_update
            self.memory.add_memory(
                f"Step {step_num} Tool Action & Result (Status: {tool_status}):\n{step_context_update}",
                {
                    "type": "tool_result",
                    "task_id": task.id,
                    "step": step_num,
                    "tool_name": tool_name,
                    "action": tool_action_from_result,
                    "result_status": tool_status,
                    "params": json.dumps(params),
                },
            )

            # Handle Tool Result Status (largely unchanged logic, uses tool_status from tool_output)
            if tool_status in [
                "completed",
                "success",
                "completed_malformed_output",
                "completed_unserializable",
                "completed_with_issue",
            ]:  # Treat non-failure as okay
                if tool_status not in ["completed", "success"]:
                    log.warning(
                        f"Tool '{tool_name}' action '{tool_action_from_result}' finished with status: {tool_status}."
                    )
                    step_log.append(
                        f"[WARNING] Tool '{tool_name}' ({tool_action_from_result}) status: {tool_status}."
                    )
                else:
                    step_log.append(
                        f"[INFO] Tool '{tool_name}' action '{tool_action_from_result}' completed."
                    )
                step_status = "processing"
                self.session_state["current_task_retries"] = 0
            elif tool_status == "failed":
                log.warning(
                    f"Tool '{tool_name}' action '{tool_action_from_result}' failed: {tool_error}"
                )
                step_log.append(
                    f"[ERROR] Tool '{tool_name}' ({tool_action_from_result}) failed: {tool_error}"
                )
                error_context = f"\n--- Step {step_num} Tool Error (Retry {current_retries+1}/{config.AGENT_MAX_STEP_RETRIES}): Tool={tool_name}, Action={tool_action_from_result}, Error={tool_error} ---"
                current_context += error_context
                self.memory.add_memory(
                    f"Step {step_num} Tool Error: {tool_error}",
                    {
                        "type": "agent_error",
                        "task_id": task.id,
                        "step": step_num,
                        "error_subtype": "tool_execution_error",
                        "tool_name": tool_name,
                        "action": tool_action_from_result,
                    },
                )
                if current_retries < config.AGENT_MAX_STEP_RETRIES:
                    self.session_state["current_task_retries"] = current_retries + 1
                    log.warning(f"Incrementing retry count to {current_retries + 1}.")
                    step_status = "error_retry"
                else:
                    log.error(
                        f"Max retries reached after tool failure for task {task.id}. Failing."
                    )
                    fail_reason = f"Failed after {step_num} steps. Max retries on tool failure: Tool={tool_name}, Action={tool_action_from_result}, Error={tool_error}"
                    self.task_queue.update_task(
                        task.id, "failed", reflections=fail_reason
                    )
                    task_status_updated_this_step = True
                    step_status = "failed"
                    failed_task_obj = self.task_queue.get_task(task.id)
                    if failed_task_obj:
                        self._handle_task_completion_or_failure(failed_task_obj)
                    self.session_state["current_task_id"] = None
                    self.session_state["investigation_context"] = ""
                    self.session_state["current_task_retries"] = 0
            else:  # Unknown tool status
                log.error(
                    f"Tool '{tool_name}' action '{tool_action_from_result}' returned UNKNOWN status: {tool_status}. Failing step."
                )
                step_log.append(
                    f"[CRITICAL] Tool '{tool_name}' ({tool_action_from_result}) unknown status: {tool_status}"
                )
                step_status = "failed"
                fail_reason = f"Failed after {step_num} steps due to unknown tool status '{tool_status}' from tool {tool_name} ({tool_action_from_result})."
                self.task_queue.update_task(task.id, "failed", reflections=fail_reason)
                task_status_updated_this_step = True
                failed_task_obj = self.task_queue.get_task(task.id)
                if failed_task_obj:
                    self._handle_task_completion_or_failure(failed_task_obj)
                self.session_state["current_task_id"] = None
                self.session_state["investigation_context"] = ""
                self.session_state["current_task_retries"] = 0

        # ... (Step 2: Final Answer - unchanged) ...
        elif action_type == "final_answer":
            log.info("[ACTION] Provide Final Answer.")
            step_log.append("[ACTION] Provide Final Answer.")
            answer = action.get("answer", "").strip()
            reflections = action.get("reflections", "").strip()
            final_answer_text = answer
            step_details_for_history["result_status"] = "completed"
            step_details_for_history["result_summary"] = (
                f"Final Answer Provided:\n{answer[:500]}"
                + ("..." if len(answer) > 500 else "")
            )
            print("\n" + "=" * 15 + f" FINAL ANSWER (Task {task.id}) " + "=" * 15)
            print(answer)
            print("=" * (34 + len(str(task.id))) + "\n")
            result_payload = {"answer": answer, "steps_taken": step_num}
            self.task_queue.update_task(
                task.id, "completed", result=result_payload, reflections=reflections
            )
            task_status_updated_this_step = True
            step_status = "completed"
            self._save_qlora_datapoint(
                source_type="task_completion",
                instruction="Given the task, provide the final answer based on the investigation.",
                input_context=task.description,
                output=answer,
            )
            self.memory.add_memory(
                f"Final Answer:\n{answer}",
                {"type": "task_result", "task_id": task.id, "final_step": step_num},
            )
            if reflections:
                self.memory.add_memory(
                    f"Final Reflections:\n{reflections}",
                    {
                        "type": "task_reflection",
                        "task_id": task.id,
                        "final_step": step_num,
                    },
                )
            completed_task_obj = self.task_queue.get_task(task.id)
            if completed_task_obj:
                self._handle_task_completion_or_failure(completed_task_obj)
            self.session_state["current_task_id"] = None
            self.session_state["investigation_context"] = ""
            self.session_state["current_task_retries"] = 0

        # ... (Step 5: Update Context and State - unchanged) ...
        if step_status in ["processing", "error_retry"]:
            context_limit = config.CONTEXT_TRUNCATION_LIMIT
            if len(current_context) > context_limit:
                current_context = f"(Truncated)\n...\n{current_context[len(current_context) - context_limit:]}"
            self.session_state["investigation_context"] = current_context
            self.session_state[f"task_{task.id}_step"] = step_num
        elif step_status in ["completed", "failed"]:
            self.session_state.pop(f"task_{task.id}_step", None)
            with self._state_lock:
                self.session_state["user_suggestion_move_on_pending"] = False

        step_duration = time.time() - step_start_time
        step_log.append(
            f"Step {step_num} duration: {step_duration:.2f}s. Step Status: {step_status}"
        )
        step_details_for_history["log_snippet"] = "\n".join(step_log[-5:])

        with self._state_lock:
            self.session_state["last_step_details"].append(step_details_for_history)
            self._ui_update_state["step_history"] = list(
                self.session_state["last_step_details"]
            )

        self.save_session_state()
        final_task_status_obj = self.task_queue.get_task(task.id)
        final_task_status = (
            final_task_status_obj.status if final_task_status_obj else "unknown"
        )

        dependent_tasks_list = []
        if self.session_state.get("current_task_id"):
            dependent_tasks = self.task_queue.get_dependent_tasks(
                self.session_state["current_task_id"]
            )
            dependent_tasks_list = [
                {"id": dt.id, "description": dt.description} for dt in dependent_tasks
            ]

        ui_memory_query = f"Memories relevant to just completed step {step_num} of task {task.id}. Action taken: {action_type}. Status: {step_status}"
        recent_memories_for_ui, _ = self.memory.retrieve_and_rerank_memories(
            query=ui_memory_query,
            task_description=task.description,
            context=current_context,
            identity_statement=self.identity_statement,
            n_final=5,
        )

        # Return updated state structure for UI
        return {
            "status": final_task_status,
            "log": "\n".join(step_log),
            "current_task_id": self.session_state.get("current_task_id"),
            "current_task_desc": (
                task.description
                if self.session_state.get("current_task_id")
                else "None"
            ),
            "thinking": thinking_to_store,
            "dependent_tasks": dependent_tasks_list,
            "last_action_type": action_type,
            "last_tool_results": tool_results_for_ui,  # Pass the structured tool output
            "recent_memories": recent_memories_for_ui,
            "last_web_content": self.session_state.get(
                "last_web_browse_content", "(No recent web browse)"
            ),
            "final_answer": final_answer_text,
            "step_history": list(self.session_state.get("last_step_details", [])),
        }

    def _handle_task_completion_or_failure(self, task: Task):
        # ... (unchanged) ...
        log.info(
            f"Handling completion/failure for task {task.id} (Status: {task.status})"
        )
        if config.ENABLE_MEMORY_SUMMARIZATION:
            try:
                self._summarize_and_prune_task_memories(task)
            except Exception as e:
                log.exception(
                    f"Error during memory summarization for task {task.id}: {e}"
                )
        if self._is_running.is_set():
            reason = f"{task.status.capitalize()} Task: {task.description[:50]}..."
            self._revise_identity_statement(reason)
        else:
            log.info(
                f"Agent paused, skipping identity revision after {task.status} task."
            )

    def process_one_step(self) -> Dict[str, Any]:
        # ... (unchanged) ...
        log.info("Processing one autonomous step...")
        step_result_log = ["Attempting one step..."]
        default_state = {
            "status": "idle",
            "log": "Idle.",
            "current_task_id": None,
            "current_task_desc": "None",
            "thinking": "(Agent Idle)",
            "dependent_tasks": [],
            "last_action_type": None,
            "last_tool_results": None,
            "recent_memories": [],
            "last_web_content": self.session_state.get(
                "last_web_browse_content", "(None)"
            ),
            "final_answer": None,
            "step_history": list(self.session_state.get("last_step_details", [])),
        }
        try:
            task_id_to_process = self.session_state.get("current_task_id")
            task: Optional[Task] = None
            if task_id_to_process:
                task = self.task_queue.get_task(task_id_to_process)
                if not task or task.status not in ["in_progress", "pending"]:
                    log.warning(
                        f"Task {task_id_to_process} from state invalid/finished. Resetting."
                    )
                    step_result_log.append(
                        f"Task {task_id_to_process} invalid/finished."
                    )
                    task = None
                    self.session_state["current_task_id"] = None
                    self.session_state["investigation_context"] = ""
                    self.session_state["current_task_retries"] = 0
                    with self._state_lock:
                        self.session_state["user_suggestion_move_on_pending"] = False
                    state_to_update = default_state.copy()
                    state_to_update["log"] = "\n".join(step_result_log)
                    self._update_ui_state(**state_to_update)
                    self.save_session_state()
            if not task:
                task = self.task_queue.get_next_task()
                if task:
                    log.info(f"Starting task: {task.id} - '{task.description[:60]}...'")
                    step_result_log.append(
                        f"Starting task {task.id}: {task.description[:60]}..."
                    )
                    self.task_queue.update_task(task.id, "in_progress")
                    self.session_state["current_task_id"] = task.id
                    self.session_state["investigation_context"] = (
                        f"Objective: {task.description}\n"
                    )
                    self.session_state.pop(f"task_{task.id}_step", None)
                    self.session_state["current_task_retries"] = 0
                    with self._state_lock:
                        self.session_state["user_suggestion_move_on_pending"] = False
                    self.save_session_state()
                    task = self.task_queue.get_task(task.id)
                    dependent_tasks = self.task_queue.get_dependent_tasks(task.id)
                    dependent_tasks_list = [
                        {"id": dt.id, "description": dt.description}
                        for dt in dependent_tasks
                    ]
                    self._update_ui_state(
                        status="starting",
                        log="\n".join(step_result_log),
                        current_task_id=task.id,
                        current_task_desc=task.description,
                        thinking="(Starting task...)",
                        dependent_tasks=dependent_tasks_list,
                        last_action_type=None,
                        last_tool_results=None,
                        recent_memories=[],
                        final_answer=None,
                        step_history=list(
                            self.session_state.get("last_step_details", [])
                        ),
                    )
                    return self.get_ui_update_state()
                else:
                    log.info("No runnable tasks found.")
                    step_result_log.append("No runnable tasks found.")
                    if not self.session_state.get("current_task_id"):
                        log.info("Agent idle, considering generating tasks...")
                        generated_desc = self.generate_new_tasks(
                            max_new_tasks=3, trigger_context="idle"
                        )
                        if generated_desc:
                            step_result_log.append(
                                f"Generated idle task: {generated_desc[:60]}..."
                            )
                        else:
                            step_result_log.append("No new idle tasks generated.")
                    state_to_update = default_state.copy()
                    state_to_update["log"] = "\n".join(step_result_log)
                    self._update_ui_state(**state_to_update)
                    return self.get_ui_update_state()
            if task and task.status == "in_progress":
                step_result = self._execute_step(task)
                self._update_ui_state(**step_result)
                return self.get_ui_update_state()
            elif task:
                log.warning(
                    f"Task {task.id} found but not in expected 'in_progress' state (State: {task.status}). Resetting."
                )
                step_result_log.append(
                    f"Task {task.id} state unexpected: {task.status}. Resetting."
                )
                self.session_state["current_task_id"] = None
                self.session_state["investigation_context"] = ""
                self.session_state["current_task_retries"] = 0
                with self._state_lock:
                    self.session_state["user_suggestion_move_on_pending"] = False
                self.save_session_state()
                state_to_update = default_state.copy()
                state_to_update["log"] = "\n".join(step_result_log)
                self._update_ui_state(**state_to_update)
                return self.get_ui_update_state()
            else:
                log.error("Task became None unexpectedly before step execution.")
                step_result_log.append("[ERROR] Task lost before execution.")
                self._update_ui_state(status="error", log="\n".join(step_result_log))
                return self.get_ui_update_state()
        except Exception as e:
            log.exception("CRITICAL Error during process_one_step loop")
            step_result_log.append(f"[CRITICAL ERROR]: {e}")
            current_task_id = self.session_state.get("current_task_id")
            if current_task_id:
                log.error(f"Failing task {current_task_id} due to critical loop error.")
                fail_reason = f"Critical loop error: {e}"
                self.task_queue.update_task(
                    current_task_id, "failed", reflections=fail_reason
                )
                failed_task_obj = self.task_queue.get_task(current_task_id)
                if failed_task_obj:
                    self._handle_task_completion_or_failure(failed_task_obj)
                self.session_state["current_task_id"] = None
                self.session_state["investigation_context"] = ""
                self.session_state["current_task_retries"] = 0
                with self._state_lock:
                    self.session_state["user_suggestion_move_on_pending"] = False
                self.save_session_state()
            state_to_update = default_state.copy()
            state_to_update["status"] = "critical_error"
            state_to_update["log"] = "\n".join(step_result_log)
            self._update_ui_state(**state_to_update)
            return self.get_ui_update_state()

    def generate_new_tasks(
        self,
        max_new_tasks: int = 3,
        last_user_message: Optional[str] = None,
        last_assistant_response: Optional[str] = None,
        trigger_context: str = "unknown",
    ) -> Optional[str]:
        # ... (unchanged) ...
        log.info(
            f"\n--- Attempting to Generate New Tasks (Trigger: {trigger_context}) ---"
        )
        memory_is_empty = False
        memory_count = 0
        try:
            memory_count = self.memory.collection.count()
            memory_is_empty = memory_count == 0
            log.info(
                f"Memory collection count: {memory_count}. Is empty: {memory_is_empty}"
            )
        except Exception as e:
            log.error(f"Failed to get memory count: {e}. Assuming not empty.")
            memory_is_empty = False
        use_initial_creative_prompt = memory_is_empty and trigger_context == "idle"
        max_tasks_to_generate = (
            config.INITIAL_NEW_TASK_N if use_initial_creative_prompt else max_new_tasks
        )
        prompt_template = (
            prompts.INITIAL_CREATIVE_TASK_GENERATION_PROMPT
            if use_initial_creative_prompt
            else prompts.GENERATE_NEW_TASKS_PROMPT
        )
        log.info(
            f"Using {'Initial Creative' if use_initial_creative_prompt else 'Standard'} Task Generation Prompt. Max tasks: {max_tasks_to_generate}"
        )
        prompt_vars = {}
        if use_initial_creative_prompt:
            prompt_vars = {
                "identity_statement": self.identity_statement,
                "tool_desc": self.get_available_tools_description(),
                "max_new_tasks": max_tasks_to_generate,
            }
            context_query = (
                "Initial run with empty memory. Generate creative starting tasks."
            )
        else:
            context_source = ""
            context_query = ""
            mem_query = ""
            critical_evaluation_instruction = ""
            if (
                trigger_context == "chat"
                and last_user_message
                and last_assistant_response
            ):
                context_source = "last chat interaction"
                context_query = f"Last User: {last_user_message}\nLast Assistant: {last_assistant_response}"
                mem_query = f"Context relevant to last chat: {last_user_message}"
                critical_evaluation_instruction = "\n**Critically Evaluate Need:** Based *specifically* on the **Last Interaction**, is a background task *truly necessary*? Output `[]` if not."
            else:
                context_source = "general agent state"
                context_query = "General status. Consider logical follow-up/exploration based on completed tasks, idle state, and my identity."
                mem_query = "Recent activities, conclusions, errors, reflections, summaries, identity revisions."
                critical_evaluation_instruction = "\n**Critically Evaluate Need:** Are new tasks genuinely needed for exploration/follow-up, consistent with identity? Output `[]` if not."
            log.info(
                f"Retrieving context for task generation (Source: {context_source})..."
            )
            recent_mems, _ = self.memory.retrieve_and_rerank_memories(
                query=mem_query,
                task_description="Task Generation Context",
                context=context_query,
                identity_statement=self.identity_statement,
                n_results=config.MEMORY_COUNT_NEW_TASKS * 2,
                n_final=config.MEMORY_COUNT_NEW_TASKS,
            )
            mem_summary_list = []
            for m in recent_mems:
                relative_time = format_relative_time(m["metadata"].get("timestamp"))
                mem_type = m["metadata"].get("type", "mem")
                snippet = m["content"][:150].strip().replace("\n", " ")
                mem_summary_list.append(f"- [{relative_time}] {mem_type}: {snippet}...")
            mem_summary = "\n".join(mem_summary_list) if mem_summary_list else "None"
            existing_tasks_info = [
                {"id": t.id, "description": t.description, "status": t.status}
                for t in self.task_queue.tasks.values()
            ]
            active_tasks_summary = (
                "\n".join(
                    [
                        f"- ID: {t['id']} (Status: {t['status']}) Desc: {t['description'][:100]}..."
                        for t in existing_tasks_info
                        if t["status"] in ["pending", "in_progress"]
                    ]
                )
                or "None"
            )
            completed_failed_summary = "\n".join(
                [
                    f"- ID: {t['id']} (Status: {t['status']}) Desc: {t['description'][:100]}..."
                    for t in existing_tasks_info
                    if t["status"] in ["completed", "failed"]
                ]
            )[-1000:]
            prompt_vars = {
                "identity_statement": self.identity_statement,
                "context_query": context_query,
                "mem_summary": mem_summary,
                "active_tasks_summary": active_tasks_summary,
                "completed_failed_summary": completed_failed_summary,
                "critical_evaluation_instruction": critical_evaluation_instruction,
                "max_new_tasks": max_tasks_to_generate,
            }
        if use_initial_creative_prompt:
            prompt_vars.setdefault("context_query", "N/A (Initial Run)")
            prompt_vars.setdefault("mem_summary", "None (Initial Run)")
            prompt_vars.setdefault("active_tasks_summary", "None (Initial Run)")
            prompt_vars.setdefault("completed_failed_summary", "None (Initial Run)")
            prompt_vars.setdefault(
                "critical_evaluation_instruction", "N/A (Initial Run)"
            )
            prompt_vars.setdefault("tool_desc", self.get_available_tools_description())
        prompt = prompt_template.format(**prompt_vars)
        log.info(
            f"Asking {self.ollama_chat_model} to generate up to {max_tasks_to_generate} new tasks..."
        )
        llm_response = call_ollama_api(
            prompt, self.ollama_chat_model, self.ollama_base_url, timeout=180
        )
        if not llm_response:
            log.error("LLM failed task gen.")
            return None
        first_task_desc_added = None
        new_tasks_added = 0
        try:
            llm_response = re.sub(r"^```json\s*", "", llm_response, flags=re.I | re.M)
            llm_response = re.sub(r"\s*```$", "", llm_response, flags=re.M).strip()
            list_start = llm_response.find("[")
            list_end = llm_response.rfind("]")
            if list_start == -1 or list_end == -1 or list_end < list_start:
                if (
                    "no new tasks" in llm_response.lower()
                    or llm_response.strip() == "[]"
                ):
                    log.info("LLM: no tasks needed.")
                    return None
                else:
                    raise json.JSONDecodeError(
                        f"JSON list '[]' not found: {llm_response}", llm_response, 0
                    )
            json_str = llm_response[list_start : list_end + 1]
            suggested_tasks = json.loads(json_str)
            if not isinstance(suggested_tasks, list):
                log.warning(f"LLM task gen not list: {suggested_tasks}")
                return None
            if not suggested_tasks:
                log.info("LLM suggested no new tasks.")
                return None
            log.info(f"LLM suggested {len(suggested_tasks)} tasks. Validating...")
            current_task_ids_in_batch = set()
            existing_tasks_info = [
                {"id": t.id, "description": t.description, "status": t.status}
                for t in self.task_queue.tasks.values()
            ]
            active_task_descriptions = {
                t["description"].strip().lower()
                for t in existing_tasks_info
                if t["status"] in ["pending", "in_progress"]
            }
            valid_existing_task_ids = set(t["id"] for t in existing_tasks_info)
            for task_data in suggested_tasks:
                if not isinstance(task_data, dict):
                    continue
                description = task_data.get("description")
                if (
                    not description
                    or not isinstance(description, str)
                    or not description.strip()
                ):
                    continue
                description = description.strip()
                if description.lower() in active_task_descriptions:
                    log.warning(f"Skipping duplicate task: '{description[:80]}...'")
                    continue
                priority = task_data.get(
                    "priority", 3 if use_initial_creative_prompt else 5
                )
                try:
                    priority = max(1, min(10, int(priority)))
                except:
                    priority = 3 if use_initial_creative_prompt else 5
                dependencies_raw = task_data.get("depends_on")
                validated_dependencies = []
                if isinstance(dependencies_raw, list):
                    for dep_id in dependencies_raw:
                        dep_id_str = str(dep_id).strip()
                        if (
                            dep_id_str in valid_existing_task_ids
                            or dep_id_str in current_task_ids_in_batch
                        ):
                            validated_dependencies.append(dep_id_str)
                        else:
                            log.warning(
                                f"Dependency '{dep_id_str}' for new task not found. Ignoring."
                            )
                new_task = Task(
                    description, priority, depends_on=validated_dependencies or None
                )
                new_task_id = self.task_queue.add_task(new_task)
                if new_task_id:
                    if new_tasks_added == 0:
                        first_task_desc_added = description
                    new_tasks_added += 1
                    active_task_descriptions.add(description.lower())
                    valid_existing_task_ids.add(new_task_id)
                    current_task_ids_in_batch.add(new_task_id)
                    log.info(
                        f"Added Task {new_task_id}: '{description[:60]}...' (Prio: {priority}, Depends: {validated_dependencies})"
                    )
                if new_tasks_added >= max_tasks_to_generate:
                    break
        except json.JSONDecodeError as e:
            log.error(
                f"Failed JSON parse task gen: {e}\nLLM Resp:\n{llm_response}\n---"
            )
            return None
        except Exception as e:
            log.exception(f"Unexpected error task gen: {e}")
            return None
        log.info(f"Finished Task Generation: Added {new_tasks_added} new tasks.")
        return first_task_desc_added

    def _autonomous_loop(self, initial_delay: float = 2.0, step_delay: float = 5.0):
        # ... (unchanged) ...
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
            log.debug("Autonomous loop: Processing step...")
            step_start = time.monotonic()
            try:
                self.process_one_step()
            except Exception as e:
                log.exception(f"CRITICAL UNHANDLED ERROR in autonomous loop iteration.")
                self._update_ui_state(
                    status="critical_error", log=f"CRITICAL LOOP ERROR: {e}"
                )
                time.sleep(step_delay * 5)
            step_duration = time.monotonic() - step_start
            remaining_delay = max(0, step_delay - step_duration)
            log.debug(
                f"Step took {step_duration:.2f}s. Sleeping {remaining_delay:.2f}s."
            )
            time.sleep(remaining_delay)
        log.info("Background agent loop shutting down.")

    def start_autonomous_loop(self):
        # ... (unchanged) ...
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
            self._is_running.set()
            self._update_ui_state(status="running", log="Agent started.")
            self._agent_thread = threading.Thread(
                target=self._autonomous_loop,
                args=(2.0, config.UI_UPDATE_INTERVAL * 4),
                daemon=True,
            )  # Adjust step delay based on UI interval
            self._agent_thread.start()

    def pause_autonomous_loop(self):
        # ... (unchanged) ...
        if self._is_running.is_set():
            log.info("Pausing agent loop...")
            self._is_running.clear()
            self._update_ui_state(status="paused", log="Agent loop paused.")
        else:
            log.info("Agent loop is already paused.")

    def shutdown(self):
        # ... (unchanged) ...
        log.info("Shutdown requested.")
        self._shutdown_request.set()
        self._is_running.clear()
        if self._agent_thread and self._agent_thread.is_alive():
            log.info("Waiting for agent thread join...")
            self._agent_thread.join(timeout=15)
            if self._agent_thread.is_alive():
                log.warning("Agent thread didn't join cleanly.")
            else:
                log.info("Agent thread joined.")
        else:
            log.info("Agent thread not running/already joined.")
        self._update_ui_state(status="shutdown", log="Agent shut down.")
        self.save_session_state()
        log.info("Shutdown complete.")

    def add_self_reflection(
        self, reflection: str, reflection_type: str = "self_reflection"
    ):
        # ... (unchanged) ...
        if not reflection or not isinstance(reflection, str) or not reflection.strip():
            return None
        if self._is_running.is_set():
            log.info(f"Adding {reflection_type} to memory...")
            return self.memory.add_memory(reflection, {"type": reflection_type})
        else:
            log.info(f"Agent paused, skipping adding {reflection_type} to memory.")
            return None

    def generate_and_add_session_reflection(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        completed_count: int,
        processed_count: int,
    ):
        # ... (unchanged) ...
        duration_minutes = (end - start).total_seconds() / 60
        log.info("Retrieving context for session reflection...")
        mem_query = "Summary session activities, task summaries, errors, outcomes, identity statements/revisions."
        recent_mems, _ = self.memory.retrieve_and_rerank_memories(
            query=mem_query,
            task_description="Session Reflection",
            context="End of work session",
            identity_statement=self.identity_statement,
            n_results=config.MEMORY_COUNT_REFLECTIONS * 2,
            n_final=config.MEMORY_COUNT_REFLECTIONS,
        )
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
        log.info(f"Asking {self.ollama_chat_model} for session reflection...")
        reflection = call_ollama_api(
            prompt, self.ollama_chat_model, self.ollama_base_url, timeout=120
        )
        if reflection and reflection.strip():
            print(f"\n--- Session Reflection ---\n{reflection}\n------")
            self.add_self_reflection(reflection, "session_reflection")
        else:
            log.warning("Failed to generate session reflection.")

    def get_agent_dashboard_state(self) -> Dict[str, Any]:
        # ... (unchanged) ...
        log.debug("Gathering dashboard state...")
        try:
            tasks_structured = self.task_queue.get_all_tasks_structured()
            memory_summary = self.memory.get_memory_summary()
            completed_failed_tasks = tasks_structured.get(
                "completed", []
            ) + tasks_structured.get("failed", [])
            summary_items = [
                f"- {m_type}: {count}" for m_type, count in memory_summary.items()
            ]
            if "agent_explicit_memory_write" in memory_summary:
                summary_items.append(
                    f"- agent_explicit_memory_write: {memory_summary['agent_explicit_memory_write']}"
                )
            memory_summary_str = (
                "**Memory Summary (by Type):**\n" + "\n".join(sorted(summary_items))
                if summary_items
                else "No memories found."
            )
            return {
                "identity_statement": self.identity_statement,
                "pending_tasks": tasks_structured.get("pending", []),
                "in_progress_tasks": tasks_structured.get("in_progress", []),
                "completed_tasks": tasks_structured.get("completed", []),
                "failed_tasks": tasks_structured.get("failed", []),
                "completed_failed_tasks_data": completed_failed_tasks,
                "memory_summary": memory_summary_str,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        except Exception as e:
            log.exception("Error gathering agent dashboard state")
            return {
                "identity_statement": "Error fetching state",
                "pending_tasks": [],
                "in_progress_tasks": [],
                "completed_tasks": [],
                "failed_tasks": [],
                "completed_failed_tasks_data": [],
                "memory_summary": f"Error: {e}",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }

    def get_formatted_memories_for_task(self, task_id: str) -> List[Dict[str, Any]]:
        # ... (unchanged) ...
        if not task_id:
            return []
        log.debug(f"Getting memories for task ID: {task_id}")
        memories = self.memory.get_memories_by_metadata(
            filter_dict={"task_id": task_id}, limit=100
        )
        formatted = []
        for mem in memories:
            metadata = mem.get("metadata", {})
            formatted.append(
                {
                    "Timestamp": metadata.get("timestamp", "N/A"),
                    "Type": metadata.get("type", "N/A"),
                    "Content Snippet": (
                        mem.get("content", "")[:200] + "..."
                        if mem.get("content")
                        else "N/A"
                    ),
                    "ID": mem.get("id", "N/A"),
                }
            )
        return sorted(formatted, key=lambda x: x.get("Timestamp", "0"))

    def get_formatted_general_memories(self) -> List[Dict[str, Any]]:
        # ... (unchanged) ...
        log.debug("Getting general memories...")
        memories = self.memory.get_general_memories(limit=50)
        formatted = []
        for mem in memories:
            metadata = mem.get("metadata", {})
            formatted.append(
                {
                    "Timestamp": metadata.get("timestamp", "N/A"),
                    "Type": metadata.get("type", "N/A"),
                    "Content Snippet": (
                        mem.get("content", "")[:200] + "..."
                        if mem.get("content")
                        else "N/A"
                    ),
                    "ID": mem.get("id", "N/A"),
                }
            )
        return sorted(formatted, key=lambda x: x.get("Timestamp", "0"), reverse=True)
