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
from typing import List, Dict, Any, Optional, Tuple # <<<--- REMOVED Tuple, re import if no longer needed
from collections import Counter

# --- Project Imports ---
import config
from data_structures import Task
from task_manager import TaskQueue
from memory import AgentMemory
from tools import load_tools
from utils import call_ollama_api, format_relative_time # format_relative_time might still be used elsewhere
import chromadb

# --- Logging Setup ---
import logging
log = logging.getLogger("AGENT")

class AutonomousAgent:
    def __init__(self, memory_collection: Optional[chromadb.Collection] = None):
        log.info("Initializing AutonomousAgent...")
        self.task_queue = TaskQueue()
        if memory_collection is None:
             raise ValueError("Memory collection is required.")
        # --- MODIFIED: Pass Ollama details to AgentMemory ---
        self.memory = AgentMemory(
            collection=memory_collection,
            ollama_chat_model=config.OLLAMA_CHAT_MODEL,
            ollama_base_url=config.OLLAMA_BASE_URL
        )
        self.tools = load_tools()
        self.ollama_base_url = config.OLLAMA_BASE_URL
        self.ollama_chat_model = config.OLLAMA_CHAT_MODEL
        self.ollama_embed_model = config.OLLAMA_EMBED_MODEL # Still needed for info/debug?
        self.session_state_path = config.AGENT_STATE_PATH
        self.qlora_dataset_path = config.QLORA_DATASET_PATH
        self.identity_statement: str = config.INITIAL_IDENTITY_STATEMENT

        self.session_state = {
            "current_task_id": None, "investigation_context": "", "current_task_retries": 0,
            "last_checkpoint": None, "last_web_browse_content": None, "identity_statement": self.identity_statement,
            "user_suggestion_move_on_pending": False
        }
        self.load_session_state()

        self._is_running = threading.Event()
        self._shutdown_request = threading.Event()
        self._agent_thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        self._ui_update_state: Dict[str, Any] = {
            "status": "paused", "log": "Agent paused.", "current_task_id": self.session_state.get("current_task_id"),
            "current_task_desc": "N/A", "recent_memories": [], "last_web_content": self.session_state.get("last_web_browse_content", "(None)"),
            "final_answer": None, "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
         }

        if self.session_state.get("current_task_id"):
             task = self.task_queue.get_task(self.session_state["current_task_id"])
             if task: self._ui_update_state["current_task_desc"] = task.description

        if self.identity_statement == config.INITIAL_IDENTITY_STATEMENT:
             log.info("Performing initial identity revision...")
             self._revise_identity_statement("Initialization / Session Start")
             self.save_session_state()
        else:
             log.info("Loaded existing identity statement. Skipping initial revision.")

        log.info("Agent Initialized.")

    # --- Methods from previous implementation (unchanged unless noted) ---
    def _revise_identity_statement(self, reason: str):
        log.info(f"Revising identity statement. Reason: {reason}")
        mem_query = f"Recent task summaries, self-reflections, session reflections, errors, key accomplishments, or notable interactions relevant to understanding my evolution, capabilities, and purpose."
        # --- MODIFIED: Call memory's method, pass identity ---
        relevant_memories, _ = self.memory.retrieve_and_rerank_memories(
            query=mem_query,
            task_description="Reflecting on identity",
            context=f"Reason: {reason}",
            identity_statement=self.identity_statement, # <<<--- PASS IDENTITY
            n_results=config.IDENTITY_REVISION_MEMORY_COUNT,
            n_final=8
        )

        memory_context_list = []
        for m in relevant_memories:
            relative_time = format_relative_time(m['metadata'].get('timestamp')) # Still use util here
            mem_type = m['metadata'].get('type', 'N/A')
            content_snippet = m['content'][:300]
            memory_context_list.append(f"- [Memory - {relative_time}] (Type: {mem_type}):\n  {content_snippet}...")
        memory_context = "\n\n".join(memory_context_list) if memory_context_list else "No specific memories selected for this revision."

        prompt = f"""You are an AI agent reflecting on your identity. Your goal is to revise your personal identity statement based on your recent experiences and purpose.

**Current Identity Statement:**
{self.identity_statement}

**Reason for Revision:**
{reason}

**Relevant Recent Memories/Experiences (Consider recency indicated by '[X time ago]'):**
{memory_context}

**Your Task:** Based *only* on the information provided, write a revised, concise, first-person identity statement (2-4 sentences).
**Guidelines:** Reflect growth, maintain cohesion, focus on purpose/capabilities, be concise. Consider the **recency** of memories when weighing their impact.
**Format:** Output *only* the revised identity statement text, starting with "I am..." or similar.

**Output Example:**
I am an AI assistant focused on research tasks. I've recently improved my web browsing skills and am learning to handle complex multi-step analyses more effectively, though I sometimes struggle with ambiguous instructions.
"""
        log.info(f"Asking {self.ollama_chat_model} to revise identity statement...")
        revised_statement_text = call_ollama_api(prompt, self.ollama_chat_model, self.ollama_base_url, timeout=150)
        if revised_statement_text and revised_statement_text.strip() and len(revised_statement_text) > 20:
            old_statement = self.identity_statement
            with self._state_lock:
                self.identity_statement = revised_statement_text.strip()
                self.session_state["identity_statement"] = self.identity_statement
            log.info(f"Identity statement revised:\nOld: {old_statement}\nNew: {self.identity_statement}")
            if self._is_running.is_set():
                self.memory.add_memory(f"Identity Statement Updated (Reason: {reason}):\n{self.identity_statement}", {"type": "identity_revision", "reason": reason})
            else:
                log.info("Agent paused, skipping memory add for identity revision.")
            self.save_session_state()
        else:
            log.warning(f"Failed to get a valid revised identity statement from LLM (Response: '{revised_statement_text}'). Keeping current.")
            if self._is_running.is_set():
                self.memory.add_memory(f"Identity statement revision failed (Reason: {reason}). LLM response insufficient.", {"type": "identity_revision_failed", "reason": reason})
            else:
                log.info("Agent paused, skipping memory add for failed identity revision.")


    def _update_ui_state(self, **kwargs):
        with self._state_lock:
            self._ui_update_state["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            for key, value in kwargs.items():
                if key in self._ui_update_state: self._ui_update_state[key] = value
                else: log.warning(f"Attempted to update unknown UI state key: {key}")

    def get_ui_update_state(self) -> Dict[str, Any]:
        with self._state_lock: return self._ui_update_state.copy()

    def load_session_state(self):
        # ... (implementation unchanged) ...
        if os.path.exists(self.session_state_path):
            try:
                with open(self.session_state_path, "r", encoding='utf-8') as f: content = f.read()
                if not content or not content.strip(): log.info(f"Session state file '{self.session_state_path}' empty."); return
                loaded = json.loads(content)
                if isinstance(loaded, dict):
                    loaded.setdefault("last_web_browse_content", None)
                    loaded.setdefault("current_task_retries", 0)
                    loaded.setdefault("user_suggestion_move_on_pending", False)
                    loaded_identity = loaded.get("identity_statement", config.INITIAL_IDENTITY_STATEMENT)
                    if not loaded_identity: loaded_identity = config.INITIAL_IDENTITY_STATEMENT

                    self.session_state.update(loaded)
                    self.identity_statement = loaded_identity
                    self.session_state["identity_statement"] = self.identity_statement

                    log.info(f"Loaded session state from {self.session_state_path}.")
                    log.info(f"  Loaded Identity: {self.identity_statement[:100]}...")
                    log.info(f"  User Suggestion Pending: {self.session_state.get('user_suggestion_move_on_pending')}")
                else: log.warning(f"Session state file '{self.session_state_path}' invalid format.")
            except json.JSONDecodeError as e: log.warning(f"Failed loading session state JSON '{self.session_state_path}': {e}.")
            except Exception as e: log.warning(f"Failed loading session state '{self.session_state_path}': {e}.")
        else:
            log.info(f"Session state file '{self.session_state_path}' not found. Using defaults.")
            self.session_state["identity_statement"] = self.identity_statement
            self.session_state["user_suggestion_move_on_pending"] = False


    def save_session_state(self):
        # ... (implementation unchanged) ...
        try:
            with self._state_lock:
                self.session_state["last_checkpoint"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                self.session_state["identity_statement"] = self.identity_statement
            with open(self.session_state_path, "w", encoding='utf-8') as f: json.dump(self.session_state, f, indent=2, ensure_ascii=False)
        except Exception as e: log.error(f"Error saving session state to '{self.session_state_path}': {e}")


    def handle_user_suggestion_move_on(self) -> str:
        # ... (implementation unchanged) ...
        feedback = "Suggestion ignored: Agent not initialized."
        current_task_id = self.session_state.get("current_task_id")
        with self._state_lock:
            if self._is_running.is_set():
                if current_task_id:
                    log.info(f"User suggested moving on from task {current_task_id}. Setting flag.")
                    self.session_state["user_suggestion_move_on_pending"] = True
                    feedback = f"Suggestion noted for current task ({current_task_id}). Agent will consider wrapping up."
                    self.save_session_state()
                else:
                    log.info("User suggested task change, but agent is idle.")
                    feedback = "Suggestion ignored: Agent is currently idle (no active task)."
            else:
                log.info(f"Agent paused, ignoring suggestion to move on from task {current_task_id or 'N/A'} (flag not set).")
                feedback = f"Suggestion noted, but agent is paused. Flag not set. (Task: {current_task_id or 'N/A'})."
        return feedback

    def create_task(self, description: str, priority: int = 1, depends_on: Optional[List[str]] = None) -> Optional[str]:
        # ... (implementation unchanged) ...
        return self.task_queue.add_task(Task(description, priority, depends_on=depends_on))

    def get_available_tools_description(self) -> str:
        # ... (implementation unchanged) ...
        if not self.tools: return "No tools available."
        active_tools = [];
        for name, tool in self.tools.items():
            is_active = True
            if name == "web_search" and not config.SEARXNG_BASE_URL: is_active = False
            if name == "memory_retrieval" and not self.memory: is_active = False
            if is_active: active_tools.append(f"- Name: {name}\n  Description: {tool.description}")
        if not active_tools: return "No tools currently active or available (check configuration)."
        return "\n".join(active_tools)

    def generate_thinking(
        self,
        task_description: str,
        context: str = "",
        tool_results: Optional[Dict[str, Any]] = None,
        user_suggested_move_on: bool = False
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]: # <<<--- Return type changed back to Optional[Dict]
        tool_desc = self.get_available_tools_description()
        memory_query = f"Info relevant to next step? Goal: {context[-500:]}\nTask: {task_description}\nConsider recent chat, user suggestions, errors, or provided info if relevant."
        if tool_results: memory_query += f"\nLast result summary: {str(tool_results)[:200]}"

        # --- MODIFIED: Call memory's method, pass identity ---
        relevant_memories, separated_suggestion = self.memory.retrieve_and_rerank_memories(
            query=memory_query,
            task_description=task_description,
            context=context,
            identity_statement=self.identity_statement, # <<<--- PASS IDENTITY
            n_results=12,
            n_final=5
        )

        # --- Format memories with relative time ---
        user_provided_info_content = []; other_memories_context_list = []
        for mem in relevant_memories:
            mem_type = mem.get('metadata', {}).get('type')
            relative_time = format_relative_time(mem['metadata'].get('timestamp')) # Use util
            dist_str = f"{mem.get('distance', -1.0):.3f}" if mem.get('distance') is not None else "N/A"
            mem_content = f"[Memory - {relative_time}] (Type: {mem_type}, Dist: {dist_str}, ID: {mem.get('id', 'N/A')}):\n{mem['content']}"
            if mem_type == 'user_provided_info_for_task':
                 user_provided_info_content.append(mem_content)
            else:
                 other_memories_context_list.append(mem_content)

        memory_context_str = "\n\n".join(other_memories_context_list) if other_memories_context_list else "No other relevant memories selected."
        user_provided_info_str = "\n---\n".join(user_provided_info_content) if user_provided_info_content else None

        prompt = f"""You are an autonomous AI agent. Your primary goal is to achieve the **Overall Task** by deciding the single best next step, while considering your identity and user interactions.

**Your Current Identity:**
{self.identity_statement}

**Overall Task:**
{task_description}

**Available Tools:**
{tool_desc}

**Relevant Memories (Re-ranked, consider recency indicated by '[X time ago]', novelty):**
{memory_context_str}"""

        if user_suggested_move_on:
            prompt += f"\n\n**USER SUGGESTION PENDING:** The user suggested considering wrapping up this task soon."
        if user_provided_info_str:
             prompt += f"\n\n**User Provided Information (Consider for current step):**\n{user_provided_info_str}\n"

        prompt += f"""
**Current Investigation Context (History - IMPORTANT: May contain previous errors):**
{context if context else "First step for this task."}\n"""
        if tool_results: prompt += f"\n**Results from Last Action:**\nTool: {tool_results.get('tool_name', 'Unknown')}\nResult:\n```json\n{json.dumps(tool_results.get('result', {}), indent=2, ensure_ascii=False)}\n```\n"
        else: prompt += "\n**Results from Last Action:**\nNone.\n"
        prompt += """
**Your Task Now:**
1.  **Analyze:** Review ALL provided information (Identity, Task, Tools, Memories - including recency/novelty, User Provided Info, Context, Last Result).
2.  **Reason:** Determine the single best action *right now* to make progress towards the **Overall Task**, consistent with your identity.
3.  **User Input Handling:**
    *   If **User Provided Information** exists, incorporate it into your plan for the *current* step if relevant.
    *   If **USER SUGGESTION PENDING** exists: Acknowledge it in your thinking. *Strongly consider* concluding the task in this step or the next using `final_answer` (e.g., summarizing findings so far) **unless** critical sub-steps are still absolutely required to meet the core task objective. Do not stop abruptly if essential work remains unfinished.
4.  **CRITICAL: Handle Previous Errors:** If the **Context** mentions an error from a previous step, focus on a **RECOVERY STRATEGY** (e.g., retry tool, different tool, fix params, refine search, use memory_retrieval tool) before considering giving up or moving on based *only* on the error.
5.  **Choose Action:** Decide between `use_tool` or `final_answer`. Use `final_answer` only when the *original* task goal is fully met, or if reasonably wrapping up based on sufficient progress and the user suggestion.
**Output Format (Strict JSON-like structure):**
THINKING:
<Your reasoning process. Explain analysis, memory consideration (recency/novelty), how you're handling user suggestion/info (if any), your plan (especially recovery or conclusion strategy), and alignment with identity/task.>
NEXT_ACTION: <"use_tool" or "final_answer">
If NEXT_ACTION is "use_tool":
TOOL: <name_of_tool_from_list>
PARAMETERS: <{{valid_json_object}}>
If NEXT_ACTION is "final_answer":
ANSWER: <Your complete answer, or a summary if concluding early based on suggestion.>
REFLECTIONS: <Optional thoughts.>
**Critical Formatting Reminders:** Start *immediately* with "THINKING:". Structure must be exact. `PARAMETERS` must be valid JSON. Tool name must be one of the available tools.
"""
        log.info(f"Asking {self.ollama_chat_model} for next action (with identity, relative time, suggestion flag)...")
        llm_response_text = call_ollama_api(prompt, self.ollama_chat_model, self.ollama_base_url, timeout=180)
        if llm_response_text is None: log.error("Failed to get thinking response from Ollama."); return "LLM communication failed.", {"type": "error", "message": "LLM communication failure (thinking).", "subtype": "llm_comm_error"}

        # --- Parsing logic unchanged ---
        try:
            action: Dict[str, Any] = {"type": "unknown"}; raw_thinking = llm_response_text
            thinking_marker = "THINKING:"; action_marker = "NEXT_ACTION:"; tool_marker = "TOOL:"; params_marker = "PARAMETERS:"; answer_marker = "ANSWER:"; reflections_marker = "REFLECTIONS:"
            thinking_start = llm_response_text.find(thinking_marker); action_start = llm_response_text.find(action_marker)
            if thinking_start != -1: end_think = action_start if action_start > thinking_start else len(llm_response_text); raw_thinking = llm_response_text[thinking_start + len(thinking_marker):end_think].strip()
            else: log.warning("'THINKING:' marker not found in LLM response."); raw_thinking = llm_response_text
            action_type_str = ""; action_subtype = "unknown_error"
            if action_start != -1: end_action = llm_response_text.find('\n', action_start); end_action = end_action if end_action != -1 else len(llm_response_text); action_type_str = llm_response_text[action_start + len(action_marker):end_action].strip()
            else: log.error("'NEXT_ACTION:' marker not found."); action = {"type": "error", "message": "Missing NEXT_ACTION marker.", "subtype": "parse_error"}; return raw_thinking, action
            if "use_tool" in action_type_str:
                action["type"] = "use_tool"; tool_name = None; params_json = None
                tool_start = llm_response_text.find(tool_marker, action_start); params_start = llm_response_text.find(params_marker, action_start)
                if tool_start != -1:
                     end_tool = llm_response_text.find('\n', tool_start)
                     if params_start > tool_start and (end_tool == -1 or params_start < end_tool): end_tool = params_start
                     end_tool = end_tool if end_tool != -1 else len(llm_response_text)
                     tool_name = llm_response_text[tool_start + len(tool_marker):end_tool].strip().strip('"').strip("'"); log.debug(f"Extracted tool name: '{tool_name}'")
                else: action = {"type": "error", "message": "Missing TOOL marker for use_tool action.", "subtype": "parse_error"}; return raw_thinking, action
                if params_start != -1:
                    params_str_start = params_start + len(params_marker); end_params = len(llm_response_text); next_marker_pos = -1
                    for marker in [f"\n{answer_marker}", f"\n{reflections_marker}", f"\n{thinking_marker}", f"\n{action_marker}", f"\n{tool_marker}", f"\n{params_marker}"]:
                         pos = llm_response_text.find(marker, params_str_start)
                         if pos != -1 and (next_marker_pos == -1 or pos < next_marker_pos): next_marker_pos = pos
                    if next_marker_pos != -1: end_params = next_marker_pos
                    raw_params = llm_response_text[params_str_start:end_params].strip()
                    raw_params = re.sub(r'^```json\s*', '', raw_params, flags=re.I|re.M); raw_params = re.sub(r'^```\s*', '', raw_params, flags=re.M); raw_params = re.sub(r'\s*```$', '', raw_params, flags=re.M); raw_params = raw_params.strip()
                    json_str = raw_params; params_json = None; log.debug(f"Raw PARAMS string: '{json_str}'")
                    if json_str:
                        try: params_json = json.loads(json_str); log.debug(f"PARAMS parsed directly: {params_json}")
                        except json.JSONDecodeError as e1:
                            log.warning(f"Direct JSON parse failed: {e1}. Fixing..."); fixed_str = json_str.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'"); fixed_str = re.sub(r",\s*([}\]])", r"\1", fixed_str); log.debug(f"Attempting parse after fixes: '{fixed_str}'")
                            try: params_json = json.loads(fixed_str); log.debug(f"PARAMS parsed after fixes: {params_json}")
                            except json.JSONDecodeError as e2:
                                first_brace = raw_params.find('{'); last_brace = raw_params.rfind('}'); extracted_str = raw_params[first_brace : last_brace + 1] if first_brace != -1 and last_brace > first_brace else raw_params; log.debug(f"Extracting braces: '{extracted_str}'")
                                try: fixed_extracted_str = extracted_str.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'"); fixed_extracted_str = re.sub(r",\s*([}\]])", r"\1", fixed_extracted_str); log.debug(f"Trying extracted+fixed: '{fixed_extracted_str}'"); params_json = json.loads(fixed_extracted_str); log.debug(f"PARAMS parsed after extraction+fixes: {params_json}")
                                except json.JSONDecodeError as e3: err_msg = f"Invalid JSON in PARAMETERS after all attempts: {e3}. Original: '{raw_params}'"; log.error(err_msg); action = {"type": "error", "message": err_msg, "subtype": "parse_error"}; return raw_thinking, action
                    else: action = {"type": "error", "message": "Empty PARAMETERS block.", "subtype": "parse_error"}; return raw_thinking, action
                else: action = {"type": "error", "message": "Missing PARAMETERS marker for use_tool action.", "subtype": "parse_error"}; return raw_thinking, action
                if action.get("type") == "use_tool":
                    if not tool_name: action = {"type": "error", "message": "Internal Error: Tool name missing.", "subtype": "parse_error"}
                    elif tool_name not in self.tools: action = {"type": "error", "message": f"Tool '{tool_name}' not available.", "subtype": "invalid_tool"}
                    elif not isinstance(params_json, dict): action = {"type": "error", "message": f"Parsed PARAMETERS not a JSON object: {type(params_json)}", "subtype": "parse_error"}
                    else:
                        valid_params = True; err_msg = ""; action_subtype = "invalid_params"
                        if tool_name == "web_browse" and ("url" not in params_json or not isinstance(params_json.get("url"), str) or not params_json["url"].strip()): err_msg = "Missing/invalid 'url' for web_browse."
                        elif tool_name == "web_search" and ("query" not in params_json or not isinstance(params_json.get("query"), str) or not params_json["query"].strip()): err_msg = "Missing/invalid 'query' for web_search."
                        elif tool_name == "memory_retrieval" and ("query" not in params_json or not isinstance(params_json.get("query"), str) or not params_json["query"].strip()): err_msg = "Missing/invalid 'query' for memory_retrieval."
                        if err_msg:
                            log.error(f"{err_msg} Params: {params_json}"); action = {"type": "error", "message": err_msg, "subtype": action_subtype}; valid_params = False
                        if valid_params: action["tool"] = tool_name; action["parameters"] = params_json; log.info(f"Parsed action: Use Tool '{tool_name}'")
            elif "final_answer" in action_type_str:
                action["type"] = "final_answer"; answer = ""; reflections = ""
                answer_start = llm_response_text.find(answer_marker, action_start); reflections_start = llm_response_text.find(reflections_marker, action_start)
                if answer_start != -1: end_answer = reflections_start if reflections_start > answer_start else len(llm_response_text); answer = llm_response_text[answer_start + len(answer_marker):end_answer].strip()
                else:
                    potential_answer_start = llm_response_text.find('\n', action_start) + 1; potential_answer_end = reflections_start if reflections_start > potential_answer_start else len(llm_response_text); answer = llm_response_text[potential_answer_start:potential_answer_end].strip() if potential_answer_start > 0 and potential_answer_start < potential_answer_end else ""
                if reflections_start != -1: reflections = llm_response_text[reflections_start + len(reflections_marker):].strip()
                action["answer"] = answer; action["reflections"] = reflections
                if not answer: action = {"type": "error", "message": "LLM chose final_answer but provided no ANSWER.", "subtype": "parse_error"}
                elif len(answer) < 100 and re.search(r'\b(error|fail|cannot|unable)\b', answer, re.IGNORECASE) and not user_suggested_move_on:
                     log.warning(f"LLM 'final_answer' looks like giving up without user suggestion: '{answer[:100]}...'"); action = {"type": "error", "message": f"LLM gave up instead of recovering: {answer}", "subtype": "give_up_error"}
                else: log.info(f"Parsed action: Final Answer.")
            elif action_type_str: action = {"type": "error", "message": f"Invalid NEXT_ACTION: '{action_type_str}'", "subtype": "parse_error"}
            if action["type"] == "unknown" and action.get("subtype", "") != "parse_error": action = {"type": "error", "message": "Could not parse action.", "subtype": "parse_error"}
            return raw_thinking, action
        except Exception as e: log.exception(f"CRITICAL failure parsing LLM thinking response: {e}\nResponse:\n{llm_response_text}"); return raw_thinking or "Error parsing.", {"type": "error", "message": f"Internal error parsing LLM response: {e}", "subtype": "internal_error"}

    # --- MODIFIED: Pass identity statement to memory_retrieval tool ---
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
         if tool_name not in self.tools: return {"error": f"Tool '{tool_name}' not found", "status": "failed"}
         tool = self.tools[tool_name]; log.info(f"Executing tool '{tool_name}' with params: {parameters}")
         try:
             # --- Pass memory instance AND identity statement if it's the memory tool ---
             if tool_name == "memory_retrieval":
                 result = tool.run(parameters, memory_instance=self.memory, identity_statement=self.identity_statement) # Pass identity
             else:
                 result = tool.run(parameters)

             log.info(f"Tool '{tool_name}' finished.")
             if tool_name == 'web_browse' and isinstance(result, dict) and 'content' in result:
                 with self._state_lock: self.session_state['last_web_browse_content'] = result.get('content', '(Empty)')
             if not isinstance(result, dict): return {"tool_name": tool_name, "result": {"unexpected_result": str(result)}, "status": "completed_malformed_output"}
             if result.get("error"): log.warning(f"Tool '{tool_name}' reported error: {result['error']}"); return {"tool_name": tool_name, "error": result["error"], "status": "failed"}
             try: json.dumps(result)
             except TypeError as json_err: log.warning(f"Result from '{tool_name}' not JSON serializable: {json_err}."); return {"tool_name": tool_name, "result": {"unserializable_result": str(result)}, "status": "completed_unserializable"}
             return {"tool_name": tool_name, "result": result, "status": "completed"}
         except Exception as e: log.exception(f"CRITICAL Error executing tool '{tool_name}': {e}"); return {"tool_name": tool_name, "error": f"Tool execution raised exception: {e}", "status": "failed"}

    def _save_qlora_datapoint(self, source_type: str, instruction: str, input_context: str, output: str):
        # ... (implementation unchanged) ...
        if not output: return
        try:
            datapoint = {"instruction": instruction, "input": input_context, "output": output, "source_type": source_type}
            with open(self.qlora_dataset_path, 'a', encoding='utf-8') as f: json.dump(datapoint, f, ensure_ascii=False); f.write('\n')
            log.info(f"QLoRA datapoint saved (Source: {source_type})")
        except Exception as e: log.exception(f"Failed to save QLoRA datapoint: {e}")

    def _summarize_and_prune_task_memories(self, task: Task):
        # ... (implementation unchanged, uses format_relative_time util) ...
        if not config.ENABLE_MEMORY_SUMMARIZATION: return
        log.info(f"Summarizing memories for {task.status} task {task.id}...")
        task_memories = self.memory.get_memories_by_metadata(filter_dict={"task_id": task.id})
        if not task_memories: return
        log.debug(f"Found {len(task_memories)} memories for task {task.id}.")
        summary_context = f"Task: {task.description}\nStatus: {task.status}\nResult: {str(task.result)[:500]}\nReflections: {task.reflections}\n--- Log (with relative time) ---\n"
        try: task_memories.sort(key=lambda m: m.get('metadata', {}).get('timestamp', ''))
        except: pass
        mem_details = []; memory_ids_to_prune = []
        for mem in task_memories:
            meta = mem.get('metadata', {}); mem_type = meta.get('type', 'memory')
            if mem_type in ['task_summary', 'identity_revision', 'session_reflection']: continue
            timestamp_str = meta.get('timestamp', 'N/A'); content = mem.get('content', '')
            relative_time = format_relative_time(timestamp_str) # Use util
            if mem_type != 'user_suggestion_move_on':
                mem_details.append(f"[{relative_time} - {mem_type}]\n{content}\n-------")
            memory_ids_to_prune.append(mem['id'])

        summary_context += "\n".join(mem_details) + "\n--- End Log ---"
        summary_prompt = f"""Summarize the execution of this agent task based on the log below. Focus on objective, key actions/findings, errors/recovery, and final outcome ({task.status}).\n\n**Input Data:**\n{summary_context[:config.CONTEXT_TRUNCATION_LIMIT // 2]}\n\n**Output:** Concise summary text only."""
        log.info(f"Asking {self.ollama_chat_model} to summarize task {task.id}...")
        summary_text = call_ollama_api(summary_prompt, self.ollama_chat_model, self.ollama_base_url, timeout=150)
        if summary_text and summary_text.strip():
            summary_metadata = {"type": "task_summary", "task_id": task.id, "original_status": task.status, "summarized_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "original_memory_count": len(memory_ids_to_prune)}
            summary_id = self.memory.add_memory(f"Task Summary ({task.status}):\n{summary_text}", summary_metadata)
            if summary_id and config.DELETE_MEMORIES_AFTER_SUMMARY:
                log.info(f"Summary added ({summary_id}). Pruning {len(memory_ids_to_prune)} original memories for task {task.id}...")
                ids_to_delete_final = [mid for mid in memory_ids_to_prune if mid != summary_id]
                if ids_to_delete_final:
                    deleted = self.memory.delete_memories(ids_to_delete_final); log.info(f"Pruning status for task {task.id}: {deleted}")
                else:
                     log.info(f"No original memories to prune for task {task.id} after summary creation.")
            elif summary_id: log.info(f"Summary added ({summary_id}). Deletion disabled.")
            else: log.error(f"Failed to add summary memory for task {task.id}.")
        else: log.warning(f"Failed to generate summary for task {task.id}.")

    def _execute_step(self, task: Task) -> Dict[str, Any]:
        # ... (logic calling generate_thinking is unchanged) ...
        step_start_time = time.time(); step_log = []; final_answer_text = None; step_status = "processing"; task_status_updated_this_step = False
        current_context = self.session_state.get("investigation_context", ""); step_num = self.session_state.get(f"task_{task.id}_step", 0) + 1
        current_retries = self.session_state.get("current_task_retries", 0)
        user_suggested_move_on = self.session_state.get("user_suggestion_move_on_pending", False)
        if user_suggested_move_on:
            log.info(f"User suggestion 'move on' is pending for task {task.id}.")
            with self._state_lock: self.session_state["user_suggestion_move_on_pending"] = False

        step_log.append(f"--- Task '{task.id}' | Step {step_num} (Retry {current_retries}/{config.AGENT_MAX_STEP_RETRIES}) ---")
        if user_suggested_move_on: step_log.append("[INFO] User suggestion to wrap up is being considered this step.")
        log.info(f"Executing step {step_num} for task {task.id} (Retry: {current_retries})")

        # --- generate_thinking is called here ---
        raw_thinking, action = self.generate_thinking(
            task.description, current_context, None, user_suggested_move_on=user_suggested_move_on
        )
        # ... (rest of step execution logic unchanged) ...
        action_type = action.get("type", "error"); action_message = action.get("message", "Unknown error"); action_subtype = action.get("subtype", "unknown_error")
        thinking_to_store = raw_thinking if raw_thinking else "Thinking process not extracted."; step_log.append(f"Thinking:\n{thinking_to_store}")
        self.memory.add_memory(f"Step {step_num} Thinking (Action: {action_type}, Subtype: {action_subtype}):\n{thinking_to_store}", {"type": "task_thinking", "task_id": task.id, "step": step_num, "action_type": action_type, "action_subtype": action_subtype})

        if action_type == "error":
            log.error(f"[STEP ERROR] LLM Action Error: {action_message} (Subtype: {action_subtype})"); step_log.append(f"[ERROR] Action Error: {action_message} (Type: {action_subtype})")
            error_context = f"\n--- Step {step_num} Error (Retry {current_retries+1}/{config.AGENT_MAX_STEP_RETRIES}): {action_message} (Type: {action_subtype}) ---"; current_context += error_context
            self.memory.add_memory(f"Step {step_num} Error: {action_message}", {"type": "agent_error", "task_id": task.id, "step": step_num, "error_subtype": action_subtype})
            if current_retries < config.AGENT_MAX_STEP_RETRIES:
                 self.session_state["current_task_retries"] = current_retries + 1; log.warning(f"Incrementing retry count to {current_retries + 1}."); step_status = "error_retry"
            else:
                log.error(f"Max retries reached for task {task.id}. Failing."); fail_reason = f"Failed after {step_num} steps. Max retries on error: {action_message}"
                self.task_queue.update_task(task.id, "failed", reflections=fail_reason); task_status_updated_this_step = True; step_status = "failed"
                failed_task_obj = self.task_queue.get_task(task.id);
                if failed_task_obj and config.ENABLE_MEMORY_SUMMARIZATION:
                    try: self._summarize_and_prune_task_memories(failed_task_obj);
                    except Exception as e: log.exception(f"Error summary on fail: {e}")
                if self._is_running.is_set():
                    self._revise_identity_statement(f"Failed Task: {task.description[:50]}...")
                else:
                     log.info("Agent paused, skipping identity revision after failed task.")
                self.session_state["current_task_id"] = None; self.session_state["investigation_context"] = ""; self.session_state["current_task_retries"] = 0
        elif action_type == "use_tool":
            tool_name = action.get("tool"); params = action.get("parameters"); log.info(f"[ACTION] Use Tool '{tool_name}'")
            step_log.append(f"[ACTION] Use Tool: {tool_name} params: {json.dumps(params, ensure_ascii=False, indent=2)}")
            tool_results = self.execute_tool(tool_name, params); tool_status = tool_results.get("status", "unknown"); tool_error = tool_results.get("error")
            step_context_update = f"\n--- Step {step_num} ---\nAction: Use Tool '{tool_name}'\nParams: {json.dumps(params, ensure_ascii=False)}\nResult Status: {tool_status}\nResult:\n```json\n{json.dumps(tool_results, indent=2, ensure_ascii=False)}\n```\n"; current_context += step_context_update
            self.memory.add_memory(f"Step {step_num} Tool Action & Result (Status: {tool_status}):\n{step_context_update}", {"type": "tool_result", "task_id": task.id, "step": step_num, "tool_name": tool_name, "result_status": tool_status, "params": json.dumps(params)})
            if tool_status == 'completed': step_log.append(f"[INFO] Tool '{tool_name}' completed."); step_status = "processing"; self.session_state["current_task_retries"] = 0;
            elif tool_status == 'failed':
                log.warning(f"Tool '{tool_name}' failed: {tool_error}"); step_log.append(f"[ERROR] Tool '{tool_name}' failed: {tool_error}")
                if current_retries < config.AGENT_MAX_STEP_RETRIES:
                     self.session_state["current_task_retries"] = current_retries + 1; log.warning(f"Incrementing retry count to {current_retries + 1}."); step_status = "error_retry"
                else:
                    log.error(f"Max retries reached after tool failure for task {task.id}. Failing."); fail_reason = f"Failed after {step_num} steps. Max retries on tool failure: {tool_error}"
                    self.task_queue.update_task(task.id, "failed", reflections=fail_reason); task_status_updated_this_step = True; step_status = "failed"
                    failed_task_obj = self.task_queue.get_task(task.id)
                    if failed_task_obj and config.ENABLE_MEMORY_SUMMARIZATION:
                        try: self._summarize_and_prune_task_memories(failed_task_obj);
                        except Exception as e: log.exception(f"Error summary on fail: {e}")
                    if self._is_running.is_set():
                         self._revise_identity_statement(f"Failed Task: {task.description[:50]}...")
                    else:
                         log.info("Agent paused, skipping identity revision after failed task.")
                    self.session_state["current_task_id"] = None; self.session_state["investigation_context"] = ""; self.session_state["current_task_retries"] = 0
            else:
                log.warning(f"Tool '{tool_name}' finished with status: {tool_status}. Treating as processing step for now.");
                step_log.append(f"[WARNING] Tool '{tool_name}' status: {tool_status}.");
                step_status = "processing"; self.session_state["current_task_retries"] = 0
        elif action_type == "final_answer":
            log.info("[ACTION] Provide Final Answer."); step_log.append("[ACTION] Provide Final Answer.")
            answer = action.get("answer", "").strip(); reflections = action.get("reflections", "").strip(); final_answer_text = answer
            print("\n" + "="*15 + f" FINAL ANSWER (Task {task.id}) " + "="*15); print(answer); print("="* (34+len(str(task.id))) + "\n")
            result_payload = {"answer": answer, "steps_taken": step_num}
            self.task_queue.update_task(task.id, "completed", result=result_payload, reflections=reflections); task_status_updated_this_step = True; step_status = "completed"
            self._save_qlora_datapoint(source_type="task_completion", instruction="Given the task, provide answer.", input_context=task.description, output=answer)
            self.memory.add_memory(f"Final Answer:\n{answer}", {"type": "task_result", "task_id": task.id, "final_step": step_num})
            if reflections: self.memory.add_memory(f"Final Reflections:\n{reflections}", {"type": "task_reflection", "task_id": task.id, "final_step": step_num})
            completed_task_obj = self.task_queue.get_task(task.id)
            if completed_task_obj and config.ENABLE_MEMORY_SUMMARIZATION:
                try: self._summarize_and_prune_task_memories(completed_task_obj);
                except Exception as e: log.exception(f"Error summary on complete: {e}")
            if self._is_running.is_set():
                 self._revise_identity_statement(f"Completed Task: {task.description[:50]}...")
            else:
                 log.info("Agent paused, skipping identity revision after completed task.")
            self.session_state["current_task_id"] = None; self.session_state["investigation_context"] = ""; self.session_state["current_task_retries"] = 0

        if step_status in ["processing", "error_retry"]:
            context_limit = config.CONTEXT_TRUNCATION_LIMIT
            if len(current_context) > context_limit: current_context = f"(Truncated)\n...\n{current_context[len(current_context) - context_limit:]}"
            self.session_state["investigation_context"] = current_context; self.session_state[f"task_{task.id}_step"] = step_num
        elif step_status in ["completed", "failed"]:
             self.session_state.pop(f"task_{task.id}_step", None)
             with self._state_lock: self.session_state["user_suggestion_move_on_pending"] = False

        self.save_session_state()
        step_duration = time.time() - step_start_time; step_log.append(f"Step {step_num} duration: {step_duration:.2f}s. Step Status: {step_status}")
        final_task_status_obj = self.task_queue.get_task(task.id); final_task_status = final_task_status_obj.status if final_task_status_obj else "unknown"

        # --- MODIFIED: Retrieve memories for UI using memory's method, pass identity ---
        ui_memory_query = f"Memories relevant to just completed step {step_num} of task {task.id}. Action taken: {action_type}. Status: {step_status}"
        recent_memories_for_ui, _ = self.memory.retrieve_and_rerank_memories(
            query=ui_memory_query,
            task_description=task.description,
            context=current_context,
            identity_statement=self.identity_statement, # <<<--- PASS IDENTITY
            n_final=5
        )

        return {
            "status": final_task_status,
            "log": "\n".join(step_log),
            "current_task_id": self.session_state.get("current_task_id"),
            "current_task_desc": task.description if self.session_state.get("current_task_id") else "None",
            "recent_memories": recent_memories_for_ui,
            "last_web_content": self.session_state.get("last_web_browse_content", "(No recent web browse)"),
            "final_answer": final_answer_text
        }


    def process_one_step(self) -> Dict[str, Any]:
        # ... (calls _execute_step, logic unchanged) ...
        log.info("Processing one autonomous step..."); step_result_log = ["Attempting one step..."]
        default_state = {"status": "idle", "log": "Idle.", "current_task_id": None, "current_task_desc": "None", "recent_memories": [], "last_web_content": self.session_state.get("last_web_browse_content", "(None)"), "final_answer": None}
        try:
            task_id_to_process = self.session_state.get("current_task_id"); task: Optional[Task] = None
            if task_id_to_process:
                task = self.task_queue.get_task(task_id_to_process)
                if not task or task.status not in ["in_progress", "pending"]:
                    log.warning(f"Task {task_id_to_process} from state invalid/finished. Searching."); step_result_log.append(f"Task {task_id_to_process} invalid/finished.")
                    task = None;
                    self.session_state["current_task_id"] = None; self.session_state["investigation_context"] = ""; self.session_state["current_task_retries"] = 0
                    with self._state_lock: self.session_state["user_suggestion_move_on_pending"] = False
            if not task:
                task = self.task_queue.get_next_task()
                if task:
                    log.info(f"Starting task: {task.id} - '{task.description[:60]}...'"); step_result_log.append(f"Starting task {task.id}: {task.description[:60]}...")
                    self.task_queue.update_task(task.id, "in_progress"); self.session_state["current_task_id"] = task.id
                    self.session_state["investigation_context"] = f"Objective: {task.description}\n"; self.session_state.pop(f"task_{task.id}_step", None); self.session_state["current_task_retries"] = 0
                    with self._state_lock: self.session_state["user_suggestion_move_on_pending"] = False
                    self.save_session_state(); task = self.task_queue.get_task(task.id)
                else:
                    log.info("No runnable tasks found."); step_result_log.append("No runnable tasks found.")
                    if not self.session_state.get("current_task_id") and not self.session_state.get("user_suggestion_move_on_pending"):
                        log.info("Agent idle, considering generating tasks..."); generated_desc = self.generate_new_tasks(max_new_tasks=1)
                        if generated_desc: step_result_log.append(f"Generated task: {generated_desc[:60]}...")
                        else: step_result_log.append("No new tasks generated.")
                    elif self.session_state.get("user_suggestion_move_on_pending"):
                        log.info("Agent idle, but user suggestion flag is set. Waiting.")
                        step_result_log.append("Idle, but user suggestion flag is set. Waiting for flag clear or new task.")

                    self._update_ui_state(status="idle", log="\n".join(step_result_log), current_task_id=None, current_task_desc="None")
                    return self.get_ui_update_state()
            if task:
                step_result = self._execute_step(task);
                self._update_ui_state(**step_result); return self.get_ui_update_state()
            else: log.error("Task became None unexpectedly before step execution."); step_result_log.append("[ERROR] Task lost before execution."); self._update_ui_state(status="error", log="\n".join(step_result_log)); return self.get_ui_update_state()
        except Exception as e:
            log.exception("CRITICAL Error during process_one_step loop"); step_result_log.append(f"[CRITICAL ERROR]: {e}")
            current_task_id = self.session_state.get("current_task_id")
            if current_task_id:
                 log.error(f"Failing task {current_task_id} due to critical loop error."); fail_reason = f"Critical loop error: {e}"; self.task_queue.update_task(current_task_id, "failed", reflections=fail_reason)
                 failed_task_obj = self.task_queue.get_task(current_task_id)
                 if failed_task_obj and config.ENABLE_MEMORY_SUMMARIZATION:
                    try: self._summarize_and_prune_task_memories(failed_task_obj);
                    except Exception as sum_e: log.exception(f"Error summary on critical fail: {sum_e}")
                 if self._is_running.is_set():
                     self._revise_identity_statement(f"Critical Failure during Task: {failed_task_obj.description[:50] if failed_task_obj else current_task_id}...")
                 else:
                     log.info("Agent paused, skipping identity revision after critical failure.")
                 self.session_state["current_task_id"] = None; self.session_state["investigation_context"] = ""; self.session_state["current_task_retries"] = 0
                 with self._state_lock: self.session_state["user_suggestion_move_on_pending"] = False
                 self.save_session_state()
            self._update_ui_state(status="critical_error", log="\n".join(step_result_log), current_task_id=None, current_task_desc="None"); return self.get_ui_update_state()

    def generate_new_tasks(self, max_new_tasks: int = 3, last_user_message: Optional[str] = None, last_assistant_response: Optional[str] = None) -> Optional[str]:
        # ... (Logic calling retrieve_and_rerank_memories changed) ...
        log.info("\n--- Attempting to Generate New Tasks ---"); context_source = ""; context_query = ""; mem_query = ""; critical_evaluation_instruction = ""
        if last_user_message and last_assistant_response: context_source = "last chat interaction"; context_query = f"Last User: {last_user_message}\nLast Assistant: {last_assistant_response}"; mem_query = f"Context relevant to last chat: {last_user_message}"; critical_evaluation_instruction = "\n**Critically Evaluate Need:** Based *specifically* on the **Last Interaction**, is a background task *truly necessary*? Output `[]` if not."
        else: context_source = "general agent state"; context_query = "General status. Consider logical follow-up/exploration based on completed tasks, idle state, and my identity."; mem_query = "Recent activities, conclusions, errors, reflections, summaries, identity revisions."; critical_evaluation_instruction = "\n**Critically Evaluate Need:** Are new tasks genuinely needed for exploration/follow-up, consistent with identity? Output `[]` if not."
        log.info(f"Retrieving context for task generation (Source: {context_source})...");
        # --- MODIFIED: Call memory's method, pass identity ---
        recent_mems, _ = self.memory.retrieve_and_rerank_memories(
            query=mem_query,
            task_description="Task Generation Context",
            context=context_query,
            identity_statement=self.identity_statement, # <<<--- PASS IDENTITY
            n_results=15,
            n_final=10
        )
        # --- Format with relative time ---
        mem_summary_list = []
        for m in recent_mems:
            relative_time = format_relative_time(m['metadata'].get('timestamp')) # Use util
            mem_type = m['metadata'].get('type','mem')
            snippet = m['content'][:150].strip().replace('\n', ' ')
            mem_summary_list.append(f"- [{relative_time}] {mem_type}: {snippet}...")
        mem_summary = "\n".join(mem_summary_list) if mem_summary_list else "None"

        existing_tasks_info = [{"id": t.id, "description": t.description, "status": t.status} for t in self.task_queue.tasks.values()]; active_tasks_summary = "\n".join([f"- ID: {t['id']} (Status: {t['status']}) Desc: {t['description'][:100]}..." for t in existing_tasks_info if t['status'] in ['pending', 'in_progress']]) or "None"
        completed_failed_summary = "\n".join([f"- ID: {t['id']} (Status: {t['status']}) Desc: {t['description'][:100]}..." for t in existing_tasks_info if t['status'] in ['completed', 'failed']])[-1000:]; valid_existing_task_ids = set(t['id'] for t in existing_tasks_info)

        prompt = f"""You are the planning component of an AI agent. Generate new, actionable tasks based on state, history, and identity.
**Agent's Current Identity:**
{self.identity_statement}
**Current Context Focus:**
{context_query}
**Recent Activity & Memory Snippets (Consider recency indicated by '[X time ago]'):**
{mem_summary}
**Existing Pending/In-Progress Tasks (Check duplicates!):**
{active_tasks_summary}
**Recently Finished Tasks (for context):**
{completed_failed_summary}
{critical_evaluation_instruction}
**Your Task:** Review all info (paying attention to recency of memories). Identify gaps/next steps relevant to Context & Identity. Generate up to {max_new_tasks} new, specific, actionable tasks. AVOID DUPLICATES of pending/in-progress tasks. Assign priority (1-10, consider identity). Add necessary `depends_on` using existing IDs only. Suggest follow-up, refinement, or (max 1) exploratory tasks.
**Guidelines:** Actionable (verb start), Specific, Novel, Relevant (to context/identity), Concise.
**Output Format (Strict JSON):** Provide *only* a valid JSON list of objects (or `[]`). Example:
```json
[
  {{"description": "Summarize findings from task [xyz], focus on research goals.", "priority": 7, "depends_on": ["xyz"]}},
  {{"description": "Explore using web_browse for real-time data on [topic relevant to identity].", "priority": 5}}
]
```"""
        log.info(f"Asking {self.ollama_chat_model} to generate up to {max_new_tasks} new tasks..."); llm_response = call_ollama_api(prompt, self.ollama_chat_model, self.ollama_base_url, timeout=180)
        # --- Parsing logic unchanged ---
        if not llm_response: log.error("LLM failed task gen."); return None
        first_task_desc_added = None; new_tasks_added = 0
        try:
            llm_response = re.sub(r'^```json\s*', '', llm_response, flags=re.I|re.M); llm_response = re.sub(r'\s*```$', '', llm_response, flags=re.M).strip()
            list_start = llm_response.find('['); list_end = llm_response.rfind(']')
            if list_start == -1 or list_end == -1 or list_end < list_start:
                if "no new tasks" in llm_response.lower() or llm_response.strip() == "[]": log.info("LLM: no tasks needed."); return None
                else: raise json.JSONDecodeError(f"JSON list '[]' not found: {llm_response}", llm_response, 0)
            json_str = llm_response[list_start : list_end + 1]; suggested_tasks = json.loads(json_str)
            if not isinstance(suggested_tasks, list): log.warning(f"LLM task gen not list: {suggested_tasks}"); return None
            if not suggested_tasks: log.info("LLM suggested no new tasks."); return None
            log.info(f"LLM suggested {len(suggested_tasks)} tasks. Validating..."); current_task_ids_in_batch = set()
            active_task_descriptions = {t['description'].strip().lower() for t in existing_tasks_info if t['status'] in ['pending', 'in_progress']}
            for task_data in suggested_tasks:
                if not isinstance(task_data, dict): continue
                description = task_data.get("description")
                if not description or not isinstance(description, str) or not description.strip(): continue
                description = description.strip()
                if description.lower() in active_task_descriptions: log.warning(f"Skipping duplicate task: '{description[:80]}...'"); continue
                priority = task_data.get("priority", 5);
                try: priority = max(1, min(10, int(priority)));
                except: priority = 5
                dependencies_raw = task_data.get("depends_on"); validated_dependencies = []
                if isinstance(dependencies_raw, list):
                    for dep_id in dependencies_raw: dep_id_str = str(dep_id).strip();
                    if dep_id_str in valid_existing_task_ids or dep_id_str in current_task_ids_in_batch: validated_dependencies.append(dep_id_str)
                    else: log.warning(f"Dependency '{dep_id_str}' for new task not found. Ignoring.")

                new_task = Task(description, priority, depends_on=validated_dependencies or None); new_task_id = self.task_queue.add_task(new_task)
                if new_task_id:
                     if new_tasks_added == 0: first_task_desc_added = description
                     new_tasks_added += 1; active_task_descriptions.add(description.lower()); valid_existing_task_ids.add(new_task_id); current_task_ids_in_batch.add(new_task_id)
                     log.info(f"Added Task {new_task_id}: '{description[:60]}...' (Prio: {priority}, Depends: {validated_dependencies})")
                if new_tasks_added >= max_new_tasks: break
        except json.JSONDecodeError as e: log.error(f"Failed JSON parse task gen: {e}\nLLM Resp:\n{llm_response}\n---"); return None
        except Exception as e: log.exception(f"Unexpected error task gen: {e}"); return None
        log.info(f"Finished Task Generation: Added {new_tasks_added} new tasks."); return first_task_desc_added

    # --- Loop/Control Methods (unchanged) ---
    def _autonomous_loop(self, initial_delay: float = 2.0, step_delay: float = 5.0):
        # ... (implementation unchanged) ...
        log.info("Background agent loop starting."); time.sleep(initial_delay)
        while not self._shutdown_request.is_set():
            if not self._is_running.is_set():
                with self._state_lock:
                    if self._ui_update_state["status"] != "paused": log.info("Agent loop pausing..."); self._ui_update_state["status"] = "paused"; self._ui_update_state["log"] = "Agent loop paused."
                time.sleep(1); continue
            log.debug("Autonomous loop: Processing step...")
            step_start = time.monotonic()
            try: self.process_one_step()
            except Exception as e: log.exception(f"CRITICAL UNHANDLED ERROR in autonomous loop iteration."); self._update_ui_state(status="critical_error", log=f"CRITICAL LOOP ERROR: {e}"); time.sleep(step_delay * 5)
            step_duration = time.monotonic() - step_start; remaining_delay = max(0, step_delay - step_duration); log.debug(f"Step took {step_duration:.2f}s. Sleeping {remaining_delay:.2f}s."); time.sleep(remaining_delay)
        log.info("Background agent loop shutting down.")

    def start_autonomous_loop(self):
        # ... (implementation unchanged) ...
        if self._agent_thread and self._agent_thread.is_alive():
            if not self._is_running.is_set(): log.info("Agent loop resuming..."); self._is_running.set(); self._update_ui_state(status="running", log="Agent resumed.")
            else: log.info("Agent loop is already running.")
        else:
            log.info("Starting new background agent loop..."); self._shutdown_request.clear(); self._is_running.set(); self._update_ui_state(status="running", log="Agent started.")
            self._agent_thread = threading.Thread(target=self._autonomous_loop, daemon=True); self._agent_thread.start()

    def pause_autonomous_loop(self):
        # ... (implementation unchanged) ...
        if self._is_running.is_set(): log.info("Pausing agent loop..."); self._is_running.clear(); self._update_ui_state(status="paused", log="Agent loop paused.")
        else: log.info("Agent loop is already paused.")

    def shutdown(self):
        # ... (implementation unchanged) ...
        log.info("Shutdown requested."); self._shutdown_request.set(); self._is_running.clear()
        if self._agent_thread and self._agent_thread.is_alive():
            log.info("Waiting for agent thread join..."); self._agent_thread.join(timeout=15)
            if self._agent_thread.is_alive(): log.warning("Agent thread didn't join cleanly.")
            else: log.info("Agent thread joined.")
        else: log.info("Agent thread not running/already joined.")
        self._update_ui_state(status="shutdown", log="Agent shut down."); self.save_session_state(); log.info("Shutdown complete.")

    # --- Reflection Methods ---
    def add_self_reflection(self, reflection: str, reflection_type: str = "self_reflection"):
         # ... (implementation unchanged) ...
         if not reflection or not isinstance(reflection, str) or not reflection.strip(): return None
         if self._is_running.is_set():
             log.info(f"Adding {reflection_type} to memory...")
             return self.memory.add_memory(reflection, {"type": reflection_type})
         else:
             log.info(f"Agent paused, skipping adding {reflection_type} to memory.")
             return None

    def generate_and_add_session_reflection(self, start: datetime.datetime, end: datetime.datetime, completed_count: int, processed_count: int):
         # ... (Logic calling retrieve_and_rerank_memories changed) ...
         duration_minutes = (end - start).total_seconds() / 60; log.info("Retrieving context for session reflection...")
         mem_query = "Summary session activities, task summaries, errors, outcomes, identity statements/revisions.";
         # --- MODIFIED: Call memory's method, pass identity ---
         recent_mems, _ = self.memory.retrieve_and_rerank_memories(
             query=mem_query,
             task_description="Session Reflection",
             context="End of work session",
             identity_statement=self.identity_statement, # <<<--- PASS IDENTITY
             n_results=20,
             n_final=15
         )
         # --- Format with relative time ---
         mem_summary_list = []
         for m in recent_mems:
             relative_time = format_relative_time(m['metadata'].get('timestamp')) # Use util
             mem_type = m['metadata'].get('type','mem')
             snippet = m['content'][:100].strip().replace('\n', ' ')
             mem_summary_list.append(f"- [{relative_time}] {mem_type}: {snippet}...")
         mem_summary = "\n".join(mem_summary_list) if mem_summary_list else "None"

         prompt = f"""You are AI agent ({self.identity_statement}). Reflect on your work session.\n**Start:** {start.isoformat()}\n**End:** {end.isoformat()}\n**Duration:** {duration_minutes:.1f} min\n**Tasks Completed:** {completed_count}\n**Tasks Processed:** {processed_count}\n**Recent Activity/Identity Notes (Consider recency indicated by '[X time ago]'):**\n{mem_summary}\n\n**Reflection Task:** Provide concise reflection: 1. Accomplishments? 2. Efficiency? 3. Challenges/errors? 4. Learnings? 5. Alignment with identity/goals? 6. Improvements?"""
         log.info(f"Asking {self.ollama_chat_model} for session reflection..."); reflection = call_ollama_api(prompt, self.ollama_chat_model, self.ollama_base_url, timeout=120)
         if reflection and reflection.strip(): print(f"\n--- Session Reflection ---\n{reflection}\n------"); self.add_self_reflection(reflection, "session_reflection")
         else: log.warning("Failed to generate session reflection.")


    # --- METHODS FOR STATE TAB (Unchanged logic, formatting handled in UI) ---
    def get_agent_dashboard_state(self) -> Dict[str, Any]:
        # ... (implementation unchanged) ...
        log.debug("Gathering dashboard state...")
        try:
            tasks_structured = self.task_queue.get_all_tasks_structured()
            memory_summary = self.memory.get_memory_summary()
            completed_failed_tasks = tasks_structured.get("completed", []) + tasks_structured.get("failed", [])
            summary_items = [f"- {m_type}: {count}" for m_type, count in memory_summary.items()]
            memory_summary_str = "**Memory Summary (by Type):**\n" + "\n".join(summary_items) if summary_items else "No memories found."
            return {
                "identity_statement": self.identity_statement,
                "pending_tasks": tasks_structured.get("pending", []),
                "in_progress_tasks": tasks_structured.get("in_progress", []),
                "completed_tasks": tasks_structured.get("completed", []),
                "failed_tasks": tasks_structured.get("failed", []),
                "completed_failed_tasks_data": completed_failed_tasks,
                "memory_summary": memory_summary_str,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        except Exception as e:
            log.exception("Error gathering agent dashboard state")
            return {
                "identity_statement": "Error fetching state",
                "pending_tasks": [], "in_progress_tasks": [], "completed_tasks": [], "failed_tasks": [],
                "completed_failed_tasks_data": [], "memory_summary": f"Error: {e}",
                 "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }

    def get_formatted_memories_for_task(self, task_id: str) -> List[Dict[str, Any]]:
        # ... (implementation unchanged) ...
        if not task_id:
            return []
        log.debug(f"Getting memories for task ID: {task_id}")
        memories = self.memory.get_memories_by_metadata(filter_dict={"task_id": task_id}, limit=100)
        formatted = []
        for mem in memories:
            metadata = mem.get('metadata', {})
            formatted.append({
                "Timestamp": metadata.get('timestamp', 'N/A'),
                "Type": metadata.get('type', 'N/A'),
                "Content Snippet": mem.get('content', '')[:200] + "..." if mem.get('content') else 'N/A',
                "ID": mem.get('id', 'N/A')
            })
        return formatted

    def get_formatted_general_memories(self) -> List[Dict[str, Any]]:
        # ... (implementation unchanged) ...
        log.debug("Getting general memories...")
        memories = self.memory.get_general_memories(limit=50)
        formatted = []
        for mem in memories:
            metadata = mem.get('metadata', {})
            formatted.append({
                "Timestamp": metadata.get('timestamp', 'N/A'),
                "Type": metadata.get('type', 'N/A'),
                "Content Snippet": mem.get('content', '')[:200] + "..." if mem.get('content') else 'N/A',
                "ID": mem.get('id', 'N/A')
            })
        return formatted