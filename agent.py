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
        # --- Updated state ---
        self.session_state = {
            "current_task_id": None,
            "investigation_context": "",
            "current_task_retries": 0, # Track consecutive retries for the current step
            "last_checkpoint": None,
            "last_web_browse_content": None
        }
        self.qlora_dataset_path = config.QLORA_DATASET_PATH
        self.load_session_state() # Load previous state first

        # --- State for Background Loop and UI Updates ---
        self._is_running = threading.Event()
        self._shutdown_request = threading.Event()
        self._agent_thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
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
                    log.warning(f"Attempted to update unknown UI state key: {key}")

    def get_ui_update_state(self) -> Dict[str, Any]:
        """Safely retrieves the current UI state."""
        with self._state_lock:
            return self._ui_update_state.copy()

    # --- Memory, State, Task, Tool Methods ---
    def retrieve_and_rerank_memories(self, query: str, task_description: str, context: str, n_candidates: int = 10, n_final: int = 4) -> List[Dict[str, Any]]:
        if not query or not isinstance(query, str) or not query.strip(): return []
        log.info(f"Retrieving & re-ranking memories. Query: '{query[:50]}...'")
        candidates = self.memory.retrieve_raw_candidates(query, n_results=n_candidates)
        if not candidates: log.info("No initial memory candidates found for re-ranking."); return []
        if len(candidates) <= n_final: log.info(f"Fewer candidates ({len(candidates)}) than requested ({n_final}). Returning all."); return candidates
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
                    # --- Ensure NEW state keys exist with defaults ---
                    self.session_state.setdefault("last_web_browse_content", None)
                    self.session_state.setdefault("current_task_retries", 0)
                    # --------------------------------------------------
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
            # Disable web search if SearXNG URL is not configured
            if name == "web_search" and not config.SEARXNG_BASE_URL:
                log.debug("Disabling 'web_search' tool as SEARXNG_BASE_URL is not set.")
                is_active = False
            # Could add checks for other tools requiring config/setup here
            if is_active: active_tools.append(f"- Name: {name}\n  Description: {tool.description}")
        if not active_tools: return "No tools currently active or available (check configuration)."
        return "\n".join(active_tools)

    def generate_thinking(self, task_description: str, context: str = "", tool_results: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Generates the agent's thinking process and determines the next action.
        Includes enhanced error handling instructions in the prompt and more robust parsing.
        """
        tool_desc = self.get_available_tools_description()
        memory_query = f"Info/past actions relevant to next step? Goal: {context[-500:]}\nTask: {task_description}"
        if tool_results: memory_query += f"\nLast result summary: {str(tool_results)[:200]}"
        relevant_memories = self.retrieve_and_rerank_memories(query=memory_query, task_description=task_description, context=context, n_candidates=10, n_final=4)
        memory_context_str = "\n\n".join([f"Relevant Memory (ID: {mem.get('id', 'N/A')}, Type: {mem['metadata'].get('type', 'N/A')}, Time: {mem['metadata'].get('timestamp', 'N/A')}, Dist: {mem.get('distance', 'N/A'):.4f}):\n{mem['content']}" for mem in relevant_memories]) if relevant_memories else "No relevant memories selected."

        # --- Prompt remains the same as the previous version with error handling ---
        prompt = f"""You are an autonomous AI agent. Your primary goal is to achieve the **Overall Task** by deciding the single best next step.\n
**Overall Task:**\n{task_description}\n
**Available Tools:**\n{tool_desc}\n
**Relevant Memories (Re-ranked):**\n{memory_context_str}\n
**Current Investigation Context (History - IMPORTANT: May contain previous errors from this task):**\n{context if context else "First step for this task."}\n"""
        if tool_results: prompt += f"\n**Results from Last Action:**\nTool: {tool_results.get('tool_name', 'Unknown')}\nResult:\n```json\n{json.dumps(tool_results.get('result', {}), indent=2, ensure_ascii=False)}\n```\n"
        else: prompt += "\n**Results from Last Action:**\nNone.\n"
        prompt += f"""**Your Task Now:**
1.  **Analyze:** Review ALL provided information (Task, Tools, Memories, Context, Last Result). Pay close attention to any errors mentioned in the **Current Investigation Context**.
2.  **Reason:** Determine the single best action to take *right now* to make progress towards the **Overall Task**.
3.  **CRITICAL: Handle Previous Errors:** If the **Current Investigation Context** mentions an error from a previous attempt on this task (e.g., 'Step X Error:', 'Tool failed', 'Invalid parameters'), **DO NOT GIVE UP**. Acknowledge the error briefly in your thinking, then focus on a **RECOVERY STRATEGY**. Examples:
    *   If a tool failed: Try the tool again, perhaps with modified parameters based on the error message. Or, try a different tool that might achieve a similar goal.
    *   If parameters were invalid: Correct the parameters and try the same tool again.
    *   If search results were poor: Rephrase the search query, make it more specific or broader.
    *   If web browse failed: Try searching for the information instead, or find an alternative source URL.
    **Only use `final_answer` when you are confident you have fully achieved the *original* Overall Task objective.** Do not use `final_answer` simply because you encountered an error.
4.  **Choose Action:** Decide between `use_tool` or `final_answer`.

**Output Format (Strict JSON-like structure):**
THINKING:
<Your reasoning process here. Explain your analysis of the situation, especially your plan if recovering from a previous error.>

NEXT_ACTION: <"use_tool" or "final_answer">

If NEXT_ACTION is "use_tool":
TOOL: <The exact name of the tool from the 'Available Tools' list, e.g., web_browse>
PARAMETERS: <A **valid JSON object** containing the parameters for the tool. Use **standard double quotes (")** for all keys and string values. Example for web_browse: {{"url": "https://example.com"}}> Example for web_search: {{"query": "latest AI research"}}>

If NEXT_ACTION is "final_answer":
ANSWER: <Your complete and comprehensive answer to the **Overall Task**.>
REFLECTIONS: <Optional: Any final thoughts or reflections on the task execution.>

**Critical Formatting Reminders:**
*   Start *immediately* with "THINKING:".
*   Follow the structure exactly.
*   `PARAMETERS` *must* be a valid JSON object using double quotes. The structure MUST match the tool's requirements (e.g., `{{"url": "..."}}` for web_browse).
*   `ANSWER` must directly address the original **Overall Task**.
"""
        log.info(f"Asking {self.ollama_chat_model} for next action (with enhanced error handling guidance)...")
        llm_response_text = call_ollama_api(prompt, self.ollama_chat_model, self.ollama_base_url, timeout=180)

        if llm_response_text is None:
            log.error("Failed to get thinking response from Ollama.")
            return "LLM communication failed.", {"type": "error", "message": "LLM communication failure (thinking).", "subtype": "llm_comm_error"}

        # --- Enhanced Parsing Logic ---
        try:
            action: Dict[str, Any] = {"type": "unknown"}; raw_thinking = llm_response_text
            thinking_marker = "THINKING:"; action_marker = "NEXT_ACTION:"; tool_marker = "TOOL:"; params_marker = "PARAMETERS:"; answer_marker = "ANSWER:"; reflections_marker = "REFLECTIONS:"
            thinking_start = llm_response_text.find(thinking_marker); action_start = llm_response_text.find(action_marker)

            if thinking_start != -1: end_think = action_start if action_start > thinking_start else len(llm_response_text); raw_thinking = llm_response_text[thinking_start + len(thinking_marker):end_think].strip()
            else: log.warning("'THINKING:' marker not found in LLM response."); raw_thinking = llm_response_text

            action_type_str = ""
            if action_start != -1: end_action = llm_response_text.find('\n', action_start); end_action = end_action if end_action != -1 else len(llm_response_text); action_type_str = llm_response_text[action_start + len(action_marker):end_action].strip()
            else: log.error("'NEXT_ACTION:' marker not found."); action = {"type": "error", "message": "Missing NEXT_ACTION marker.", "subtype": "parse_error"}; return raw_thinking, action

            if "use_tool" in action_type_str:
                action["type"] = "use_tool"; tool_name = None; params_json = None
                tool_start = llm_response_text.find(tool_marker, action_start); params_start = llm_response_text.find(params_marker, action_start)

                if tool_start != -1:
                     end_tool = llm_response_text.find('\n', tool_start)
                     if params_start > tool_start and (end_tool == -1 or params_start < end_tool): end_tool = params_start
                     end_tool = end_tool if end_tool != -1 else len(llm_response_text)
                     # --- Robust Tool Name Cleaning ---
                     tool_name = llm_response_text[tool_start + len(tool_marker):end_tool].strip().strip('"').strip("'")
                     log.debug(f"Extracted tool name: '{tool_name}'")
                     # ----------------------------------
                else:
                     log.error(f"'TOOL:' marker not found after 'NEXT_ACTION: use_tool'.")
                     action = {"type": "error", "message": "Missing TOOL marker for use_tool action.", "subtype": "parse_error"}
                     return raw_thinking, action

                if params_start != -1:
                    params_str_start = params_start + len(params_marker)
                    end_params = len(llm_response_text)
                    next_marker_pos = -1
                    for marker in [f"\n{answer_marker}", f"\n{reflections_marker}", f"\n{thinking_marker}", f"\n{action_marker}", f"\n{tool_marker}", f"\n{params_marker}"]:
                         pos = llm_response_text.find(marker, params_str_start)
                         if pos != -1 and (next_marker_pos == -1 or pos < next_marker_pos): next_marker_pos = pos
                    if next_marker_pos != -1: end_params = next_marker_pos

                    raw_params = llm_response_text[params_str_start:end_params].strip()
                    raw_params = re.sub(r'^```json\s*', '', raw_params, flags=re.I|re.M); raw_params = re.sub(r'^```\s*', '', raw_params, flags=re.M); raw_params = re.sub(r'\s*```$', '', raw_params, flags=re.M); raw_params = raw_params.strip()
                    json_str = raw_params; params_json = None
                    log.debug(f"Raw PARAMS string after cleaning markers/fences: '{json_str}'")

                    if json_str:
                        try:
                            params_json = json.loads(json_str)
                            log.debug(f"PARAMS parsed directly as JSON: {params_json}")
                        except json.JSONDecodeError as e1:
                            log.warning(f"Direct JSON parse failed: {e1}. Attempting fixes...")
                            try:
                                fixed_str = json_str.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")
                                fixed_str = re.sub(r",\s*([}\]])", r"\1", fixed_str) # Remove trailing commas
                                log.debug(f"Attempting parse after fixes: '{fixed_str}'")
                                params_json = json.loads(fixed_str)
                                log.debug(f"PARAMS parsed after fixes: {params_json}")
                            except json.JSONDecodeError as e2:
                                first_brace = raw_params.find('{'); last_brace = raw_params.rfind('}')
                                if first_brace != -1 and last_brace > first_brace:
                                    extracted_str = raw_params[first_brace : last_brace + 1]; log.debug(f"Extracting content between braces: '{extracted_str}'")
                                    try:
                                         fixed_extracted_str = extracted_str.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")
                                         fixed_extracted_str = re.sub(r",\s*([}\]])", r"\1", fixed_extracted_str)
                                         log.debug(f"Trying extracted content after fixes: '{fixed_extracted_str}'");
                                         params_json = json.loads(fixed_extracted_str); log.debug(f"PARAMS parsed after extraction + fixes: {params_json}")
                                    except json.JSONDecodeError as e3: err_msg = f"Invalid JSON in PARAMETERS after all fix attempts: {e3}. Original: '{raw_params}'"; log.error(err_msg); action = {"type": "error", "message": err_msg, "subtype": "parse_error"}; return raw_thinking, action
                                else: err_msg = f"Invalid JSON in PARAMETERS: {e2}. Fix attempt: '{fixed_str if 'fixed_str' in locals() else json_str}'"; log.error(err_msg); action = {"type": "error", "message": err_msg, "subtype": "parse_error"}; return raw_thinking, action
                    else: log.warning("PARAMETERS block empty after cleaning."); action = {"type": "error", "message": "Empty PARAMETERS block.", "subtype": "parse_error"}; return raw_thinking, action
                else:
                     log.error(f"'PARAMETERS:' marker not found after 'NEXT_ACTION: use_tool'.")
                     action = {"type": "error", "message": "Missing PARAMETERS marker for use_tool action.", "subtype": "parse_error"}
                     return raw_thinking, action

                # --- Final Validation for use_tool Action ---
                if action.get("type") == "use_tool":
                    if not tool_name:
                         # This case should be caught earlier by marker checks, but belts and braces...
                         err_msg = "Internal Error: Tool name missing despite parsing attempt."; log.error(err_msg); action = {"type": "error", "message": err_msg, "subtype": "parse_error"}
                    elif tool_name not in self.tools:
                         err_msg = f"Tool '{tool_name}' specified by LLM is not available/loaded."; log.error(err_msg); action = {"type": "error", "message": err_msg, "subtype": "invalid_tool"}
                    elif not isinstance(params_json, dict):
                        # This check is crucial as subsequent checks assume a dict
                        err_msg = f"Parsed PARAMETERS is not a JSON object (dict), but {type(params_json)}. Value: {params_json}"; log.error(err_msg); action = {"type": "error", "message": err_msg, "subtype": "parse_error"}
                    else:
                        # --- Tool-Specific Parameter Validation ---
                        valid_params = True
                        if tool_name == "web_browse":
                            if "url" not in params_json or not isinstance(params_json.get("url"), str) or not params_json["url"].strip():
                                err_msg = "Missing or invalid 'url' parameter (must be a non-empty string) for web_browse tool."
                                log.error(err_msg + f" Params received: {params_json}")
                                action = {"type": "error", "message": err_msg, "subtype": "invalid_params"}
                                valid_params = False
                        elif tool_name == "web_search":
                             if "query" not in params_json or not isinstance(params_json.get("query"), str) or not params_json["query"].strip():
                                err_msg = "Missing or invalid 'query' parameter (must be a non-empty string) for web_search tool."
                                log.error(err_msg + f" Params received: {params_json}")
                                action = {"type": "error", "message": err_msg, "subtype": "invalid_params"}
                                valid_params = False
                        # --- Add validation for other tools here ---
                        # elif tool_name == "some_other_tool":
                        #    ... check params_json ...

                        if valid_params:
                             # Success case for use_tool parsing
                             action["tool"] = tool_name
                             action["parameters"] = params_json
                             log.info(f"Successfully parsed action: Use Tool '{tool_name}' with params: {params_json}")

            elif "final_answer" in action_type_str:
                # --- Parsing logic for final_answer remains the same ---
                action["type"] = "final_answer"; answer = ""; reflections = ""
                answer_start = llm_response_text.find(answer_marker, action_start); reflections_start = llm_response_text.find(reflections_marker, action_start)

                if answer_start != -1:
                     end_answer = reflections_start if reflections_start > answer_start else len(llm_response_text)
                     answer = llm_response_text[answer_start + len(answer_marker):end_answer].strip()
                else:
                    log.warning("ANSWER marker not found after 'final_answer'. Trying to infer.")
                    potential_answer_start = llm_response_text.find('\n', action_start) + 1
                    potential_answer_end = reflections_start if reflections_start > potential_answer_start else len(llm_response_text)
                    if potential_answer_start > 0 and potential_answer_start < potential_answer_end:
                         answer = llm_response_text[potential_answer_start:potential_answer_end].strip()
                    else:
                         answer = ""

                if reflections_start != -1: reflections = llm_response_text[reflections_start + len(reflections_marker):].strip()

                action["answer"] = answer; action["reflections"] = reflections
                if not answer: log.warning("LLM chose 'final_answer' but ANSWER text is missing or empty."); action = {"type": "error", "message": "LLM chose final_answer but provided no ANSWER text.", "subtype": "parse_error"}
                elif len(answer) < 150 and re.search(r'\b(error|fail|cannot|unable|issue|problem)\b', answer, re.IGNORECASE) and not re.search(r'recovery|retry|alternative', raw_thinking or "", re.IGNORECASE):
                    log.warning(f"LLM 'final_answer' looks like giving up without attempting recovery: '{answer[:100]}...'")
                    action = {"type": "error", "message": f"LLM gave up instead of recovering: {answer}", "subtype": "give_up_error"}
                else:
                    log.info(f"Successfully parsed action: Final Answer.")


            elif action_type_str: # Handle unexpected action strings
                 log.error(f"Unrecognized NEXT_ACTION value: '{action_type_str}'")
                 action = {"type": "error", "message": f"Invalid NEXT_ACTION specified: '{action_type_str}'", "subtype": "parse_error"}

            # If action['type'] is still 'unknown' after all checks (shouldn't happen if logic is sound)
            if action["type"] == "unknown" and action.get("subtype") != "parse_error" : # Avoid overwriting specific parse errors
                 err_msg = "Could not parse a valid action ('use_tool' or 'final_answer') from the LLM response."; log.error(f"{err_msg}\nLLM Response:\n{llm_response_text}"); action = {"type": "error", "message": err_msg, "subtype": "parse_error"}

            # Return the thinking process and the parsed (or error) action
            return raw_thinking, action
        except Exception as e:
            log.exception(f"CRITICAL: Unexpected failure parsing LLM thinking response: {e}\nResponse:\n{llm_response_text}")
            # Return the raw thinking (if any) and a generic internal error action
            return raw_thinking or "Error during parsing.", {"type": "error", "message": f"Internal error parsing LLM response: {e}", "subtype": "internal_error"}

    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
         """Executes a tool and handles potential exceptions during execution."""
         if tool_name not in self.tools:
             log.error(f"Tool '{tool_name}' requested but not found/loaded.")
             return {"error": f"Tool '{tool_name}' not found", "status": "failed"}
         tool = self.tools[tool_name];
         log.info(f"Executing tool '{tool_name}' with params: {parameters}")
         try:
             result = tool.run(parameters) # The tool's run method should return a dict
             log.info(f"Tool '{tool_name}' finished execution.")

             # Basic check for dictionary result
             if not isinstance(result, dict):
                 log.warning(f"Tool '{tool_name}' did not return a dictionary. Result: {result}")
                 return {"tool_name": tool_name, "result": {"unexpected_result": str(result)}, "status": "completed_malformed_output"}

             # Check if the tool itself reported an error internally
             if result.get("error"):
                 log.warning(f"Tool '{tool_name}' reported an error: {result['error']}")
                 # Return the tool's error, keeping status as 'failed'
                 return {"tool_name": tool_name, "error": result["error"], "status": "failed"}

             # Attempt to JSON serialize for safety check, but don't modify original dict if successful
             try: json.dumps(result)
             except TypeError as json_err:
                 log.warning(f"Result from '{tool_name}' not JSON serializable: {json_err}. Returning string representation.")
                 # Return a structure indicating success but with unserializable data
                 return {"tool_name": tool_name, "result": {"unserializable_result": str(result)}, "status": "completed_unserializable"}

             # If no internal error and serializable, assume completed successfully
             return {"tool_name": tool_name, "result": result, "status": "completed"}

         except Exception as e:
             log.exception(f"CRITICAL Error executing tool '{tool_name}' with params {parameters}: {e}")
             # Return a failure status with the exception message
             return {"tool_name": tool_name, "error": f"Tool execution raised exception: {e}", "status": "failed"}

    def _save_qlora_datapoint(self, source_type: str, instruction: str, input_context: str, output: str):
        if not output: log.warning("Skipping QLoRA datapoint save due to empty output."); return
        try:
            datapoint = {"instruction": instruction, "input": input_context, "output": output, "source_type": source_type}
            # Append as JSON Lines
            with open(self.qlora_dataset_path, 'a', encoding='utf-8') as f:
                json_line = json.dumps(datapoint, ensure_ascii=False)
                f.write(json_line + '\n')
            log.info(f"QLoRA datapoint saved (Source: {source_type})")
        except Exception as e: log.exception(f"Failed to save QLoRA datapoint to {self.qlora_dataset_path}: {e}")


    # --- NEW: Memory Summarization Method ---
    def _summarize_and_prune_task_memories(self, task: Task):
        """Generates a summary of a completed/failed task and optionally prunes its detailed memories."""
        if not config.ENABLE_MEMORY_SUMMARIZATION:
            log.debug(f"Memory summarization disabled. Skipping for task {task.id}.")
            return

        log.info(f"Summarizing memories for {task.status} task {task.id}...")
        task_memories = self.memory.get_memories_by_metadata(filter_dict={"task_id": task.id})

        if not task_memories:
            log.info(f"No memories found for task {task.id}. Skipping summarization.")
            return

        log.debug(f"Found {len(task_memories)} memories to potentially summarize/prune for task {task.id}.")

        # Prepare context for summary generation
        summary_context = f"Task Description: {task.description}\n"
        summary_context += f"Task Final Status: {task.status}\n"
        if task.result: summary_context += f"Task Final Result/Answer: {str(task.result)[:500]}\n" # Truncate long results
        if task.reflections: summary_context += f"Task Reflections: {task.reflections}\n"
        summary_context += "\n--- Task Memory Log (Chronological Order Might Be Approximate) ---\n"

        # Sort memories roughly by timestamp if available, otherwise keep order
        try:
            task_memories.sort(key=lambda m: m.get('metadata', {}).get('timestamp', ''))
        except:
            log.warning("Could not sort task memories by timestamp.") # Ignore sort error

        mem_details = []
        memory_ids_to_prune = []
        for mem in task_memories:
            meta = mem.get('metadata', {})
            mem_type = meta.get('type', 'memory')
            # Don't include previous summaries in the context for the *new* summary
            if mem_type == 'task_summary':
                continue

            timestamp = meta.get('timestamp', 'N/A')
            content = mem.get('content', '')
            mem_details.append(f"[{timestamp} - {mem_type}]\n{content}\n-------")
            memory_ids_to_prune.append(mem['id']) # Add ID to potential prune list

        summary_context += "\n".join(mem_details)
        summary_context += "\n--- End Memory Log ---"

        summary_prompt = f"""You are an AI assistant tasked with summarizing the execution of a completed agent task. Based on the task description, final status/result, and the detailed memory log below, provide a concise summary.

**Task Summary Request:**
*   Briefly state the task objective.
*   Summarize the key actions taken by the agent (tools used, main findings).
*   Mention any significant errors or recovery attempts if applicable.
*   State the final outcome ({task.status}).

**Input Data:**
{summary_context[:config.CONTEXT_TRUNCATION_LIMIT // 2]}

**Output:**
Provide only the concise summary text.
""" # Limit context size for summary prompt

        log.info(f"Asking {self.ollama_chat_model} to summarize task {task.id}...")
        summary_text = call_ollama_api(summary_prompt, self.ollama_chat_model, self.ollama_base_url, timeout=150)

        if summary_text and summary_text.strip():
            summary_metadata = {
                "type": "task_summary",
                "task_id": task.id,
                "original_status": task.status,
                "summarized_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "original_memory_count": len(memory_ids_to_prune) # Store how many memories this summarizes
            }
            summary_id = self.memory.add_memory(f"Task Summary ({task.status}):\n{summary_text}", summary_metadata)

            if summary_id and config.DELETE_MEMORIES_AFTER_SUMMARY:
                log.info(f"Summary added (ID: {summary_id}). Pruning {len(memory_ids_to_prune)} original memories for task {task.id}...")
                if not memory_ids_to_prune:
                     log.warning("No memory IDs found to prune, although summary was generated.")
                else:
                     deleted = self.memory.delete_memories(memory_ids_to_prune)
                     if not deleted:
                         log.error(f"Failed to prune original memories for task {task.id} after adding summary.")
            elif summary_id:
                 log.info(f"Summary added (ID: {summary_id}). Deletion of original memories is disabled.")
            else:
                 log.error(f"Failed to add task summary memory for task {task.id}. Original memories were not pruned.")
        else:
            log.warning(f"Failed to generate summary for task {task.id}. Original memories were not pruned.")


    def _execute_step(self, task: Task) -> Dict[str, Any]:
        """
        Executes a single thinking/action step for the given task.
        Includes retry logic for recoverable errors.
        """
        step_start_time = time.time()
        step_log = []
        final_answer_text = None
        step_status = "processing" # Initial status for the step
        task_status_updated_this_step = False # Track if task status changed (completed/failed)

        current_context = self.session_state.get("investigation_context", "")
        step_num = self.session_state.get(f"task_{task.id}_step", 0) + 1
        current_retries = self.session_state.get("current_task_retries", 0) # Get retries for this task

        step_log.append(f"--- Task '{task.id}' | Step {step_num} (Retry attempt {current_retries}/{config.AGENT_MAX_STEP_RETRIES}) ---")
        log.info(f"Executing step {step_num} for task {task.id} (Retry: {current_retries})")

        # --- Generate Thinking and Action ---
        # Tool results from PREVIOUS step are not passed here; context has that info.
        raw_thinking, action = self.generate_thinking(task.description, current_context, None)

        # --- Handle Action/Errors ---
        action_type = action.get("type", "error") # Default to error if type missing
        action_message = action.get("message", "Unknown error")
        action_subtype = action.get("subtype", "unknown_error") # More specific error category

        thinking_to_store = raw_thinking if raw_thinking else "Thinking process not extracted or LLM failed."
        step_log.append(f"Thinking:\n{thinking_to_store}")
        # Store thinking regardless of action outcome, include action type/subtype if known
        self.memory.add_memory(
            f"Step {step_num} Thinking (Action Type: {action_type}, Subtype: {action_subtype}):\n{thinking_to_store}",
            {"type": "task_thinking", "task_id": task.id, "step": step_num, "action_type": action_type, "action_subtype": action_subtype}
        )

        # --- Branch based on action type ---
        if action_type == "error":
            log.error(f"[STEP ERROR] LLM Action Error: {action_message} (Subtype: {action_subtype})")
            step_log.append(f"[ERROR] Action Error: {action_message} (Type: {action_subtype})")

            # Add error details to context for next attempt
            error_context = f"\n--- Step {step_num} Error (Retry {current_retries+1}/{config.AGENT_MAX_STEP_RETRIES}): {action_message} (Type: {action_subtype}) ---"
            current_context += error_context
            self.memory.add_memory(f"Step {step_num} Error: {action_message}", {"type": "agent_error", "task_id": task.id, "step": step_num, "error_subtype": action_subtype})

            # --- Retry Logic ---
            if current_retries < config.AGENT_MAX_STEP_RETRIES:
                self.session_state["current_task_retries"] = current_retries + 1
                log.warning(f"Recoverable error encountered. Incrementing retry count to {current_retries + 1}. Task will continue.")
                step_status = "error_retry" # Indicates a recoverable error occurred
            else:
                log.error(f"Maximum step retries ({config.AGENT_MAX_STEP_RETRIES}) reached for task {task.id}. Failing task.")
                self.task_queue.update_task(task.id, "failed", reflections=f"Failed after {step_num} steps. Max retries reached on error: {action_message}")
                self.session_state["current_task_id"] = None
                self.session_state["investigation_context"] = ""
                # Reset retries for the *next* task
                self.session_state["current_task_retries"] = 0
                step_status = "failed" # This step caused the task failure
                task_status_updated_this_step = True
                # --- Attempt memory summarization on failure ---
                if config.ENABLE_MEMORY_SUMMARIZATION:
                    try: self._summarize_and_prune_task_memories(task)
                    except Exception as summ_err: log.exception(f"Error during memory summarization for failed task {task.id}: {summ_err}")
                # ---------------------------------------------

        elif action_type == "use_tool":
            tool_name = action.get("tool")
            params = action.get("parameters")
            log.info(f"[ACTION] Chosen: Use Tool '{tool_name}'")
            step_log.append(f"[ACTION] Use Tool: {tool_name} with params: {json.dumps(params, ensure_ascii=False, indent=2)}")

            # Already validated tool_name and params format in generate_thinking parsing
            tool_results = self.execute_tool(tool_name, params)
            tool_status = tool_results.get("status", "unknown")
            tool_error = tool_results.get("error")

            step_context_update = (
                f"\n--- Step {step_num} ---\nAction: Use Tool '{tool_name}'\nParams: {json.dumps(params, ensure_ascii=False)}\n"
                f"Result Status: {tool_status}\nResult:\n```json\n{json.dumps(tool_results, indent=2, ensure_ascii=False)}\n```\n"
            )
            current_context += step_context_update
            self.memory.add_memory(
                f"Step {step_num} Tool Action & Result (Status: {tool_status}):\n{step_context_update}",
                {"type": "tool_result", "task_id": task.id, "step": step_num, "tool_name": tool_name, "result_status": tool_status, "params": json.dumps(params)} # Store params too
            )

            if tool_status == 'completed':
                step_log.append(f"[INFO] Tool '{tool_name}' completed successfully.")
                step_status = "processing" # Ready for next step
                self.session_state["current_task_retries"] = 0 # Reset retries on successful step
                if tool_name == 'web_browse':
                    self.session_state["last_web_browse_content"] = tool_results.get('result', {}).get('content', '(No content found/extracted)')
            elif tool_status == 'failed':
                log.warning(f"Tool '{tool_name}' failed: {tool_error}")
                step_log.append(f"[ERROR] Tool '{tool_name}' failed: {tool_error}")
                # --- Retry Logic for Tool Failure ---
                if current_retries < config.AGENT_MAX_STEP_RETRIES:
                    self.session_state["current_task_retries"] = current_retries + 1
                    log.warning(f"Tool failure is recoverable. Incrementing retry count to {current_retries + 1}. Task will continue.")
                    step_status = "error_retry" # Allow retry
                else:
                    log.error(f"Maximum step retries ({config.AGENT_MAX_STEP_RETRIES}) reached after tool failure for task {task.id}. Failing task.")
                    self.task_queue.update_task(task.id, "failed", reflections=f"Failed after {step_num} steps. Max retries reached on tool failure: {tool_error}")
                    self.session_state["current_task_id"] = None
                    self.session_state["investigation_context"] = ""
                    self.session_state["current_task_retries"] = 0 # Reset for next task
                    step_status = "failed"
                    task_status_updated_this_step = True
                    # --- Attempt memory summarization on failure ---
                    if config.ENABLE_MEMORY_SUMMARIZATION:
                       try: self._summarize_and_prune_task_memories(task)
                       except Exception as summ_err: log.exception(f"Error during memory summarization for failed task {task.id}: {summ_err}")
                    # ---------------------------------------------
            else:
                # Handle unexpected tool statuses (e.g., completed_unserializable)
                log.warning(f"Tool '{tool_name}' completed with status '{tool_status}'. Treating as success for step progression.")
                step_status = "processing" # Continue processing
                self.session_state["current_task_retries"] = 0 # Reset retries

        elif action_type == "final_answer":
            log.info("[ACTION] Chosen: Provide Final Answer.")
            step_log.append("[ACTION] Provide Final Answer.")
            answer = action.get("answer", "").strip(); reflections = action.get("reflections", "").strip()

            final_answer_text = answer # Store for UI update
            print("\n" + "="*15 + f" FINAL ANSWER (Task {task.id}) " + "="*15); print(answer); print("="* (34+len(str(task.id))) + "\n") # Use str(task.id)
            result_payload = {"answer": answer, "steps_taken": step_num};
            self.task_queue.update_task(task.id, "completed", result=result_payload, reflections=reflections)
            task_status_updated_this_step = True # Task is finished
            self._save_qlora_datapoint(source_type="task_completion", instruction="Given the following task, provide a comprehensive answer based on the investigation.", input_context=task.description, output=answer)
            self.memory.add_memory(f"Final Answer:\n{answer}", {"type": "task_result", "task_id": task.id, "final_step": step_num})
            if reflections: self.memory.add_memory(f"Final Reflections:\n{reflections}", {"type": "task_reflection", "task_id": task.id, "final_step": step_num})

            self.session_state["current_task_id"] = None; self.session_state["investigation_context"] = ""
            self.session_state["current_task_retries"] = 0 # Reset retries for next task
            step_status = "completed"

            # --- Attempt memory summarization on completion ---
            if config.ENABLE_MEMORY_SUMMARIZATION:
                try:
                    # Fetch the updated task object which includes the result/reflections
                    completed_task = self.task_queue.get_task(task.id)
                    if completed_task:
                        self._summarize_and_prune_task_memories(completed_task)
                    else:
                        log.warning(f"Cannot find completed task {task.id} in queue for summarization.")
                except Exception as summ_err:
                    log.exception(f"Error during memory summarization for completed task {task.id}: {summ_err}")
            # ------------------------------------------------

        # --- Update Context and State ---
        if step_status in ["processing", "error_retry"]: # Continue task if processing or recoverable error
            context_limit = config.CONTEXT_TRUNCATION_LIMIT
            if len(current_context) > context_limit:
                log.info(f"Context length ({len(current_context)}) exceeded {context_limit}. Truncating...")
                # Truncate from the beginning, keeping the end
                trunc_point = len(current_context) - context_limit
                current_context = f"(Context truncated)\n...\n{current_context[trunc_point:]}"
            self.session_state["investigation_context"] = current_context
            self.session_state[f"task_{task.id}_step"] = step_num
            # Retries are handled above and saved with session state
        elif step_status in ["completed", "failed"]: # Task finished
             self.session_state.pop(f"task_{task.id}_step", None) # Clean up step counter
             # Retries and context already reset above

        self.save_session_state() # Save state after each step attempt
        step_duration = time.time() - step_start_time
        step_log.append(f"Step {step_num} duration: {step_duration:.2f}s. Step Status: {step_status}")

        # Get the task's *final* status after this step (might have changed)
        final_task_status = self.task_queue.get_task(task.id).status if task else "unknown"

        # --- Return UI update data ---
        return {
            "status": final_task_status, # Report task's current status
            "log": "\n".join(step_log),
            "current_task_id": self.session_state.get("current_task_id"), # Might be None if task finished
            "current_task_desc": task.description if self.session_state.get("current_task_id") else "None",
            "recent_memories": self.memory.retrieve_raw_candidates(query=f"Activity related to task {task.id}, step {step_num}", n_results=5), # Query relevant to *this* step
            "last_web_content": self.session_state.get("last_web_browse_content", "(No recent web browse)"),
            "final_answer": final_answer_text # Only populated if action was final_answer
        }


    def process_one_step(self) -> Dict[str, Any]:
        """Processes a single step of the highest priority runnable task. Returns UI state dict."""
        log.info("Processing one autonomous step...")
        step_result_log = ["Attempting to process one step..."]
        default_state = {
            "status": "idle", "log": "Agent idle.", "current_task_id": None,
            "current_task_desc": "None", "recent_memories": [],
            "last_web_content": self.session_state.get("last_web_browse_content", "(No recent web browse)"),
            "final_answer": None
        }

        try:
            task_id_to_process = self.session_state.get("current_task_id")
            task: Optional[Task] = None

            if task_id_to_process:
                task = self.task_queue.get_task(task_id_to_process)
                if not task or task.status not in ["in_progress", "pending"]: # Allow resuming pending if state got weird
                    log.warning(f"Task {task_id_to_process} from state invalid or finished ({task.status if task else 'Not Found'}). Searching for next.")
                    step_result_log.append(f"Task {task_id_to_process} invalid/finished.")
                    task = None; self.session_state["current_task_id"] = None
                    self.session_state["investigation_context"] = "" # Clear context if task invalid
                    self.session_state["current_task_retries"] = 0 # Reset retries

            if not task:
                task = self.task_queue.get_next_task()
                if task:
                    log.info(f"Starting next pending task: {task.id} - '{task.description[:60]}...'")
                    step_result_log.append(f"Starting task {task.id}: {task.description[:60]}...")
                    self.task_queue.update_task(task.id, "in_progress")
                    self.session_state["current_task_id"] = task.id
                    # Initialize context and state for the new task
                    self.session_state["investigation_context"] = f"Objective: {task.description}\n"
                    self.session_state.pop(f"task_{task.id}_step", None) # Remove old step counter if exists
                    self.session_state["current_task_retries"] = 0 # Reset retries for new task
                    self.save_session_state()
                    task = self.task_queue.get_task(task.id) # Re-fetch task with 'in_progress' status
                else:
                    log.info("No runnable tasks found.")
                    step_result_log.append("No runnable tasks found.")
                    # Generate exploratory tasks if idle?
                    if not self.session_state.get("current_task_id"): # Only generate if truly idle
                        log.info("Agent idle, considering generating exploratory tasks...")
                        generated_desc = self.generate_new_tasks(max_new_tasks=1) # Generate one task
                        if generated_desc:
                             step_result_log.append(f"Generated exploratory task: {generated_desc[:60]}...")
                        else:
                             step_result_log.append("No new exploratory tasks generated.")

                    self._update_ui_state(status="idle", log="\n".join(step_result_log), current_task_id=None, current_task_desc="None")
                    return self.get_ui_update_state()

            # --- Execute one step ---
            if task: # Ensure task is valid before executing step
                 step_result = self._execute_step(task)
                 # Update the shared state with the result of the step
                 self._update_ui_state(**step_result)
                 return self.get_ui_update_state() # Return the latest state
            else:
                 log.error("Task became None unexpectedly before step execution.")
                 step_result_log.append("[ERROR] Internal error: Task lost before execution.")
                 self._update_ui_state(status="error", log="\n".join(step_result_log))
                 return self.get_ui_update_state()


        except Exception as e:
            log.exception("CRITICAL Error during process_one_step loop")
            step_result_log.append(f"[CRITICAL ERROR] during step processing loop: {e}")
            current_task_id = self.session_state.get("current_task_id")
            if current_task_id: # Attempt to fail the current task if a critical loop error occurs
                 log.error(f"Failing task {current_task_id} due to critical loop error.")
                 self.task_queue.update_task(current_task_id, "failed", reflections=f"Critical agent loop error: {e}")
                 self.session_state["current_task_id"] = None
                 self.session_state["investigation_context"] = ""
                 self.session_state["current_task_retries"] = 0
                 self.save_session_state()
                 # Optionally summarize memory here too?
                 failed_task = self.task_queue.get_task(current_task_id)
                 if failed_task and config.ENABLE_MEMORY_SUMMARIZATION:
                     try: self._summarize_and_prune_task_memories(failed_task)
                     except Exception as summ_err: log.exception(f"Error during memory summarization for critically failed task {current_task_id}: {summ_err}")

            self._update_ui_state(status="critical_error", log="\n".join(step_result_log), current_task_id=None, current_task_desc="None")
            return self.get_ui_update_state() # Return error state

    def generate_new_tasks(self, max_new_tasks: int = 3, last_user_message: Optional[str] = None, last_assistant_response: Optional[str] = None) -> Optional[str]:
        """Generates new tasks based on context, recent memories, and optionally chat."""
        log.info("\n--- Attempting to Generate New Tasks ---")
        context_source = ""
        if last_user_message and last_assistant_response:
            context_source = "last chat interaction"
            context_query = f"Last User Query: {last_user_message}\nLast Assistant Response: {last_assistant_response}"
            mem_query = f"Context relevant to last chat: {last_user_message}"
            critical_evaluation_instruction = "\n**Critically Evaluate Need:** Based *specifically* on the **Last Interaction**, is a new background task *truly necessary*? Avoid redundant tasks. If the chat resolved the issue or only a simple follow-up is needed, output `[]`."
        else:
            context_source = "general agent state (idle or post-task)"
            log.info(f"Task generation triggered by {context_source}.")
            context_query = "General agent status and recent activity. Consider if any logical follow-up or exploration is warranted based on completed tasks or idle state."
            mem_query = "Recent agent activities, conclusions, errors, reflections, completed task summaries, open questions."
            critical_evaluation_instruction = "\n**Critically Evaluate Need:** Are new tasks genuinely needed for exploration or follow-up, or is the current state satisfactory? Avoid creating trivial or redundant tasks. If no tasks needed, output `[]`."

        log.info(f"Retrieving context for task generation (Source: {context_source})...")
        recent_mems = self.memory.retrieve_raw_candidates(query=mem_query, n_results=15)
        mem_summary = "\n".join([f"- Type: {m['metadata'].get('type','mem')}, Time: {m['metadata'].get('timestamp', 'N/A')}\n  Content: {m['content'][:150].strip()}..." for m in recent_mems]) if recent_mems else "No recent memories relevant."

        existing_tasks_info = [{"id": t.id, "description": t.description, "status": t.status} for t in self.task_queue.tasks.values()]
        active_tasks_summary = "\n".join([f"- ID: {t['id']} (Status: {t['status']})\n  Desc: {t['description'][:100]}..." for t in existing_tasks_info if t['status'] in ['pending', 'in_progress']]) if any(t['status'] in ['pending', 'in_progress'] for t in existing_tasks_info) else "None"
        completed_failed_summary = "\n".join([f"- ID: {t['id']} (Status: {t['status']})\n  Desc: {t['description'][:100]}..." for t in existing_tasks_info if t['status'] in ['completed', 'failed']])[-1000:] # Last 1000 chars of finished tasks

        valid_existing_task_ids = set(t['id'] for t in existing_tasks_info) # All known IDs for dependency check

        # --- CORRECTED PROMPT F-STRING ---
        prompt = f"""You are the planning component of an AI agent. Your goal is to generate new, actionable tasks based on the agent's current state and recent history.

**Current Context Focus:**
{context_query}

**Recent Activity & Memory Snippets:**
{mem_summary}

**Existing Pending/In-Progress Tasks (Check for duplicates!):**
{active_tasks_summary}

**Recently Finished Tasks (for context):**
{completed_failed_summary}
{critical_evaluation_instruction}

**Your Task:**
1.  Review the **Current Context Focus**, **Recent Activity**, and **Existing Tasks**.
2.  Identify potential gaps, logical next steps, or interesting exploratory tangents *directly relevant* to the Context Focus.
3.  Generate up to **{max_new_tasks}** *new, specific, and actionable* task descriptions.
4.  **AVOID DUPLICATES:** Do not suggest tasks that are identical or semantically very similar to existing *pending* or *in-progress* tasks.
5.  **Prioritize:** Assign a priority (1-10, default 5, higher is more important).
6.  **Dependencies:** If a new task *logically requires* another task to be finished first, add its ID to `depends_on`. Use *only existing* task IDs. Dependencies should be necessary, not just related.
7.  **Task Types (Suggest a Mix if Appropriate):**
    *   **Follow-up Tasks:** Build directly on the results or failures of recently completed tasks.
    *   **Refinement Tasks:** Improve or elaborate on a previous result.
    *   **Exploratory Tasks (Max 1 per generation):** Investigate a related area *only if strongly suggested* by the context and not already covered.

**Guidelines:**
*   **Actionable:** Start with a verb (e.g., "Analyze...", "Summarize...", "Compare...", "Investigate...").
*   **Specific:** Clearly define the goal. Avoid vague requests.
*   **Novel:** Ensure it's not effectively a duplicate of an active task.
*   **Relevant:** Directly related to the current context focus.
*   **Concise:** Keep descriptions clear and to the point.

**Output Format (Strict JSON):**
Provide *only* a valid JSON list of objects. Each object represents a task. Output `[]` if no new tasks are needed or appropriate.
Example:
```json
[
  {{  # <-- Escaped brace
    "description": "Summarize the key findings from the completed analysis task [ID xyz].",
    "priority": 7,
    "depends_on": ["xyz"]
  }}, # <-- Escaped brace
  {{  # <-- Escaped brace
    "description": "Investigate alternative methods for [specific topic from context].",
    "priority": 5
  }}  # <-- Escaped brace
]
```
"""
        # --- END OF CORRECTED PROMPT F-STRING ---

        log.info(f"Asking {self.ollama_chat_model} to generate up to {max_new_tasks} new tasks...")
        llm_response = call_ollama_api(prompt, self.ollama_chat_model, self.ollama_base_url, timeout=180)
        if not llm_response: log.error("Failed to get response from LLM for task generation."); return None

        first_task_desc_added = None; new_tasks_added = 0
        try:
            # Clean potential markdown fences
            llm_response = re.sub(r'^```json\s*', '', llm_response, flags=re.I|re.M)
            llm_response = re.sub(r'\s*```$', '', llm_response, flags=re.M).strip()

            # Find the outermost list brackets
            list_start = llm_response.find('[')
            list_end = llm_response.rfind(']')
            if list_start == -1 or list_end == -1 or list_end < list_start:
                if "no new tasks" in llm_response.lower() or llm_response.strip() == "[]":
                    log.info("LLM indicated no new tasks are needed.")
                    return None
                else:
                    raise json.JSONDecodeError(f"JSON list '[]' not found in response: {llm_response}", llm_response, 0)

            json_str = llm_response[list_start : list_end + 1]
            suggested_tasks = json.loads(json_str)

            if not isinstance(suggested_tasks, list): log.warning(f"LLM task gen response not a list: {suggested_tasks}"); return None
            if not suggested_tasks: log.info("LLM suggested no new tasks."); return None

            log.info(f"LLM suggested {len(suggested_tasks)} tasks. Validating and adding...")
            current_task_ids_in_batch = set()

            active_task_descriptions = {t['description'].strip().lower() for t in existing_tasks_info if t['status'] in ['pending', 'in_progress']}

            for task_data in suggested_tasks:
                if not isinstance(task_data, dict): log.warning(f"Skipping invalid item (not dict): {task_data}"); continue

                description = task_data.get("description")
                if not description or not isinstance(description, str) or not description.strip(): log.warning(f"Skipping task with invalid description: {task_data}"); continue
                description = description.strip()

                is_duplicate = description.lower() in active_task_descriptions
                if is_duplicate: log.warning(f"Skipping potential duplicate task: '{description[:80]}...'"); continue

                priority = task_data.get("priority", 5);
                try: priority = max(1, min(10, int(priority)))
                except: priority = 5

                dependencies_raw = task_data.get("depends_on"); validated_dependencies = []
                if dependencies_raw:
                    if isinstance(dependencies_raw, list):
                        for dep_id in dependencies_raw:
                            dep_id_str = str(dep_id).strip()
                            if dep_id_str in valid_existing_task_ids or dep_id_str in current_task_ids_in_batch:
                                validated_dependencies.append(dep_id_str)
                            else:
                                log.warning(f"Ignoring invalid/unknown dependency ID '{dep_id_str}' for task '{description[:50]}...'")
                    else: log.warning(f"Invalid 'depends_on' format for '{description[:50]}...' (should be a list). Ignoring.")

                new_task = Task(description, priority, depends_on=validated_dependencies or None)
                new_task_id = self.task_queue.add_task(new_task)
                if new_task_id:
                     if new_tasks_added == 0: first_task_desc_added = description
                     new_tasks_added += 1
                     active_task_descriptions.add(description.lower())
                     valid_existing_task_ids.add(new_task_id)
                     current_task_ids_in_batch.add(new_task_id)
                     log.info(f"Added Task {new_task_id}: '{description[:60]}...' (Prio: {priority}, Depends: {validated_dependencies})")
                else:
                    log.error(f"Failed to add task '{description[:60]}...' to queue.")

                if new_tasks_added >= max_new_tasks: break

        except json.JSONDecodeError as e: log.error(f"Failed JSON parse for task gen response: {e}\nLLM Resp:\n{llm_response}\n---"); return None
        except Exception as e: log.exception(f"Unexpected error processing task generation response: {e}"); return None

        log.info(f"Finished Task Generation: Added {new_tasks_added} new tasks.");
        return first_task_desc_added


    def _autonomous_loop(self, initial_delay: float = 2.0, step_delay: float = 5.0):
        """The main loop for autonomous task processing, runs in a thread."""
        log.info("Background agent loop starting.")
        time.sleep(initial_delay)

        while not self._shutdown_request.is_set():
            if not self._is_running.is_set():
                # Ensure UI state reflects pause accurately
                with self._state_lock:
                    if self._ui_update_state["status"] != "paused":
                        log.info("Agent loop pausing...")
                        self._ui_update_state["status"] = "paused"
                        self._ui_update_state["log"] = "Agent loop paused."
                time.sleep(1) # Sleep while paused
                continue

            # Agent is running
            log.debug("Autonomous loop: Processing step...")
            step_start = time.monotonic()
            try:
                # process_one_step handles task selection, execution (_execute_step),
                # state updates, and returns the state for the UI.
                self.process_one_step()

            except Exception as e:
                # This catches errors *outside* process_one_step's own try/except
                log.exception(f"CRITICAL UNHANDLED ERROR in autonomous loop iteration.")
                # Attempt to update UI state to reflect critical error
                self._update_ui_state(status="critical_error", log=f"CRITICAL LOOP ERROR: {e}")
                # Consider pausing or stopping the agent after a critical loop error?
                # For now, just log and sleep longer.
                time.sleep(step_delay * 5) # Longer sleep after critical error

            step_duration = time.monotonic() - step_start
            remaining_delay = max(0, step_delay - step_duration)
            log.debug(f"Step took {step_duration:.2f}s. Sleeping for {remaining_delay:.2f}s.")
            time.sleep(remaining_delay) # Ensure minimum delay between steps

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
            self._update_ui_state(status="running", log="Agent started.") # Update state immediately
            self._agent_thread = threading.Thread(target=self._autonomous_loop, daemon=True)
            self._agent_thread.start()


    def pause_autonomous_loop(self):
        """Signals the background loop to pause."""
        if self._is_running.is_set():
            log.info("Pausing agent loop...")
            self._is_running.clear() # Loop will see this and update status in its next check
            # Status update happens within the loop itself now
        else:
            log.info("Agent loop is already paused.")

    def shutdown(self):
        """Signals the background loop to stop and waits for it to join."""
        log.info("Shutdown requested for agent background thread.")
        self._shutdown_request.set() # Signal loop to exit
        self._is_running.clear()    # Ensure it stops processing if paused

        if self._agent_thread and self._agent_thread.is_alive():
            log.info("Waiting for agent thread to join...")
            self._agent_thread.join(timeout=15) # Increased timeout slightly
            if self._agent_thread.is_alive():
                log.warning("Agent thread did not join cleanly after 15 seconds.")
            else:
                log.info("Agent thread joined successfully.")
        else:
            log.info("Agent thread was not running or already joined.")

        # Final UI state update
        self._update_ui_state(status="shutdown", log="Agent shut down.")
        # Save final state? Maybe not necessary if saved per step
        # self.save_session_state() # Optionally save state one last time
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
         # Query for summaries and results too
         recent_mems = self.memory.retrieve_raw_candidates(query="Summary of session activities, task summaries, errors, outcomes.", n_results=20)
         mem_summary = "\n".join([f"- {m['metadata'].get('type','mem')}: {m['content'][:100].strip()}..." for m in recent_mems]) if recent_mems else "None"
         prompt = f"""You are the AI agent. Reflect on your work session/period.\n**Period Start:** {start.isoformat()}\n**Period End:** {end.isoformat()}\n**Duration:** {duration_minutes:.1f} min\n**Tasks Completed:** {completed_count}\n**Tasks Processed (incl. failed):** {processed_count}\n**Recent Activity Snippets & Task Summaries:**\n{mem_summary}\n\n**Reflection Task:** Provide concise reflection on the session: 1. Key accomplishments or progress? 2. Overall efficiency? 3. Significant challenges or errors encountered? 4. Key insights or learnings? 5. Potential improvements for next time?"""
         log.info(f"Asking {self.ollama_chat_model} for session reflection...")
         reflection = call_ollama_api(prompt, self.ollama_chat_model, self.ollama_base_url, timeout=120)
         if reflection and reflection.strip():
             print("\n--- Generated Session Reflection ---"); print(reflection); print("--- End Reflection ---")
             self.add_self_reflection(reflection, "session_reflection")
         else: log.warning("Failed to generate session reflection.")