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