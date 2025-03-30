# autonomous_agent/agent.py
import os
import json
import time
import datetime
import uuid
import re
import traceback
from typing import List, Dict, Any, Optional, Tuple

# --- Project Imports ---
import config
from data_structures import Task
from task_manager import TaskQueue
from memory import AgentMemory # setup_chromadb is used in main.py now
from tools import load_tools
from llm_utils import call_ollama_api
import chromadb

# --- Logging Setup ---
import logging
log = logging.getLogger("AGENT")


# --- Autonomous Agent Class ---
class AutonomousAgent:
    """The main agent class orchestrating tasks, memory, tools, and LLM calls."""
    def __init__(self, memory_collection: Optional[chromadb.Collection] = None):
        log.info("Initializing AutonomousAgent...")
        self.task_queue = TaskQueue()

        if memory_collection is None:
             log.critical("Memory collection not provided during Agent initialization.")
             raise ValueError("Memory collection is required for Agent initialization.")
        self.memory = AgentMemory(memory_collection)

        self.tools = load_tools()
        self.ollama_base_url = config.OLLAMA_BASE_URL
        self.ollama_chat_model = config.OLLAMA_CHAT_MODEL
        self.ollama_embed_model = config.OLLAMA_EMBED_MODEL
        self.session_state_path = config.AGENT_STATE_PATH
        self.session_state = {"current_task_id": None, "investigation_context": "", "last_checkpoint": None}
        self.qlora_dataset_path = config.QLORA_DATASET_PATH # Store dataset path

        self.load_session_state()

        log.info("Agent Initialized:")
        log.info(f"  Chat Model: {self.ollama_chat_model} @ {self.ollama_base_url}")
        log.info(f"  Embed Model: {self.ollama_embed_model}")
        log.info(f"  Available Tools: {list(self.tools.keys())}")
        log.info(f"  QLoRA Dataset Path: {self.qlora_dataset_path}") # Log path
        if "web_search" in self.tools and not config.SEARXNG_BASE_URL:
            log.warning("  WebSearchTool enabled but SEARXNG_BASE_URL missing in config.")

    # --- Memory Retrieval with LLM Re-ranking ---
    # (No changes)
    def retrieve_and_rerank_memories(self, query: str, task_description: str, context: str, n_candidates: int = 10, n_final: int = 4) -> List[Dict[str, Any]]:
        # ... (implementation as before) ...
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

    # --- Load/Save Session State ---
    # (No changes)
    def load_session_state(self):
        # ... (implementation as before) ...
        if os.path.exists(self.session_state_path):
            try:
                with open(self.session_state_path, "r", encoding='utf-8') as f: content = f.read()
                if not content or not content.strip(): log.info(f"Session state file '{self.session_state_path}' empty."); return
                loaded = json.loads(content)
                if isinstance(loaded, dict): self.session_state.update(loaded); log.info(f"Loaded session state from {self.session_state_path}.")
                else: log.warning(f"Session state file '{self.session_state_path}' invalid format.")
            except json.JSONDecodeError as e: log.warning(f"Failed loading session state JSON '{self.session_state_path}': {e}.")
            except Exception as e: log.warning(f"Failed loading session state '{self.session_state_path}': {e}.")
        else: log.info(f"Session state file '{self.session_state_path}' not found. Using defaults.")
    def save_session_state(self):
        # ... (implementation as before) ...
        try:
            self.session_state["last_checkpoint"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with open(self.session_state_path, "w", encoding='utf-8') as f: json.dump(self.session_state, f, indent=2, ensure_ascii=False)
        except Exception as e: log.error(f"Error saving session state to '{self.session_state_path}': {e}")


    # --- Task Creation & Tool Description ---
    # (No changes)
    def create_task(self, description: str, priority: int = 1, depends_on: Optional[List[str]] = None) -> Optional[str]:
        return self.task_queue.add_task(Task(description, priority, depends_on=depends_on))
    def get_available_tools_description(self) -> str:
        # ... (implementation as before) ...
        if not self.tools: return "No tools available."
        active_tools = [];
        for name, tool in self.tools.items():
            is_active = True
            if name == "web_search" and not config.SEARXNG_BASE_URL: is_active = False
            if is_active: active_tools.append(f"- Name: {name}\n  Description: {tool.description}")
        if not active_tools: return "No tools currently active or available."
        return "\n".join(active_tools)

    # --- Generate Thinking ---
    # (No changes from previous version - includes error handling prompt and JSON fixes)
    def generate_thinking(self, task_description: str, context: str = "", tool_results: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        # ... (implementation as before) ...
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
                        try: params_json = json.loads(json_str); log.debug("PARAMS parsed directly.") # Attempt 1
                        except json.JSONDecodeError as e1:
                            log.debug(f"Direct JSON parse failed: {e1}. Fixing quotes...")
                            try: fixed_str = json_str.replace('“', '"').replace('”', '"').replace("'", '"'); log.debug(f"Attempting parse after quote fix: '{fixed_str}'"); params_json = json.loads(fixed_str); log.debug("PARAMS parsed after quote fix.") # Attempt 2
                            except json.JSONDecodeError as e2:
                                first_brace = raw_params.find('{'); last_brace = raw_params.rfind('}')
                                if first_brace != -1 and last_brace > first_brace: # Attempt 3
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

    # --- Execute Tool ---
    # (No changes)
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        # ... (implementation as before) ...
        if tool_name not in self.tools: log.error(f"Tool '{tool_name}' requested but not found."); return {"error": f"Tool '{tool_name}' not found"}
        tool = self.tools[tool_name]
        try:
            result = tool.run(parameters); log.info(f"Tool '{tool_name}' finished execution.")
            try: json.dumps(result)
            except TypeError: log.warning(f"Result from '{tool_name}' not JSON serializable."); return {"tool_name": tool_name, "result": {"unserializable_result": str(result)}, "status": "completed_unserializable"}
            return {"tool_name": tool_name, "result": result, "status": "completed"}
        except Exception as e: log.exception(f"Error executing tool '{tool_name}': {e}"); return {"tool_name": tool_name, "error": f"Tool execution failed: {e}", "status": "failed"}

    # --- NEW: Save QLoRA Datapoint ---
    def _save_qlora_datapoint(self, task_description: str, final_answer: str):
        """Formats and appends a task-answer pair to the QLoRA dataset file."""
        try:
            # Basic format: Prompt is the task, Output is the answer
            # Could be customized (e.g., add context, reflections) if needed
            datapoint = {
                # Option 1: Simple Prompt/Output
                # "prompt": f"Task: {task_description}",
                # "output": final_answer

                # Option 2: Instruction-style (potentially better for instruction tuning)
                 "instruction": "Given the following task, provide a comprehensive answer.",
                 "input": task_description,
                 "output": final_answer

                # Option 3: Alpaca format (requires input field, task desc fits here)
                # "instruction": "Provide a comprehensive answer for the following task.",
                # "input": task_description,
                # "output": final_answer
            }

            # Append as a JSON line
            with open(self.qlora_dataset_path, 'a', encoding='utf-8') as f:
                json_line = json.dumps(datapoint, ensure_ascii=False)
                f.write(json_line + '\n')
            log.info(f"QLoRA datapoint saved for task: '{task_description[:50]}...'")

        except Exception as e:
            log.exception(f"Failed to save QLoRA datapoint to {self.qlora_dataset_path}: {e}")


    # --- Process Task ---
    def process_task(self, task_id: str, max_steps: int = config.DEFAULT_MAX_STEPS_PER_TASK):
        """Processes a single task, executing steps until completion, failure, or max steps."""
        task = self.task_queue.get_task(task_id)
        if not task: log.error(f"Task {task_id} not found for processing."); return {"error": "Task not found"}
        if task.status not in ["pending", "in_progress"]: log.info(f"Task {task_id} status '{task.status}', skipping."); return {"status": "already_finished_or_failed", "task_status": task.status}

        if task.status == "pending": self.task_queue.update_task(task_id, "in_progress"); task = self.task_queue.get_task(task_id) # Re-fetch

        context = "";
        if self.session_state.get("current_task_id") == task_id and self.session_state.get("investigation_context"): context = self.session_state.get("investigation_context", ""); log.info(f"Resuming task {task_id}...")
        else: log.info(f"Starting task {task_id}: '{task.description}'"); context = f"Objective: {task.description}\n"
        self.session_state["current_task_id"] = task_id; self.session_state["investigation_context"] = context; self.save_session_state()

        last_tool_results = None; current_step = 0

        while current_step < max_steps:
            step_num = current_step + 1; print(f"\n--- Task '{task.id}' | Step {step_num}/{max_steps} ---") # Demarcation
            raw_thinking, action = self.generate_thinking(task.description, context, last_tool_results); last_tool_results = None

            if action is None: log.critical("LLM communication failed during thinking. Stopping task."); self.task_queue.update_task(task_id, "failed", reflections="LLM communication failed (thinking)."); self.session_state["current_task_id"] = None; self.session_state["investigation_context"] = ""; self.save_session_state(); return {"error": "LLM communication failed"}

            thinking_to_store = raw_thinking if raw_thinking else "Thinking process not extracted."
            self.memory.add_memory(f"Step {step_num} Thinking:\n{thinking_to_store}", {"type": "task_thinking", "task_id": task_id, "step": step_num, "action_type": action.get('type', 'unknown')})
            action_type = action.get("type", "error")

            if action_type == "error":
                err_msg = action.get('message', 'Unknown LLM action error.'); log.error(f"[ACTION] LLM Action Error: {err_msg}"); context += f"\n--- Step {step_num} Error: {err_msg} ---\nAttempting to proceed..."; self.memory.add_memory(f"Step {step_num} Error: {err_msg}", {"type": "agent_error", "task_id": task_id, "step": step_num}); current_step += 1; time.sleep(1); continue
            elif action_type == "use_tool":
                tool_name = action.get("tool"); params = action.get("parameters")
                log.info(f"[ACTION] Chosen: Use Tool '{tool_name}' with params: {json.dumps(params, ensure_ascii=False)}")
                if not tool_name or not isinstance(params, dict): log.error(f"Internal Error: Invalid 'use_tool' action: {action}. Failing task."); self.task_queue.update_task(task_id, "failed", reflections="Internal agent error: Malformed use_tool."); break
                tool_results = self.execute_tool(tool_name, params); last_tool_results = tool_results
                step_context_update = (f"\n--- Step {step_num} ---\nAction: Use Tool '{tool_name}'\nParams: {json.dumps(params, ensure_ascii=False)}\nResult:\n```json\n{json.dumps(tool_results, indent=2, ensure_ascii=False)}\n```\n"); context += step_context_update
                self.memory.add_memory(f"Step {step_num} Tool Result:\nTool: {tool_name}\nParams: {json.dumps(params, ensure_ascii=False)}\nResult: {json.dumps(tool_results, indent=2, ensure_ascii=False)}", {"type": "tool_result", "task_id": task_id, "step": step_num, "tool_name": tool_name, "result_status": tool_results.get("status", "unknown")})
            elif action_type == "final_answer":
                log.info("[ACTION] Chosen: Provide Final Answer.")
                answer = action.get("answer", "").strip(); reflections = action.get("reflections", "").strip()
                if not answer: log.warning("LLM chose 'final_answer' but no ANSWER text. Asking again."); context += f"\n--- Step {step_num} Warning: No answer text. ---\n"; self.memory.add_memory(f"Step {step_num} Warning: LLM chose final_answer but no ANSWER.", {"type": "agent_warning", "task_id": task_id, "step": step_num}); current_step += 1; continue

                # Print final answer clearly
                print("\n" + "="*15 + " FINAL ANSWER " + "="*15); print(answer); print("="*44 + "\n")
                result_payload = {"answer": answer, "steps_taken": step_num}
                self.task_queue.update_task(task_id, "completed", result=result_payload, reflections=reflections)

                # --- Save QLoRA Datapoint ---
                self._save_qlora_datapoint(task.description, answer)
                # ----------------------------

                self.memory.add_memory(f"Final Answer:\n{answer}", {"type": "task_result", "task_id": task_id, "final_step": step_num})
                if reflections: self.memory.add_memory(f"Final Reflections:\n{reflections}", {"type": "task_reflection", "task_id": task_id, "final_step": step_num})
                self.session_state["current_task_id"] = None; self.session_state["investigation_context"] = ""; self.save_session_state(); log.info(f"Task {task_id} completed successfully in {step_num} steps."); return result_payload

            context_limit = config.CONTEXT_TRUNCATION_LIMIT
            if len(context) > context_limit: log.info(f"Context length ({len(context)}) exceeded {context_limit}. Truncating..."); context = f"(Context truncated)\n...\n{context[-context_limit:]}"
            self.session_state["investigation_context"] = context; self.save_session_state()
            current_step += 1; time.sleep(1)

        # --- After the Loop ---
        if current_step >= max_steps:
            log.info(f"Task {task_id} hit max steps ({max_steps}). Remains 'in_progress'.")
            self.task_queue.update_task(task_id, "in_progress", reflections=f"Max steps ({max_steps}) reached.")
            self.session_state["investigation_context"] = context; self.save_session_state()
            return {"status": "incomplete", "reason": "max_steps_reached", "steps_taken": max_steps, "task_id": task_id }
        else: # Loop finished before max_steps (due to break)
            log.error(f"Task {task_id} processing stopped unexpectedly after step {current_step}.")
            task = self.task_queue.get_task(task_id); task_status = task.status if task else "unknown"
            if task_status == "in_progress": self.task_queue.update_task(task_id, "failed", reflections="Processing loop stopped unexpectedly.")
            self.session_state["current_task_id"] = None; self.session_state["investigation_context"] = ""; self.save_session_state()
            return {"error": "Task processing stopped unexpectedly."}

    # --- Resume & Process Next Task ---
    # (No changes)
    def resume_current_task(self, max_additional_steps: int = 5):
        # ... (implementation as before) ...
        current_task_id = self.session_state.get("current_task_id");
        if not current_task_id: log.info("No task currently in progress to resume."); return {"status": "no_task_to_resume"}
        task = self.task_queue.get_task(current_task_id)
        if not task: log.error(f"Inconsistent state: Task ID '{current_task_id}' from session not found. Clearing."); self.session_state["current_task_id"] = None; self.session_state["investigation_context"] = ""; self.save_session_state(); return {"error": "Inconsistent state."}
        if task.status != "in_progress": log.info(f"Task '{current_task_id}' status '{task.status}'. Cannot resume. Clearing state."); self.session_state["current_task_id"] = None; self.session_state["investigation_context"] = ""; self.save_session_state(); return {"status": "already_finished_or_failed", "task_status": task.status}
        log.info(f"Attempting to resume task '{current_task_id}' for up to {max_additional_steps} more steps...")
        return self.process_task(current_task_id, max_additional_steps)
    def process_next_task(self, max_steps: int = 5):
        # ... (implementation as before) ...
        next_task = self.task_queue.get_next_task()
        if not next_task: return {"status": "no_tasks_available"}
        log.info(f"\n=== Processing Next Pending Task: {next_task.id} (Prio: {next_task.priority}, Depends: {next_task.depends_on}) ===")
        log.info(f"    Description: {next_task.description}")
        self.session_state["investigation_context"] = "" # Clear context
        return self.process_task(next_task.id, max_steps)

    # --- Generate New Tasks ---
    # (No changes from previous version - includes novelty prompt)
    def generate_new_tasks(self,
                           max_new_tasks: int = 3,
                           last_user_message: Optional[str] = None,    # <<< ADDED
                           last_assistant_response: Optional[str] = None # <<< ADDED
                           ) -> Optional[str]: # Return task desc or None
        """
        Analyzes context (including optional last chat turn) and existing tasks
        to generate new task suggestions using the LLM. Returns the description
        of the first task added, if any.
        """
        log.info("\n--- Attempting to Generate New Tasks ---")
        # Determine context source (chat vs general idle)
        if last_user_message and last_assistant_response:
            log.info("Task generation triggered by chat interaction.")
            context_query = f"Last User Query: {last_user_message}\nLast Assistant Response: {last_assistant_response}"
            mem_query = f"Context relevant to last chat: {last_user_message}"
            critical_evaluation_instruction = "\n**Critically Evaluate Need:** Based *specifically* on the **Last Interaction**, is a new background task *truly necessary*? If the interaction was simple, fully resolved, or doesn't require deeper follow-up, output `[]`."
        else:
            # This path will now only be taken if called manually or from run_session (which we removed)
            log.info("Task generation triggered without specific chat context (e.g., idle state).")
            context_query = "General agent status and recent activity."
            mem_query = "Recent agent activities, conclusions, errors, reflections, open questions."
            critical_evaluation_instruction = ""

        # ... (rest of the method: retrieve memories, build prompt, call LLM, parse, add task - unchanged) ...
        log.info("Retrieving context for task generation...")
        recent_mems = self.memory.retrieve_raw_candidates(query=mem_query, n_results=15)
        mem_summary = "\n".join([f"- Type: {m['metadata'].get('type','mem')}, Time: {m['metadata'].get('timestamp', 'N/A')}\n  Content: {m['content'][:150].strip()}..." for m in recent_mems]) if recent_mems else "No recent memories relevant to this context."
        existing_tasks_info = [{"id": t.id, "description": t.description} for t in self.task_queue.tasks.values() if t.status in ["pending", "in_progress"]]
        existing_tasks_summary = "\n".join([f"- ID: {t['id']}\n  Desc: {t['description'][:100]}..." for t in existing_tasks_info]) if existing_tasks_info else "None"
        valid_existing_task_ids = set(t['id'] for t in existing_tasks_info)
        prompt = f"""You are the planning component of an AI agent. Generate new, actionable tasks based on the agent's context.\n\n**Current Context Focus:**\n{context_query}\n\n**Recent Activity & Memory Snippets:**\n{mem_summary}\n\n**Existing Pending/In-Progress Tasks (with IDs):**\n{existing_tasks_summary}{critical_evaluation_instruction}\n\n**Your Task:** Review context. Identify gaps, next steps, tangents *relevant to Context Focus*. Generate up to {max_new_tasks} *new, specific, actionable* task descriptions.\n**Include a Mix (if appropriate):** 1. **Follow-up Tasks:** Directly address context. Use `depends_on` with *existing* IDs if needed. 2. **Exploratory Task (ONE max, if relevant):** Explore related tangent *only if strongly suggested*. \n**Guidelines:** Actionable (verb), Specific, Novel (no duplicates), Relevant, Concise. **Dependencies:** `depends_on`: [list of *existing* task IDs] only if logically required.\n**Output Format (Strict JSON):** Provide *only* a valid JSON list of objects. Each: {{"description": "...", "priority": (opt, 1-10, def 5), "depends_on": (opt, list_of_existing_ids)}}. Example: `[ {{"description": "Summarize task abc.", "depends_on": ["abc"]}}, {{"description": "Explore X."}} ]`. Output `[]` if no tasks needed."""
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
                priority = task_data.get("priority", 5)
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
                        # if validated_dependencies: log.info(f"  Task '{description[:50]}...' depends on: {validated_dependencies}") # Logged by add_task
                    else: log.warning(f"Invalid 'depends_on' format for '{description[:50]}...'. Ignoring.")
                new_task_id = self.create_task(description, priority, depends_on=validated_dependencies or None)
                if new_task_id:
                     if new_tasks_added == 0: first_task_desc_added = description # Store first one
                     new_tasks_added += 1
                     existing_tasks_info.append({"id": new_task_id, "description": description})
                     valid_existing_task_ids.add(new_task_id); current_task_ids_in_batch.add(new_task_id)
                if new_tasks_added >= max_new_tasks: break
        except json.JSONDecodeError as e: log.error(f"Failed JSON parse for task gen: {e}\nLLM Resp:\n{llm_response}\n---"); return None
        except Exception as e: log.exception(f"Error processing task gen response: {e}"); return None
        log.info(f"Finished Task Generation: Added {new_tasks_added} new tasks.")
        return first_task_desc_added # Return description or None

    # --- Generate and Save Activity Summary ---
    # (No changes)
    def _generate_and_save_activity_summary(self, session_stats: Dict[str, Any]):
        # ... (implementation as before) ...
        log.info("Generating session activity summary...")
        try:
            summary_folder = config.SUMMARY_FOLDER; os.makedirs(summary_folder, exist_ok=True)
            today_str = datetime.date.today().isoformat(); summary_filename = os.path.join(summary_folder, f"summary_{today_str}.txt")
            recent_mems = self.memory.retrieve_raw_candidates(query="Session reflections, task results, critical errors.", n_results=5)
            mem_summary_str = "\nRecent Key Memories:\n"
            if recent_mems: mem_summary_str += "".join([f"- ({mem['metadata'].get('type','mem')}) {mem['content'][:200].strip()}...\n" for mem in recent_mems])
            else: mem_summary_str += "  (None retrieved)\n"
            summary_lines = [f"\n--- Session Summary: {datetime.datetime.now(datetime.timezone.utc).isoformat()} ---", f"Session Start: {session_stats.get('start_time', 'N/A')}", f"Session End:   {session_stats.get('end_time', 'N/A')}", f"Duration:      {session_stats.get('duration_minutes', 0):.2f} minutes", f"Tasks Processed (Attempts): {session_stats.get('tasks_processed', 0)}", f"Tasks Completed:          {session_stats.get('tasks_completed', 0)}", f"Tasks Failed:             {session_stats.get('tasks_failed', 0)}", f"New Tasks Generated:      {session_stats.get('new_tasks_generated', 0)}", mem_summary_str, f"--- End Session Summary ---"]
            summary = "\n".join(summary_lines) + "\n";
            with open(summary_filename, 'a', encoding='utf-8') as f: f.write(summary)
            log.info(f"Appended session summary to {summary_filename}")
        except Exception as e: log.exception(f"Failed to generate or save session summary: {e}")

    # --- Run Session & Reflection ---
    # (No changes)
    def run_session(self, duration_minutes: int = config.DEFAULT_SESSION_DURATION_MINUTES, max_steps_per_task: int = config.DEFAULT_MAX_STEPS_PER_TASK):
        # ... (implementation as before, including session stat tracking) ...
        start_time = datetime.datetime.now(datetime.timezone.utc); end_time = start_time + datetime.timedelta(minutes=duration_minutes)
        print(f"\n{'='*10} Starting Agent Session {'='*10}")
        log.info(f"Start Time: {start_time.isoformat()}")
        log.info(f"Target End Time: ~{end_time.isoformat()} ({duration_minutes} min)")
        log.info(f"Max Steps per Task Cycle: {max_steps_per_task}")
        tasks_processed_count = 0; tasks_completed_count = 0; tasks_failed_count = 0; new_tasks_generated_count = 0
        session_active = True; active_task_id: Optional[str] = self.session_state.get("current_task_id")
        consecutive_no_new_task_cycles = 0
        if active_task_id:
            log.info(f"Attempting initial resume of task: {active_task_id}")
            process_result = self.process_task(active_task_id, max_steps_per_task); tasks_processed_count += 1
            if isinstance(process_result, dict):
                if process_result.get("answer"): tasks_completed_count += 1; active_task_id = None
                elif process_result.get("status") == "incomplete": active_task_id = process_result.get("task_id")
                elif process_result.get("error"): tasks_failed_count += 1; active_task_id = None
                else: active_task_id = None
            else: tasks_failed_count += 1; active_task_id = None
            if datetime.datetime.now(datetime.timezone.utc) >= end_time: log.info("Session time expired during initial resume."); session_active = False
        while session_active and datetime.datetime.now(datetime.timezone.utc) < end_time:
            process_result = None
            if active_task_id: process_result = self.process_task(active_task_id, max_steps_per_task); consecutive_no_new_task_cycles = 0
            else: process_result = self.process_next_task(max_steps_per_task); consecutive_no_new_task_cycles = 0
            if isinstance(process_result, dict):
                status = process_result.get("status"); task_id_processed = process_result.get("task_id")
                if status == "no_tasks_available":
                    log.info("No pending tasks available.")
                    if consecutive_no_new_task_cycles >= 2: log.warning("Task generation yielded no new tasks twice. Pausing longer..."); time.sleep(30); consecutive_no_new_task_cycles = 0; continue
                    tasks_added = self.generate_new_tasks(); new_tasks_generated_count += tasks_added
                    if tasks_added == 0: consecutive_no_new_task_cycles += 1; log.info("Task generation added 0 tasks. Will check/pause."); time.sleep(5)
                    continue
                if status != "no_tasks_available": tasks_processed_count += 1
                if process_result.get("error"): log.warning(f"Task processing error: {process_result['error']}."); tasks_failed_count += 1; active_task_id = None; time.sleep(2)
                elif process_result.get("answer"): tasks_completed_count += 1; active_task_id = None
                elif status == "incomplete": active_task_id = task_id_processed; log.info(f"Task {active_task_id} hit max steps, will continue.")
                elif status == "already_finished_or_failed": active_task_id = None
                else: log.warning(f"Unhandled task status: {status}. Clearing active task."); active_task_id = None
            else: log.warning(f"Unexpected result type from task processing: {type(process_result)}. Clearing active task."); tasks_failed_count += 1; active_task_id = None; time.sleep(5)
            if datetime.datetime.now(datetime.timezone.utc) >= end_time: print("\n[INFO] Session time expired."); session_active = False
        end_session_time = datetime.datetime.now(datetime.timezone.utc); total_duration_minutes = (end_session_time - start_time).total_seconds() / 60
        print(f"\n{'='*10} Agent Session Ended {'='*10}")
        log.info(f"End Time: {end_session_time.isoformat()}")
        log.info(f"Actual Duration: {total_duration_minutes:.2f} minutes")
        log.info(f"Tasks Processed (Attempts): {tasks_processed_count}")
        log.info(f"Tasks Completed: {tasks_completed_count}")
        log.info(f"Tasks Failed: {tasks_failed_count}")
        log.info(f"New Tasks Generated: {new_tasks_generated_count}")
        session_stats = {"start_time": start_time.isoformat(), "end_time": end_session_time.isoformat(), "duration_minutes": total_duration_minutes, "tasks_processed": tasks_processed_count, "tasks_completed": tasks_completed_count, "tasks_failed": tasks_failed_count, "new_tasks_generated": new_tasks_generated_count}
        self._generate_and_save_activity_summary(session_stats)
        self.generate_and_add_session_reflection(start_time, end_session_time, tasks_completed_count, tasks_processed_count)
        return session_stats

    # --- Add Self Reflection ---
    # (No changes)
    def add_self_reflection(self, reflection: str, reflection_type: str = "self_reflection"):
        # ... (implementation as before) ...
        if not reflection or not isinstance(reflection, str) or not reflection.strip(): log.warning("Attempted to add empty reflection."); return None
        log.info(f"Adding {reflection_type} to memory...")
        return self.memory.add_memory(reflection, {"type": reflection_type})

    # --- Generate Session Reflection ---
    # (No changes)
    def generate_and_add_session_reflection(self, start: datetime.datetime, end: datetime.datetime, completed_count: int, processed_count: int):
        # ... (implementation as before) ...
        duration_minutes = (end - start).total_seconds() / 60
        log.info("Retrieving context for session reflection...")
        recent_mems = self.memory.retrieve_raw_candidates(query="Summary of session activities, errors, outcomes.", n_results=15)
        mem_summary = "\n".join([f"- {m['metadata'].get('type','mem')}: {m['content'][:100].strip()}..." for m in recent_mems]) if recent_mems else "None"
        prompt = f"""You are the AI agent. Reflect on your work session.\n**Session Details:** Start: {start.isoformat()}, End: {end.isoformat()}, Duration: {duration_minutes:.1f} min, Processed: {processed_count}, Completed: {completed_count}.\n**Recent Activity Snippets:**\n{mem_summary}\n\n**Reflection Task:** Provide concise reflection: 1. Progress? 2. Efficiency/Effectiveness? 3. Challenges/Errors? 4. Insights/Learnings? 5. Improvements?"""
        log.info(f"Asking {self.ollama_chat_model} for session reflection...")
        reflection = call_ollama_api(prompt, self.ollama_chat_model, self.ollama_base_url, timeout=120)
        if reflection and reflection.strip():
            print("\n--- Generated Session Reflection ---"); print(reflection); print("--- End Reflection ---") # Print reflection
            self.add_self_reflection(reflection, "session_reflection")
        else: log.warning("Failed to generate session reflection.")