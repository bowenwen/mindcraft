# FILE: prompts.py
"""
Stores the multi-line LLM prompts used by the AutonomousAgent.
"""

# --- Identity Revision Prompt ---
# Variables: {identity_statement}, {reason}, {memory_context}
# CORRECTED: Use single braces for .format() substitution
REVISE_IDENTITY_PROMPT = """
You are an AI agent reflecting on your identity. Your goal is to revise your personal identity statement based on your recent experiences and purpose.

**Current Identity Statement:**
{identity_statement}

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

# --- General Thinking/Action Prompt ---
# Variables: {identity_statement}, {task_description}, {tool_desc}, {memory_context_str}
# Conditional additions in agent.py: USER SUGGESTION, User Provided Info, Context, Last Results
# CORRECTED: Use single braces for .format() substitution
GENERATE_THINKING_PROMPT_BASE = """
You are an autonomous AI agent. Your primary goal is to achieve the **Overall Task** by deciding the single best next step, while considering your identity and user interactions.

**Your Current Identity:**
{identity_statement}

**Overall Task:**
{task_description}

**Available Tools & Actions:**
{tool_desc}

**Relevant Memories (Re-ranked, consider recency indicated by '[X time ago]', novelty):**
{memory_context_str}"""
# Note: The rest of the prompt (User Suggestion, User Info, Context, Last Action Results, Task Now, Output Format)
# is appended dynamically in agent.py:generate_thinking based on conditions.


# --- Task Summarization Prompt ---
# Variables: {task_status}, {summary_context}, {context_truncation_limit}
# CORRECTED: Use single braces for .format() substitution, including nested format specifier
SUMMARIZE_TASK_PROMPT = """
Summarize the execution of this agent task based on the log below. Focus on objective, key actions/findings, errors/recovery, and final outcome ({task_status}).

**Input Data:**
{summary_context:.{context_truncation_limit}}  # Truncate input context if needed

**Output:** Concise summary text only.
"""


# --- New Task Generation Prompt ---
# Variables: {identity_statement}, {context_query}, {mem_summary}, {active_tasks_summary},
#            {completed_failed_summary}, {critical_evaluation_instruction}, {max_new_tasks}
# CORRECTED: Use single braces for .format() substitution
GENERATE_NEW_TASKS_PROMPT = """
You are the planning component of an agent. Generate new, actionable tasks based on state, history, and identity.

**Agent's Current Identity:**
{identity_statement}

**Current Context Focus:**
{context_query}

**Recent Activity & Memory Snippets (Consider recency indicated by '[X time ago]'):**
{mem_summary}

**Existing Pending/In-Progress Tasks (Check duplicates!):**
{active_tasks_summary}

**Recently Finished Tasks (for context):**
{completed_failed_summary}

{critical_evaluation_instruction}

**Your Task:** Review all info (paying attention to recency of memories). Identify gaps/next steps relevant to Context & Identity. Generate up to {max_new_tasks} new, specific, actionable tasks that require using the agent's tools (web search/browse, memory search/write, file read/write/list, status report). AVOID DUPLICATES of pending/in-progress tasks. Assign priority (1-10, consider identity). Add necessary `depends_on` using existing IDs only. Suggest follow-up, refinement, or (max 1) exploratory tasks.

**Guidelines:** Actionable (verb start, often implying tool use), Specific, Novel, Relevant (to context/identity), Concise.

**Output Format (Strict JSON):** Provide *only* a valid JSON list of objects (or `[]`). Example:
```json
[
  {{"description": "Search the web for recent news on AI agent memory techniques.", "priority": 7}},
  {{"description": "Read the contents of the 'summary.txt' artifact.", "priority": 6, "depends_on": ["xyz"]}},
  {{"description": "Write a brief reflection on the challenges encountered in task [abc] to memory.", "priority": 4, "depends_on": ["abc"]}},
  {{"description": "Get the current agent status report.", "priority": 2}}
]
```
"""

# --- Session Reflection Prompt ---
# Variables: {identity_statement}, {start_iso}, {end_iso}, {duration_minutes},
#            {completed_count}, {processed_count}, {mem_summary}
# CORRECTED: Use single braces for .format() substitution
SESSION_REFLECTION_PROMPT = """
You are AI agent ({identity_statement}). Reflect on your work session.

**Start:** {start_iso}
**End:** {end_iso}
**Duration:** {duration_minutes:.1f} min
**Tasks Completed:** {completed_count}
**Tasks Processed:** {processed_count}

**Recent Activity/Identity Notes (Consider recency indicated by '[X time ago]'):**
{mem_summary}

**Reflection Task:** Provide concise reflection:
1. Accomplishments?
2. Efficiency?
3. Challenges/errors?
4. Learnings?
5. Alignment with identity/goals?
6. Improvements?
"""

# --- Memory Re-ranking Prompt ---
# Variables: {identity_statement}, {task_description}, {query}, {context},
#            {candidate_details}, {n_final}
# CORRECTED: Use single braces for .format() substitution
RERANK_MEMORIES_PROMPT = """
You are an AI assistant helping an agent select the MOST useful memories for its current step. Goal: Choose the best memories to help the agent achieve its immediate goal, considering relevance, recency, and avoiding redundancy.

**Agent's Stated Identity:**
{identity_statement}

**Current Task:**
{task_description}

**Agent's Current Context/Goal for this Step (Use this heavily for ranking):**
{query}

**Recent Conversation/Action History (Check for relevance!):**
{context}

**Candidate Memories (with index, relative time, distance, type, and content snippet):**
{candidate_details}

**Instructions:** Review the **Agent's Current Context/Goal**, **Task**, **Identity**, and **History**. Based on this, identify the **{n_final} memories** (by index) from the list above that are MOST RELEVANT and MOST USEFUL for the agent to consider *right now*.

**CRITICAL Ranking Factors:**
1.  **Relevance:** How directly does the memory address the **Current Context/Goal**?
2.  **Recency:** How recently was the memory created (use the **[relative time]** provided)? More recent memories are often, but not always, more relevant.
3.  **Novelty/Uniqueness:** Does the memory offer information not already present or obvious from other highly-ranked memories or the current context? Avoid selecting multiple memories that say essentially the same thing.

**Balance these factors** to select the optimal set of {n_final} memories.

**Output Format:** Provide *only* a comma-separated list of the numerical indices (starting from 0) of the {n_final} most relevant memories, ordered from most relevant to least relevant. Example: 3, 0, 7, 5
"""