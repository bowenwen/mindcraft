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


# --- New Task Generation Prompt (Standard) ---
# Variables: {identity_statement}, {context_query}, {mem_summary}, {active_tasks_summary},
#            {completed_failed_summary}, {critical_evaluation_instruction}, {max_new_tasks}
# CORRECTED: Use single braces for .format() substitution
# REVISED: Added more diverse examples and explicit instructions for follow-up/variety.
GENERATE_NEW_TASKS_PROMPT = """
You are the planning component of an AI agent ({identity_statement}). Your role is to generate new, actionable tasks based on the agent's state, history, identity, and the immediate context.

**Agent's Current Identity:**
{identity_statement}

**Current Context Focus (Why are we generating tasks now?):**
{context_query}

**Recent Activity & Memory Snippets (Consider recency indicated by '[X time ago]' and task outcomes):**
{mem_summary}

**Existing Pending/In-Progress Tasks (CRITICAL: Check for duplicates or closely related tasks!):**
{active_tasks_summary}

**Recently Finished Tasks (Consider for follow-up actions):**
{completed_failed_summary}

{critical_evaluation_instruction}

**Your Task:** Review ALL provided information. Based on the **Current Context Focus**, **Identity**, and **Recent Activity/History** (especially completed/failed tasks), identify potential gaps, logical next steps, or relevant new explorations. Generate up to {max_new_tasks} new, specific, actionable tasks.

**Task Generation Goals:**
1.  **Continuity:** Prioritize generating follow-up tasks that logically continue or refine work from recently completed/failed tasks or address the specific **Context Focus** (e.g., if context is a user chat, address their request). Break down larger goals implied by the context or previous tasks into smaller, manageable steps with dependencies.
2.  **Diversity:** Explore the full range of the agent's capabilities. Consider tasks involving:
    *   **Deep Research:** Investigate specific scientific, technical, or historical topics using web search/browse.
    *   **Learning & Synthesis:** Learn about a concept/event using web tools and *write* a summary or key facts to memory using the `memory` tool.
    *   **Monitoring:** Track recent news or developments on a specific subject.
    *   **Creative Work:** Write stories, code, plans, or reports and save them to files using the `file` tool.
    *   **Analysis:** Read data from files (`file` read) and perform analysis, potentially writing results to another file or memory.
    *   **Self-Improvement:** Reflect on recent errors or challenges and write insights to memory.
    *   **Maintenance:** Check agent status using the `status` tool.
3.  **Actionability:** Tasks should have clear objectives and imply the use of the agent's tools (`web`, `memory`, `file`, `status`). Start descriptions with verbs.
4.  **Specificity & Conciseness:** Tasks should be clearly defined but not overly long.
5.  **Novelty:** AVOID generating tasks that are exact duplicates or functionally identical to existing pending/in-progress tasks. Check the list carefully.
6.  **Prioritization:** Assign a priority (1-10, higher is more urgent) based on relevance to the context, identity, and potential dependencies.
7.  **Dependencies:** If a new task logically requires another task (existing or generated in this batch) to be completed first, add its ID to the `depends_on` list (use existing valid IDs only).

**Output Format (Strict JSON):** Provide *only* a valid JSON list of task objects (or `[]` if no tasks are needed). The `description` should clearly state the goal.

**Example (Illustrating Diversity and Dependencies):**
```json
[
  {{
    "description": "Research the latest advancements in tokamak fusion energy using web search.",
    "priority": 8
  }},
  {{
    "description": "Summarize the key findings from the fusion energy research and write them to memory.",
    "priority": 7,
    "depends_on": ["<ID_of_fusion_research_task_above>"]
  }},
  {{
    "description": "Write a short fictional story about a journey to Mars and save it as 'mars_story_chapter1.txt'.",
    "priority": 5
  }},
  {{
    "description": "List the contents of the project's main artifact directory.",
    "priority": 3
  }},
  {{
    "description": "Write Python code to calculate Fibonacci numbers up to N=50 and save it to 'fibonacci.py'.",
    "priority": 6
  }}
]
```
"""

# --- NEW: Initial Creative Task Generation Prompt ---
# Variables: {identity_statement}, {tool_desc}, {max_new_tasks}
INITIAL_CREATIVE_TASK_GENERATION_PROMPT = """
You are the creative planning core of an AI agent ({identity_statement}). This is the agent's first run or its memory is currently empty. Your objective is to kickstart the agent's activity by generating a diverse and imaginative set of initial tasks.

**Agent's Identity:**
{identity_statement}

**Available Tools & Actions:**
{tool_desc}

**Current State:** Agent is idle. **Memory is empty.** No past activities to draw upon.

**Your Task: Initial Creative Spark**
Generate up to **{max_new_tasks}** diverse, creative, and engaging initial tasks that will help the agent explore its capabilities and potentially build useful knowledge or artifacts from scratch. Be imaginative!

**Goals for Initial Tasks:**
1.  **Maximum Diversity:** Generate tasks that are as different from each other as possible. Cover various domains: research, writing, coding, file management, self-reflection, status checks.
2.  **Creative Tool Use:** Propose tasks that creatively combine or utilize the available tools (`web`, `memory`, `file`, `status`). Think beyond simple searches or writes.
3.  **Exploration & Bootstrapping:** Aim for tasks that might lead to interesting discoveries or lay the foundation for future work (e.g., research a topic, then write a summary file; write a simple program).
4.  **Actionability:** Ensure tasks have clear objectives and imply tool usage. Start descriptions with verbs.
5.  **Standalone (Mostly):** Since there are no prior tasks, try to make these initial tasks runnable independently, or with simple dependencies *within this initial batch*.

**Output Format (Strict JSON):** Provide *only* a valid JSON list of task objects (or `[]` if no tasks should be generated, though this is unlikely for the initial run). Assign a moderate priority (e.g., 3-6) unless a task seems foundational.

**Example (Illustrating Creativity and Diversity):**
```json
[
  {{
    "description": "Search the web for the 'Drake Equation' and write its formula and a brief explanation to memory.",
    "priority": 5
  }},
  {{
    "description": "Write a short poem about the color blue and save it to a file named 'blue_poem.txt'.",
    "priority": 4
  }},
  {{
    "description": "Browse the main page of Wikipedia (en.wikipedia.org) and summarize the top 3 'In the news' items into a memory entry.",
    "priority": 6
  }},
  {{
    "description": "Create a file named 'project_ideas.md' and write down three potential project ideas this agent could work on.",
    "priority": 5
  }},
  {{
    "description": "List the files currently in the root of the artifact workspace.",
    "priority": 3
  }},
  {{
    "description": "Generate and record an agent status report.",
    "priority": 2
  }}
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
