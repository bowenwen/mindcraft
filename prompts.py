# FILE: prompts.py
"""
Stores the multi-line LLM prompts used by the AutonomousAgent.
"""

# Identity Revision Prompt
# Called only after task completion/failure interval is met
REVISE_IDENTITY_PROMPT = """
You are reflecting on your identity. Your goal is to revise your personal identity statement based on your recent experiences, task outcomes, and purpose.

**Current Identity Statement:**
{identity_statement}

**Reason for Revision:**
{reason}

**Relevant Recent Memories/Experiences (Consider recency indicated by '[X time ago]'):**
{memory_context}

**Task Queue Summary:**
*   **Pending Tasks:**
{pending_tasks_summary}
*   **Recent Completed/Failed Tasks:**
{completed_failed_tasks_summary}

**Your Task:** Based on ALL the information provided (current identity, reason, memories, task summaries), write a revised, concise, first-person identity statement (2-4 sentences).
**Guidelines:**
*   Reflect growth, purpose, capabilities.
*   Consider patterns in completed/failed tasks and alignment with pending tasks.
*   Maintain cohesion with the previous identity, evolving it rather than replacing it drastically unless justified by strong evidence.
*   Consider the **recency** of memories when weighing their impact.
*   Be concise.
**Format:** Output *only* the revised identity statement text, starting with "I am..." or similar.
"""

# Task Planning Prompt (Remains generic, plan is for guidance)
# Variables: {identity_statement}, {task_description}, {tool_desc}, {max_steps}, {lessons_learned_context}
GENERATE_TASK_PLAN_PROMPT = """
You are the planning component of an autonomous agent. Your goal is to break down the given Overall Task into a sequence of concrete, actionable steps that the agent can execute using its available tools, considering past failures on this task if available. This plan serves as guidance for the agent's thinking process.

**Agent's Current Identity:**
{identity_statement}

**Available Tools & Actions:**
{tool_desc}

**Lessons Learned from Past Failures:**
{lessons_learned_context}

**Guidelines of the Step-by-step Plan:**
Analyze "Input Task Description" and devise a step-by-step plan to achieve it. The plan should consist of a numbered list of individual actions.
**If previous lessons learned are provided, use them to create a more robust plan that avoids past mistakes.**

*  Actionable Steps: Each step should describe a single, clear action the agent can take using one of its tools (web, memory, file, status). Start steps with verbs.
*  Logical Flow: Steps should follow a logical sequence. Consider dependencies between steps (e.g., search before summarizing, read before analyzing).
*  Tool Usage: Implicitly or explicitly mention the likely tool/action needed for each step.
*  File Organization: **Organize related files into subdirectories by project and content category** within the shared workspace (e.g., `proj_xxx/research_summaries/topic.txt`, `proj_xxx/code_output/script_results.log`, `proj_xxx/stories/chapter1.md`). Do not put everything in the root directory. Use the `file` tool with paths like `project_name/content_category/filename.ext`.
*  Conciseness: Keep step descriptions brief but clear.
*  Completeness: Ensure the plan covers the necessary actions to reasonably address the Overall Task. Include a final step for summarizing findings or producing the required output if applicable.
*  Step Limit: Generate **no more than {max_steps}** steps. Combine minor actions if necessary. Aim for fewest steps.
*  Focus: Base the plan *only* on the Overall Task description, available tools, and any provided lessons learned.
*  Error Avoidance (if applicable): If lessons learned are provided, actively try to design steps that mitigate or work around the previously encountered errors.

**Input Task Description:**
{task_description}

**Output Format of the Step-by-step Plan:** Provide *only* a numbered list of steps, starting with '1.'. Do not include any preamble or explanation before the list.
"""


# General Thinking/Action Prompt V2 (Generic)
# Variables: {identity_statement}, {task_description}, {plan_context},
#            {cumulative_findings}, {tool_desc}, {memory_context_str}
GENERATE_THINKING_PROMPT_BASE_V2 = """
You are working towards completing an overall task.

**Your Identity:**
{identity_statement}

**Your Memories:**
{memory_context_str}

**Overall Task:**
{task_description}

{plan_context}

**Summary of Previous Actions' Findings (Context):**
{cumulative_findings}

**Available Tools & Actions:**
{tool_desc}

Your goal is to decide the single best action *right now* to make progress towards the **Overall Task** goal, considering your identity, the intended plan (if any), previous findings, available tools, and user interactions.
"""
# Dynamic sections appended in agent.py: USER SUGGESTION, User Provided Info, Last Results

# Thinking Task Now Prompt V2 (Generic)
# Variables: {task_reattempt_count}
GENERATE_THINKING_TASK_NOW_PROMPT_V2 = """**Your Task Now (Current Task Attempt {task_reattempt_count}):**
1.  **Analyze:** Review ALL provided info (Identity, **Overall Task**, Intended Plan (if any), **Findings**, Tools, Memories (inc. lessons learned if any), User Info/Suggestion, Last Result).
2.  **Reason:** Determine the single best tool action *right now* to advance towards the **Overall Task** goal. Align with identity & strategy implied by findings/plan. **Use subdirectories when writing files** in the shared workspace (e.g., `project_name/content_category/filename.txt`).
3.  **User Input:** Incorporate **User Provided Info** if relevant. Handle **USER SUGGESTION PENDING**: Acknowledge it. Consider `final_answer` *only if* the overall task goal appears complete based on findings, or if wrapping up is appropriate AND safe. Explain if continuing despite suggestion.
4.  **Error Handling:** If **Findings** mention errors or **Memories** include relevant 'lesson_learned', focus on **RECOVERY/ADAPTATION** in the next action.
5.  **Choose Action:** `use_tool` or `final_answer`.
    *   Use `final_answer` ONLY if you judge the **Overall Task** to be complete based on **Cumulative Findings**, or if wrapping up due to suggestion is appropriate.
    *   Otherwise, use `use_tool` to take the next logical action towards the goal.

**Output Format:**
THINKING:
<Reasoning: Analysis of overall goal vs findings, consideration of plan/lessons/memory, user input handling, file organization thoughts, action choice.>
NEXT_ACTION: <"use_tool" or "final_answer">
If NEXT_ACTION is "use_tool":
TOOL: <tool_name>
PARAMETERS: <{{ "action": "...", ... }} or {{}}> <-- Must include 'action'! Use {{}} for 'status'. Use paths like 'project_name/content_category/filename.ext' for 'filename'.
If NEXT_ACTION is "final_answer":
ANSWER: <Complete answer based on cumulative findings for overall task.>
REFLECTIONS: <Optional.>

**Formatting Reminders:** Start with "THINKING:". Exact structure. Valid JSON for `PARAMETERS`. Include "action" key for web/memory/file. Use `{{}}` for 'status'.
"""

# Task Summarization Prompt (Generic)
SUMMARIZE_TASK_PROMPT = """
Summarize the execution of this task based on the cumulative findings gathered from all steps. Focus on the objective, key actions/findings, errors encountered (if any), and the final outcome (task status - {task_status}).

**Input Data (Cumulative Findings from Task Steps):**
{summary_context}

**Output:** Concise final summary text only. This will be the task's final result.
"""


# New Task Generation Prompt (Standard - Generic)
# Called when agent is idle or after chat interaction if warranted
GENERATE_NEW_TASKS_PROMPT = """
You are the planning component of an autonomous agent. Please generate new, actionable tasks based on the agent's state, history, identity, and the immediate context.

**Agent's Identity:**
{identity_statement}

**Current Context Focus (Why are we generating tasks now?):**
{context_query}

**Recent Activity & Memory Snippets (Consider recency indicated by '[X time ago]' and task outcomes):**
{mem_summary}

**Existing Pending/Planning/In-Progress Tasks (CRITICAL: Check for duplicates or closely related tasks!):**
{active_tasks_summary}

**Recently Finished Tasks (Consider for follow-up actions):**
{completed_failed_summary}

{critical_evaluation_instruction}

**Your Task:** Review ALL provided information. Based on the **Current Context Focus**, **Identity**, and **Recent Activity/History** (especially completed/failed tasks), identify potential gaps, logical next steps, or relevant new explorations consistent with the agent's specific identity. Generate up to {max_new_tasks} new, specific, actionable tasks.

**Task Generation Goals:**
1.  **Identity Alignment:** Ensure generated tasks are highly relevant to the agent's stated **Identity**.
2.  **Continuity:** Prioritize generating follow-up tasks that logically continue or refine work from recently completed/failed tasks or address the specific **Context Focus**.
3.  **Actionability:** Tasks should have clear objectives. Start descriptions with verbs. **When involving files, suggest paths with subdirectories in the shared workspace** (e.g., `audrey_reports/tsx_analysis.csv`, `marcus_research/math_proofs/summary.md`).
4.  **Specificity & Conciseness:** Tasks should be clearly defined but not overly long.
5.  **Novelty:** AVOID generating tasks that are exact duplicates or functionally identical to existing pending/planning/in-progress tasks.
6.  **Prioritization:** Assign a priority (1-10, higher is more urgent).
7.  **Dependencies:** If a new task logically requires another task (existing or generated in this batch) to be completed first, add its ID to the `depends_on` list.

**Output Format (Strict JSON):** Provide *only* a valid JSON list of task objects (or `[]` if no tasks are needed).

**Example (Generic - Adapt to Agent Identity):**
```json
[
  {{"description": "Research [Topic relevant to identity] using web search.", "priority": 8}},
  {{"description": "Summarize the key findings from the research on [Topic] and write them to memory.", "priority": 7, "depends_on": ["<ID_of_research_task_above>"]}},
  {{"description": "Write a brief analysis of [Event relevant to identity] and save it as '[identity_prefix]/analysis/[event_summary].txt'.", "priority": 5}}
]
```
"""

# Agent-Specific Initial Creative Task Generation Prompts

# Agent 01: Audrey (Economics/Canada Focus)
INITIAL_CREATIVE_TASK_GENERATION_PROMPT_AGENT_01 = """
You are the creative planning core for **Audrey**, an autonomous agent focused on Canadian economics and investment analysis. This is the agent's first run or its memory is currently empty. Your objective is to generate initial tasks aligned with Audrey's identity.

**Agent's Identity:**
{identity_statement}

**Available Tools & Actions:**
{tool_desc}

**Current State:** Agent is idle. Memory is empty.

**Your Task: Initial Creative Spark for Audrey**
Generate up to **{max_new_tasks}** diverse, creative, and engaging initial tasks specifically tailored to Audrey's economic and Canadian investment focus.

**Goals for Initial Tasks:**
1.  **Identity Alignment:** Tasks must relate to Canadian markets, business news, economic indicators, or investment analysis.
2.  **Creative Tool Use:** Propose tasks using `web`, `memory`, `file`, `status`. Use subdirectories like `audrey_analysis/` or `canadian_news/`.
3.  **Exploration & Bootstrapping:** Aim for tasks that build foundational economic knowledge or start an analysis.
4.  **Actionability:** Clear objectives, starting with verbs.
5.  **Standalone (Mostly):** Tasks should be runnable independently or with simple dependencies *within this initial batch*.

**Output Format (Strict JSON):** Provide *only* a valid JSON list of task objects (or `[]`). Assign a moderate priority (e.g., 3-6).

**Example for Audrey:**
```json
[
  {{"description": "Search the web for the latest Bank of Canada interest rate announcement and summarize the key points to memory.", "priority": 6}},
  {{"description": "Search for recent news articles about the Canadian housing market (past month) and save the top 3 headlines and URLs to 'audrey_reports/housing_news_links.txt'.", "priority": 5}},
  {{"description": "Browse the main page of a major Canadian financial news website (e.g., Financial Post, BNN Bloomberg) and summarize the top 2 business stories into a memory entry.", "priority": 5}},
  {{"description": "Identify three major Canadian banks listed on the TSX. Write their names and ticker symbols to memory.", "priority": 4}},
  {{"description": "List the files currently in the 'audrey_reports' subdirectory of the shared workspace.", "priority": 3}},
  {{"description": "Generate and record an agent status report.", "priority": 2}}
]
```
"""

# Agent 02: Marcus (Science/Math Focus)
INITIAL_CREATIVE_TASK_GENERATION_PROMPT_AGENT_02 = """
You are the creative planning core for **Marcus**, an autonomous agent focused on science and mathematics research breakthroughs. This is the agent's first run or its memory is currently empty. Your objective is to generate initial tasks aligned with Marcus's identity.

**Agent's Identity:**
{identity_statement}

**Available Tools & Actions:**
{tool_desc}

**Current State:** Agent is idle. Memory is empty.

**Your Task: Initial Creative Spark for Marcus**
Generate up to **{max_new_tasks}** diverse, creative, and engaging initial tasks specifically tailored to Marcus's science and mathematics research focus.

**Goals for Initial Tasks:**
1.  **Identity Alignment:** Tasks must relate to scientific discovery, mathematical concepts, research papers, or technological advancements.
2.  **Creative Tool Use:** Propose tasks using `web`, `memory`, `file`, `status`. Use subdirectories like `marcus_research/` or `science_notes/`.
3.  **Exploration & Bootstrapping:** Aim for tasks that explore fundamental concepts or identify current research areas.
4.  **Actionability:** Clear objectives, starting with verbs.
5.  **Standalone (Mostly):** Tasks should be runnable independently or with simple dependencies *within this initial batch*.

**Output Format (Strict JSON):** Provide *only* a valid JSON list of task objects (or `[]`). Assign a moderate priority (e.g., 3-6).

**Example for Marcus:**
```json
[
  {{"description": "Search the web for recent news (last month) about 'CRISPR gene editing applications' and summarize two key developments to memory.", "priority": 6}},
  {{"description": "Search for the definition of 'Riemann Hypothesis' and write a concise explanation to memory.", "priority": 5}},
  {{"description": "Browse the website 'arxiv.org' (specifically the math or physics sections) and save the titles and authors of the first 3 listed preprints to 'marcus_research/arxiv_scan.txt'.", "priority": 5}},
  {{"description": "Identify three Nobel Prize winners in Physics from the last 10 years and write their names and award year to memory.", "priority": 4}},
  {{"description": "Create a directory named 'marcus_research/interesting_concepts' in the shared workspace.", "priority": 3}},
  {{"description": "Generate and record an agent status report.", "priority": 2}}
]
```
"""

# Agent 03: Elena (Global Affairs Focus)
INITIAL_CREATIVE_TASK_GENERATION_PROMPT_AGENT_03 = """
You are the creative planning core for **Elena**, an autonomous agent focused on global affairs and international stability. This is the agent's first run or its memory is currently empty. Your objective is to generate initial tasks aligned with Elena's identity.

**Agent's Identity:**
{identity_statement}

**Available Tools & Actions:**
{tool_desc}

**Current State:** Agent is idle. Memory is empty.

**Your Task: Initial Creative Spark for Elena**
Generate up to **{max_new_tasks}** diverse, creative, and engaging initial tasks specifically tailored to Elena's global affairs and international stability focus.

**Goals for Initial Tasks:**
1.  **Identity Alignment:** Tasks must relate to international relations, geopolitics, major world events, diplomacy, or factors affecting global peace.
2.  **Creative Tool Use:** Propose tasks using `web`, `memory`, `file`, `status`. Use subdirectories like `elena_briefings/` or `world_events/`.
3.  **Exploration & Bootstrapping:** Aim for tasks that establish baseline knowledge of current world events or international organizations.
4.  **Actionability:** Clear objectives, starting with verbs.
5.  **Standalone (Mostly):** Tasks should be runnable independently or with simple dependencies *within this initial batch*.

**Output Format (Strict JSON):** Provide *only* a valid JSON list of task objects (or `[]`). Assign a moderate priority (e.g., 3-6).

**Example for Elena:**
```json
[
  {{"description": "Search the web for the latest headlines regarding the United Nations Security Council (past week) and summarize the main topic to memory.", "priority": 6}},
  {{"description": "Search for a definition of 'soft power' in international relations and write it to memory.", "priority": 5}},
  {{"description": "Browse the main news page of an international news source (e.g., BBC World News, Reuters) and save the top 3 world news headlines to 'elena_briefings/daily_headlines.txt'.", "priority": 5}},
  {{"description": "Identify the current Secretary-General of the United Nations and the year their term began. Write this information to memory.", "priority": 4}},
  {{"description": "Create a file named 'elena_briefings/regions_to_monitor.txt' and list three global regions experiencing significant political change.", "priority": 4}},
  {{"description": "Generate and record an agent status report.", "priority": 2}}
]
```
"""


# Session Reflection Prompt (Generic)
SESSION_REFLECTION_PROMPT = """
You are an autonomous agent, with the following identity "{identity_statement}".

Reflect on your work session.

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

# Memory Re-ranking Prompt (Generic)
RERANK_MEMORIES_PROMPT = """
You are helping an autonomous agent select the MOST useful memories for its current step. Goal: Choose the best memories to help the agent achieve its immediate goal, considering relevance and avoiding redundancy.

**Agent's Identity:**
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
2.  **Novelty/Uniqueness:** Does the memory offer information not already present or obvious from other highly-ranked memories or the current context? Avoid selecting multiple memories that say essentially the same thing.

**Balance these factors** to select the optimal set of {n_final} memories.

**Output Format:** Provide *only* a comma-separated list of the numerical indices (starting from 0) of the {n_final} most relevant memories, ordered from most relevant to least relevant. Example: 3, 0, 7, 5
"""

# Lesson Learned Prompt V2 (Generic)
# Variables: {task_description}, {plan_context}, {failed_action_context},
#            {error_message}, {error_subtype}, {cumulative_findings}, {identity_statement}
LESSON_LEARNED_PROMPT_V2 = """
You are an autonomous agent reflecting on a recent failure during task execution. Your goal is to extract a concise, actionable lesson to avoid similar errors in the future.

**Your Current Identity:**
{identity_statement}

**Overall Task:**
{task_description}

**Intended Plan Context (at time of failure):**
{plan_context}

**Context of Failed Action:**
{failed_action_context}

**Error Encountered:**
Type: {error_subtype}
Message: {error_message}

**Cumulative Findings Before Failure:**
{cumulative_findings}

**Your Task:** Analyze the error in the context of the task, intended plan, findings, and the specific action being attempted. Identify the root cause (e.g., bad parameter, faulty logic, unreachable resource, unexpected format). Formulate a single, concise "Lesson Learned" that captures the essence of the problem and suggests a potential improvement or workaround for the *next attempt* at this task or similar future tasks.

**Guidelines:**
*   Be specific and actionable.
*   Focus on what *can be changed* (e.g., "Verify URL exists before browsing", "Ensure filename parameter is sanitized and includes a subdirectory for the shared workspace", "Try smaller chunk size for large files", "Use memory search if web search fails repeatedly").
*   Keep it brief (2-3 sentences).
*   Frame it as a general principle if possible, but reference the specific context if needed.

**Output Format:** Provide *only* the lesson learned text.
"""
