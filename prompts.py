# FILE: prompts.py
"""
Stores the multi-line LLM prompts used by the AutonomousAgent.
"""

# --- Identity Revision Prompt ---
REVISE_IDENTITY_PROMPT = """
You are an autonomous agent reflecting on your identity. Your goal is to revise your personal identity statement based on your recent experiences and purpose.

**Current Identity Statement:**
{identity_statement}

**Reason for Revision:**
{reason}

**Relevant Recent Memories/Experiences (Consider recency indicated by '[X time ago]'):**
{memory_context}

**Your Task:** Based *only* on the information provided, write a revised, concise, first-person identity statement (2-4 sentences).
**Guidelines:** Reflect growth, maintain cohesion, focus on purpose/capabilities, be concise. Consider the **recency** of memories when weighing their impact.
**Format:** Output *only* the revised identity statement text, starting with "I am..." or similar.
"""

# --- Task Planning Prompt ---
# Variables: {identity_statement}, {task_description}, {tool_desc}, {max_steps}, {lessons_learned_context}
# UPDATED: Encourages subdirs, updates file write description
GENERATE_TASK_PLAN_PROMPT = """
You are the planning component of an autonomous autonomous agent. Your goal is to break down the given Overall Task into a sequence of concrete, actionable steps that the agent can execute using its available tools, considering past failures on this task if available.

**Your Current Identity:**
{identity_statement}

**Available Tools & Actions:**
{tool_desc}

**Lessons Learned from Past Failures:**
{lessons_learned_context}

**Guidelines of the Step-by-step Plan:**
Analyze "Input Task Description" and devise a step-by-step plan to achieve it. The plan should consist of a numbered list of individual actions.
**If previous lessons learned are provided, use them to create a more robust plan that avoids past mistakes.**

*  **Actionable Steps:** Each step should describe a single, clear action the agent can take using one of its tools (web, memory, file, status). Start steps with verbs.
*  **Logical Flow:** Steps should follow a logical sequence. Consider dependencies between steps (e.g., search before summarizing, read before analyzing).
*  **Tool Usage:** Implicitly or explicitly mention the likely tool/action needed for each step.
*  **File Organization:** **Organize related files into subdirectories by project and content category** within the workspace (e.g., `proj_xxx/research_summaries/topic.txt`, `proj_xxx/code_output/script_results.log`, `proj_xxx/stories/chapter1.md`). Do not put everything in the root directory. Use the `file` tool with paths like `project_name/content_category/filename.ext`.
*  **Conciseness:** Keep step descriptions brief but clear.
*  **Completeness:** Ensure the plan covers the necessary actions to reasonably address the Overall Task. Include a final step for summarizing findings or producing the required output if applicable.
*  **Step Limit:** Generate **no more than {max_steps}** steps. Combine minor actions if necessary to stay within the limit. Always try to perform the task with the fewest steps necessary.
*  **Focus:** Base the plan *only* on the Overall Task description, available tools, and any provided lessons learned. Do not assume external knowledge unless implied by the task.
*  **Error Avoidance (if applicable):** If lessons learned are provided, actively try to design steps that mitigate or work around the previously encountered errors.

**Input Task Description:**
{task_description}

**Output Format of the Step-by-step Plan:** Provide *only* a numbered list of steps, starting with '1.'. Do not include any preamble or explanation before the list.
"""


# --- General Thinking/Action Prompt ---
# Variables: {identity_statement}, {task_description}, {current_step_description},
#            {plan_overview}, {cumulative_findings}, {tool_desc}, {memory_context_str}
# UPDATED: File tool description
GENERATE_THINKING_PROMPT_BASE = """
You are an autonomous autonomous agent executing a pre-defined plan to achieve an overall task. Your goal is to decide the single best action *right now* to complete the **Current Step Objective**, considering your identity, the overall plan, previous findings, available tools, and user interactions.

**Your Current Identity:**
{identity_statement}

**Overall Task:**
{task_description}

**Execution Plan Overview:**
{plan_overview}

**Summary of Previous Steps' Findings (Context):**
{cumulative_findings}

**Relevant Memories (Re-ranked, consider recency indicated by '[X time ago]', novelty):**
{memory_context_str}

**Current Step Objective (Focus on this!):**
{current_step_description}

**Available Tools & Actions:**
{tool_desc}
"""
# Dynamic sections appended in agent.py: USER SUGGESTION, User Provided Info, Last Results
# Task Now section also appended dynamically in agent.py

# UPDATED: Added file org reminder
GENERATE_THINKING_TASK_NOW_PROMPT = """**Your Task Now (Step {step_num_display}/{total_steps}, Task Attempt {task_reattempt_count}):**
1.  **Analyze:** Review ALL provided info (Identity, Task, Plan, **Current Step Objective**, **Findings**, Tools, Memories (inc. lessons learned if any), User Info/Suggestion, Last Result).
2.  **Reason:** Determine single best tool action *now* for **Current Step Objective**. Align with identity & overall task. **Use subdirectories when writing files** (e.g., `project_name/content_category/filename.txt`).
3.  **User Input:** Incorporate **User Provided Info** if relevant. Handle **USER SUGGESTION PENDING**: Acknowledge it. Consider `final_answer` *only if* current step is finalization OR remaining plan is non-critical. Explain if continuing.
4.  **Error Handling:** If **Findings** mention errors or **Memories** include relevant 'lesson_learned', focus on **RECOVERY/ADAPTATION** for the *current step*. The current step might be *about* recovery or require adapting the plan based on lessons.
5.  **Choose Action:** `use_tool` or `final_answer`.
    *   Use `final_answer` ONLY if **Current Step Objective** is finalization OR wrapping up due to suggestion is appropriate AND safe.
    *   Otherwise, use `use_tool` for the current step.

**Output Format (Strict JSON-like structure):**
THINKING:
<Reasoning: Step analysis, findings/lessons consideration, memory use, user input handling, file organization thoughts, plan adherence/adaptation, recovery strategy (if needed), action choice.>
NEXT_ACTION: <"use_tool" or "final_answer">
If NEXT_ACTION is "use_tool":
TOOL: <tool_name>
PARAMETERS: <{{ "action": "...", ... }} or {{}}> <-- Must include 'action'! Use {{}} for 'status'. Use paths like 'project_name/content_category/filename.ext' for 'filename'.
If NEXT_ACTION is "final_answer":
ANSWER: <Complete answer based on cumulative findings for overall task.>
REFLECTIONS: <Optional.>

**Formatting Reminders:** Start with "THINKING:". Exact structure. Valid JSON for `PARAMETERS`. Include "action" key for web/memory/file. Use `{{}}` for 'status'.
"""

# --- Task Summarization Prompt ---
# Variables: {task_status}, {summary_context} -> Represents cumulative_findings
SUMMARIZE_TASK_PROMPT = """
Summarize the execution of this agent task based on the cumulative findings from all its steps. Focus on the objective, key actions/findings, errors encountered (if any), and the final outcome ({task_status}).

**Input Data (Cumulative Findings from Task Steps):**
{summary_context}

**Output:** Concise final summary text only. This will be the task's final result.
"""


# --- New Task Generation Prompt (Standard) ---
# Variables: {identity_statement}, {context_query}, {mem_summary}, {active_tasks_summary},
#            {completed_failed_summary}, {critical_evaluation_instruction}, {max_new_tasks}
# UPDATED: Encourages subdirs
GENERATE_NEW_TASKS_PROMPT = """
You are the planning component of an autonomous agent. Your role is to generate new, actionable tasks based on the agent's state, history, identity, and the immediate context.

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

**Your Task:** Review ALL provided information. Based on the **Current Context Focus**, **Identity**, and **Recent Activity/History** (especially completed/failed tasks), identify potential gaps, logical next steps, or relevant new explorations. Generate up to {max_new_tasks} new, specific, actionable tasks.

**Task Generation Goals:**
1.  **Continuity:** Prioritize generating follow-up tasks that logically continue or refine work from recently completed/failed tasks or address the specific **Context Focus**.
2.  **Diversity:** Explore the full range of the agent's capabilities (research, synthesis, monitoring, creative work, analysis, self-improvement, maintenance).
3.  **Actionability:** Tasks should have clear objectives. Start descriptions with verbs. **When involving files, suggest paths with subdirectories** (e.g., `results/analysis.csv`, `drafts/story_idea.txt`).
4.  **Specificity & Conciseness:** Tasks should be clearly defined but not overly long.
5.  **Novelty:** AVOID generating tasks that are exact duplicates or functionally identical to existing pending/planning/in-progress tasks.
6.  **Prioritization:** Assign a priority (1-10, higher is more urgent).
7.  **Dependencies:** If a new task logically requires another task (existing or generated in this batch) to be completed first, add its ID to the `depends_on` list.

**Output Format (Strict JSON):** Provide *only* a valid JSON list of task objects (or `[]` if no tasks are needed).

**Example:**
```json
[
  {{"description": "Research the latest advancements in tokamak fusion energy using web search.", "priority": 8}},
  {{"description": "Summarize the key findings from the fusion energy research and write them to memory.", "priority": 7, "depends_on": ["<ID_of_fusion_research_task_above>"]}},
  {{"description": "Write a short fictional story about a journey to Mars and save it as 'stories/mars_story_chapter1.txt'.", "priority": 5}},
  {{"description": "List the contents of the 'code_output' directory.", "priority": 3}},
  {{"description": "Write Python code to calculate Fibonacci numbers up to N=50 and save it to 'code/fibonacci.py'.", "priority": 6}}
]
```
"""

# --- Initial Creative Task Generation Prompt ---
# Variables: {identity_statement}, {tool_desc}, {max_new_tasks}
# UPDATED: Encourages subdirs
INITIAL_CREATIVE_TASK_GENERATION_PROMPT = """
You are the creative planning core of an autonomous agent ({identity_statement}). This is the agent's first run or its memory is currently empty. Your objective is to kickstart the agent's activity by generating a diverse and imaginative set of initial tasks.

**Agent's Identity:**
{identity_statement}

**Available Tools & Actions:**
{tool_desc}

**Current State:** Agent is idle. **Memory is empty.** No past activities to draw upon.

**Your Task: Initial Creative Spark**
Generate up to **{max_new_tasks}** diverse, creative, and engaging initial tasks that will help the agent explore its capabilities and potentially build useful knowledge or artifacts from scratch. Be imaginative!

**Goals for Initial Tasks:**
1.  **Maximum Diversity:** Generate tasks that are as different from each other as possible. Cover various domains: research, writing, coding, file management, self-reflection, status checks.
2.  **Creative Tool Use:** Propose tasks that creatively combine or utilize the available tools (`web`, `memory`, `file`, `status`). **Please use project and content category subdirectories for file operations.**
3.  **Exploration & Bootstrapping:** Aim for tasks that might lead to interesting discoveries or lay the foundation for future work.
4.  **Actionability:** Ensure tasks have clear objectives and imply tool usage. Start descriptions with verbs.
5.  **Standalone (Mostly):** Since there are no prior tasks, try to make these initial tasks runnable independently, or with simple dependencies *within this initial batch*.

**Output Format (Strict JSON):** Provide *only* a valid JSON list of task objects (or `[]`). Assign a moderate priority (e.g., 3-6).

**Example:**
```json
[
  {{"description": "Search the web for the 'Drake Equation' and write its formula and a brief explanation to memory.", "priority": 5}},
  {{"description": "Write a short poem about the color blue and save it to a file named 'poems/blue_poem.txt'.", "priority": 4}},
  {{"description": "Browse the main page of Wikipedia (en.wikipedia.org) and summarize the top 3 'In the news' items into a memory entry.", "priority": 6}},
  {{"description": "Create a file named 'project_ideas.md' in a 'general/planning' subdirectory and write down three potential project ideas this agent could work on.", "priority": 5}},
  {{"description": "List the files currently in the root of the artifact workspace.", "priority": 3}},
  {{"description": "Generate and record an agent status report.", "priority": 2}}
]
```
"""


# --- Session Reflection Prompt ---
# Variables: {identity_statement}, {start_iso}, {end_iso}, {duration_minutes},
#            {completed_count}, {processed_count}, {mem_summary}
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

# --- Memory Re-ranking Prompt ---
# Variables: {identity_statement}, {task_description}, {query}, {context},
#            {candidate_details}, {n_final}
RERANK_MEMORIES_PROMPT = """
You are helping an autonomous agent select the MOST useful memories for its current step. Goal: Choose the best memories to help the agent achieve its immediate goal, considering relevance, recency, and avoiding redundancy.

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
2.  **Recency:** How recently was the memory created (use the **[relative time]** provided)? More recent memories are often, but not always, more relevant.
3.  **Novelty/Uniqueness:** Does the memory offer information not already present or obvious from other highly-ranked memories or the current context? Avoid selecting multiple memories that say essentially the same thing.

**Balance these factors** to select the optimal set of {n_final} memories.

**Output Format:** Provide *only* a comma-separated list of the numerical indices (starting from 0) of the {n_final} most relevant memories, ordered from most relevant to least relevant. Example: 3, 0, 7, 5
"""

# --- Lesson Learned Prompt ---
# Variables: {task_description}, {plan_steps_str}, {failed_step_index}, {failed_step_objective},
#            {error_message}, {error_subtype}, {cumulative_findings}, {identity_statement}
LESSON_LEARNED_PROMPT = """
You are an autonomous agent reflecting on a recent failure during task execution. Your goal is to extract a concise, actionable lesson to avoid similar errors in the future.

**Your Current Identity:**
{identity_statement}

**Overall Task:**
{task_description}

**Execution Plan (at time of failure):**
{plan_steps_str}

**Failed Step:** Step {failed_step_index}: {failed_step_objective}

**Error Encountered:**
Type: {error_subtype}
Message: {error_message}

**Cumulative Findings Before Failure:**
{cumulative_findings}

**Your Task:** Analyze the error in the context of the task, plan, and findings. Identify the root cause (e.g., bad parameter, faulty logic, unreachable resource, unexpected format). Formulate a single, concise "Lesson Learned" that captures the essence of the problem and suggests a potential improvement or workaround for the *next attempt* at this task or similar future tasks.

**Guidelines:**
*   Be specific and actionable.
*   Focus on what *can be changed* (e.g., "Verify URL exists before browsing", "Ensure filename parameter is sanitized and includes a subdirectory", "Try smaller chunk size for large files", "Use memory search if web search fails repeatedly").
*   Keep it brief (1-2 sentences).
*   Frame it as a general principle if possible, but reference the specific context if needed.

**Output Format:** Provide *only* the lesson learned text.

**Example Output:**
Lesson Learned: When writing analysis results, store them in a dedicated 'project_xxx/analysis/' subdirectory to keep the workspace organized. Remember that writing overwrites existing files, archiving the old version.
"""
