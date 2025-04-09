# FILE: task_manager.py
# autonomous_agent/task_manager.py
import os
import json
import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

from data_structures import Task
from config import TASK_MAX_REATTEMPT, get_agent_task_queue_path  # Use path function

import logging

# Keep logging simple for this file
logging.basicConfig(level=logging.INFO, format="[%(levelname)s][TASK] %(message)s")
log = logging.getLogger(__name__)


class TaskQueue:
    """Manages the persistence and retrieval of tasks for a specific agent."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.queue_path = get_agent_task_queue_path(agent_id)
        self.tasks: Dict[str, Task] = {}
        log.info(
            f"Initializing TaskQueue for Agent '{agent_id}' using path: {self.queue_path}"
        )
        self.load_queue()

    def load_queue(self):
        if os.path.exists(self.queue_path):
            try:
                with open(self.queue_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if not content or not content.strip():
                    log.info(
                        f"Task queue file '{self.queue_path}' empty for Agent '{self.agent_id}'."
                    )
                    self.tasks = {}
                    return
                data = json.loads(content)
                if not isinstance(data, list):
                    log.error(
                        f"Task queue '{self.queue_path}' not a list for Agent '{self.agent_id}'. Starting fresh."
                    )
                    self.tasks = {}
                    return
                loaded = 0
                skipped_reattempt = 0
                for task_data in data:
                    if isinstance(task_data, dict) and "id" in task_data:
                        try:
                            status = task_data.get("status", "pending")
                            reattempt_count = task_data.get("reattempt_count", 0)
                            task_id_log = task_data["id"][:8]  # Short ID for logs

                            if (
                                status not in ["completed", "failed"]
                                and reattempt_count >= TASK_MAX_REATTEMPT
                            ):
                                log.warning(
                                    f"[Agent:{self.agent_id}] Task {task_id_log} loaded with reattempt count {reattempt_count} >= {TASK_MAX_REATTEMPT} and status '{status}'. Marking as failed."
                                )
                                task_data["status"] = "failed"
                                task_data["reflections"] = (
                                    task_data.get("reflections", "")
                                    + f"\n[Loaded State]: Marked failed due to exceeding max reattempts ({reattempt_count})."
                                )
                                task_data["completed_at"] = (
                                    task_data.get("completed_at")
                                    or datetime.datetime.now(
                                        datetime.timezone.utc
                                    ).isoformat()
                                )
                                skipped_reattempt += 1
                                task_data["plan"] = None
                                task_data["current_step_index"] = 0
                            elif status not in [
                                "pending",
                                "planning",
                                "in_progress",
                                "completed",
                                "failed",
                            ]:
                                log.warning(
                                    f"[Agent:{self.agent_id}] Task {task_id_log} has invalid status '{status}', resetting to pending."
                                )
                                task_data["status"] = "pending"
                                task_data["plan"] = None
                                task_data["current_step_index"] = 0
                                task_data["cumulative_findings"] = ""
                                task_data["reattempt_count"] = 0
                            elif (
                                status == "pending"
                                and task_data.get("plan") is not None
                            ):
                                log.warning(
                                    f"[Agent:{self.agent_id}] Task {task_id_log} loaded as pending but had a plan. Clearing plan."
                                )
                                task_data["plan"] = None
                                task_data["current_step_index"] = 0
                                task_data["cumulative_findings"] = ""
                            if "current_step_index" not in task_data:
                                task_data["current_step_index"] = 0

                            self.tasks[task_data["id"]] = Task.from_dict(task_data)
                            loaded += 1
                        except Exception as e:
                            log.warning(
                                f"[Agent:{self.agent_id}] Skipping task data parse error: {e}. Data: {task_data}"
                            )
                    else:
                        log.warning(
                            f"[Agent:{self.agent_id}] Skipping invalid task data item: {task_data}"
                        )
                log.info(
                    f"[Agent:{self.agent_id}] Loaded {loaded} tasks from {self.queue_path}. Marked {skipped_reattempt} as failed due to reattempt limit."
                )
            except json.JSONDecodeError as e:
                log.error(
                    f"[Agent:{self.agent_id}] Invalid JSON in '{self.queue_path}': {e}. Starting fresh."
                )
                self.tasks = {}
            except Exception as e:
                log.error(
                    f"[Agent:{self.agent_id}] Failed loading task queue '{self.queue_path}': {e}. Starting fresh."
                )
                self.tasks = {}
        else:
            log.info(
                f"[Agent:{self.agent_id}] Task queue file '{self.queue_path}' not found. Starting fresh."
            )
            self.tasks = {}

    def save_queue(self):
        try:
            with open(self.queue_path, "w", encoding="utf-8") as f:
                sorted_tasks = sorted(self.tasks.values(), key=lambda t: t.created_at)
                json.dump(
                    [task.to_dict() for task in sorted_tasks],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as e:
            log.error(
                f"[Agent:{self.agent_id}] Error saving task queue to '{self.queue_path}': {e}"
            )

    def add_task(self, task: Task) -> Optional[str]:
        if not isinstance(task, Task):
            log.error(
                f"[Agent:{self.agent_id}] Attempted to add non-Task object: {task}"
            )
            return None
        lower_desc = task.description.strip().lower()
        for existing_task in self.tasks.values():
            if (
                existing_task.description.strip().lower() == lower_desc
                and existing_task.status in ["pending", "planning", "in_progress"]
            ):
                log.warning(
                    f"[Agent:{self.agent_id}] Skipping task add - potential duplicate description for active task: '{task.description[:60]}...'"
                )
                return None
        self.tasks[task.id] = task
        self.save_queue()
        log.info(
            f"[Agent:{self.agent_id}] Task added: {task.id[:8]} - '{task.description[:60]}...' (Prio: {task.priority}, Depends: {task.depends_on})"
        )
        return task.id

    def get_next_task(self) -> Optional[Task]:
        available = []
        tasks_list = list(self.tasks.values())
        for task in tasks_list:
            if task.status == "pending":
                if task.reattempt_count >= TASK_MAX_REATTEMPT:
                    log.warning(
                        f"[Agent:{self.agent_id}] Task {task.id[:8]} skipped: Exceeded max reattempts ({task.reattempt_count}/{TASK_MAX_REATTEMPT})."
                    )
                    self.update_task(
                        task.id,
                        "failed",
                        reflections=f"Skipped: Exceeded max reattempts ({task.reattempt_count}).",
                    )
                    continue
                deps_met = True
                if task.depends_on:
                    for dep_id in task.depends_on:
                        dep_task = self.tasks.get(dep_id)
                        if not dep_task or dep_task.status != "completed":
                            deps_met = False
                            break
                if deps_met:
                    available.append(task)
        if not available:
            return None
        return sorted(available, key=lambda t: (-t.priority, t.created_at))[0]

    def update_task(
        self,
        task_id: str,
        status: str,
        result: Any = None,
        reflections: Optional[str] = None,
    ):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if (
                task.status == status
                and task.result == result
                and task.reflections == reflections
            ):
                return
            log.info(
                f"[Agent:{self.agent_id}] Updating task {task_id[:8]} status to: {status}"
            )
            task.status = status
            if status in ["completed", "failed"] and not task.completed_at:
                task.completed_at = datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat()
            if result is not None:
                task.result = result
            if reflections is not None:
                task.reflections = reflections
            if status == "pending":
                task.plan = None
                task.current_step_index = 0
                task.cumulative_findings = ""
            if status in ["completed", "failed"]:
                task.plan = None
                task.current_step_index = 0
            self.save_queue()
        else:
            log.error(
                f"[Agent:{self.agent_id}] Cannot update task - ID '{task_id}' not found."
            )

    def prepare_task_for_reattempt(self, task_id: str, lesson_learned: str) -> bool:
        if task_id in self.tasks:
            task = self.tasks[task_id]
            log.info(
                f"[Agent:{self.agent_id}] Preparing task {task_id[:8]} for re-attempt (Current count: {task.reattempt_count})."
            )
            task.reattempt_count += 1
            task.status = "planning"
            task.plan = None
            task.current_step_index = 0
            task.cumulative_findings += f"\nTask Re-attempting (Attempt {task.reattempt_count})\nLesson Learned: {lesson_learned}\nResetting Plan\n"
            task.completed_at = None
            task.reflections = (
                task.reflections or ""
            ) + f"\n[Reattempting task - Attempt {task.reattempt_count}]"
            self.save_queue()
            log.info(
                f"[Agent:{self.agent_id}] Task {task.id[:8]} state reset for re-attempt {task.reattempt_count}. New status: planning."
            )
            return True
        else:
            log.error(
                f"[Agent:{self.agent_id}] Cannot prepare task for re-attempt - ID '{task_id}' not found."
            )
            return False

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def set_priority(self, task_id: str, priority: int) -> bool:
        if task_id in self.tasks:
            task = self.tasks[task_id]
            new_priority = max(1, min(10, int(priority)))
            if task.priority == new_priority:
                return True
            log.info(
                f"[Agent:{self.agent_id}] Updating task {task_id[:8]} priority from {task.priority} to {new_priority}."
            )
            task.priority = new_priority
            self.save_queue()
            return True
        else:
            log.error(
                f"[Agent:{self.agent_id}] Cannot set priority - Task ID '{task_id}' not found."
            )
            return False

    def get_dependent_tasks(self, task_id: str) -> List[Task]:
        if not task_id:
            return []
        dependent = [task for task in self.tasks.values() if task_id in task.depends_on]
        log.debug(
            f"[Agent:{self.agent_id}] Found {len(dependent)} tasks dependent on {task_id[:8]}."
        )
        return sorted(dependent, key=lambda t: (-t.priority, t.created_at))

    def get_tasks_by_status(self, statuses: List[str]) -> List[Task]:
        if not statuses:
            return []
        return [task for task in self.tasks.values() if task.status in statuses]

    # get_all_tasks_structured remains unchanged in its logic, but operates on the agent's specific task list
    def get_all_tasks_structured(self) -> Dict[str, List[Dict[str, Any]]]:
        categorized: Dict[str, List[Task]] = defaultdict(list)
        for task in self.tasks.values():
            categorized[task.status].append(task)
        output: Dict[str, List[Dict[str, Any]]] = {
            "pending": [],
            "planning": [],
            "in_progress": [],
            "completed": [],
            "failed": [],
        }
        pending_sorted = sorted(
            categorized["pending"], key=lambda t: (-t.priority, t.created_at)
        )
        output["pending"] = [
            {
                "ID": t.id,
                "Priority": t.priority,
                "Description": t.description,
                "Depends On": ", ".join(t.depends_on),
                "Created": t.created_at,
            }
            for t in pending_sorted
        ]
        planning_sorted = sorted(
            categorized["planning"], key=lambda t: (-t.priority, t.created_at)
        )
        output["planning"] = [
            {
                "ID": t.id,
                "Priority": t.priority,
                "Description": t.description,
                "Status": f"Planning (Attempt {t.reattempt_count + 1})",
                "Created": t.created_at,
            }
            for t in planning_sorted
        ]
        inprogress_sorted = sorted(
            categorized["in_progress"], key=lambda t: t.created_at
        )
        output["in_progress"] = [
            {
                "ID": t.id,
                "Priority": t.priority,
                "Description": t.description,
                "Status": f"Executing (Cycle {t.current_step_index}, Attempt {t.reattempt_count + 1})",
                "Created": t.created_at,
            }
            for t in inprogress_sorted
        ]
        completed_sorted = sorted(
            categorized["completed"],
            key=lambda t: t.completed_at or t.created_at,
            reverse=True,
        )
        output["completed"] = [
            {
                "ID": t.id,
                "Description": t.description,
                "Completed At": t.completed_at,
                "Result Snippet": str(t.result)[:100] + "..." if t.result else "N/A",
            }
            for t in completed_sorted
        ]
        failed_sorted = sorted(
            categorized["failed"],
            key=lambda t: t.completed_at or t.created_at,
            reverse=True,
        )
        output["failed"] = [
            {
                "ID": t.id,
                "Description": t.description,
                "Failed At": t.completed_at,
                "Reason": str(t.reflections)[:100] + "..." if t.reflections else "N/A",
                "Reattempts": t.reattempt_count,
            }
            for t in failed_sorted
        ]
        return output
