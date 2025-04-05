# FILE: task_manager.py
# autonomous_agent/task_manager.py
import os
import json
import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

from data_structures import Task
from config import TASK_QUEUE_PATH, TASK_MAX_REATTEMPT  # Import new config

import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s][TASK] %(message)s")
log = logging.getLogger(__name__)


class TaskQueue:
    """Manages the persistence and retrieval of tasks."""

    def __init__(self, queue_path: str = TASK_QUEUE_PATH):
        self.queue_path = queue_path
        self.tasks: Dict[str, Task] = {}
        self.load_queue()

    def load_queue(self):
        if os.path.exists(self.queue_path):
            try:
                with open(self.queue_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if not content or not content.strip():
                    log.info(f"Task queue file '{self.queue_path}' empty.")
                    self.tasks = {}
                    return
                data = json.loads(content)
                if not isinstance(data, list):
                    log.error(
                        f"Task queue '{self.queue_path}' not a list. Starting fresh."
                    )
                    self.tasks = {}
                    return
                loaded = 0
                skipped_reattempt = 0
                for task_data in data:
                    if isinstance(task_data, dict) and "id" in task_data:
                        try:
                            # --- Validate status and reattempt count ---
                            status = task_data.get("status", "pending")
                            reattempt_count = task_data.get("reattempt_count", 0)

                            # If loaded task already exceeded max reattempts and isn't completed/failed, mark as failed
                            if (
                                status not in ["completed", "failed"]
                                and reattempt_count >= TASK_MAX_REATTEMPT
                            ):
                                log.warning(
                                    f"Task {task_data['id']} loaded with reattempt count {reattempt_count} >= {TASK_MAX_REATTEMPT} and status '{status}'. Marking as failed."
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
                                # Reset plan fields as it's failed now
                                task_data["plan"] = None
                                task_data["current_step_index"] = 0
                                # Keep cumulative findings for context

                            elif status not in [
                                "pending",
                                "planning",
                                "in_progress",
                                "completed",
                                "failed",
                            ]:
                                log.warning(
                                    f"Task {task_data['id']} has invalid status '{status}', resetting to pending."
                                )
                                task_data["status"] = "pending"
                                task_data["plan"] = None
                                task_data["current_step_index"] = 0
                                task_data["cumulative_findings"] = ""
                                task_data["reattempt_count"] = (
                                    0  # Reset reattempts if status reset
                                )

                            # Reset plan fields if status is pending but shouldn't have a plan
                            elif (
                                status == "pending"
                                and task_data.get("plan") is not None
                            ):
                                log.warning(
                                    f"Task {task_data['id']} loaded as pending but had a plan. Clearing plan."
                                )
                                task_data["plan"] = None
                                task_data["current_step_index"] = 0
                                task_data["cumulative_findings"] = ""

                            # --- End Validation ---

                            self.tasks[task_data["id"]] = Task.from_dict(task_data)
                            loaded += 1
                        except Exception as e:
                            log.warning(
                                f"Skipping task data parse error: {e}. Data: {task_data}"
                            )
                    else:
                        log.warning(f"Skipping invalid task data item: {task_data}")
                log.info(
                    f"Loaded {loaded} tasks from {self.queue_path}. Marked {skipped_reattempt} as failed due to reattempt limit."
                )
            except json.JSONDecodeError as e:
                log.error(f"Invalid JSON in '{self.queue_path}': {e}. Starting fresh.")
                self.tasks = {}
            except Exception as e:
                log.error(
                    f"Failed loading task queue '{self.queue_path}': {e}. Starting fresh."
                )
                self.tasks = {}
        else:
            log.info(f"Task queue file '{self.queue_path}' not found. Starting fresh.")
            self.tasks = {}

    def save_queue(self):
        try:
            with open(self.queue_path, "w", encoding="utf-8") as f:
                # Sort primarily by creation date for consistent file output
                sorted_tasks = sorted(self.tasks.values(), key=lambda t: t.created_at)
                json.dump(
                    [task.to_dict() for task in sorted_tasks],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as e:
            log.error(f"Error saving task queue to '{self.queue_path}': {e}")

    def add_task(self, task: Task) -> Optional[str]:
        if not isinstance(task, Task):
            log.error(f"Attempted to add non-Task object: {task}")
            return None
        # Simple duplicate check based on description (active tasks only)
        lower_desc = task.description.strip().lower()
        for existing_task in self.tasks.values():
            if (
                existing_task.description.strip().lower() == lower_desc
                and existing_task.status in ["pending", "planning", "in_progress"]
            ):
                log.warning(
                    f"Skipping task add - potential duplicate description for active task: '{task.description[:60]}...'"
                )
                return None
        self.tasks[task.id] = task
        self.save_queue()
        log.info(
            f"Task added: {task.id} - '{task.description[:60]}...' (Prio: {task.priority}, Depends: {task.depends_on})"
        )
        return task.id

    def get_next_task(self) -> Optional[Task]:
        """Gets the next runnable task, excluding those that exceeded reattempts."""
        available = []
        tasks_list = list(self.tasks.values())
        for task in tasks_list:
            # Only 'pending' tasks are candidates to be started
            if task.status == "pending":
                # --- Check reattempt limit ---
                if task.reattempt_count >= TASK_MAX_REATTEMPT:
                    log.warning(
                        f"Task {task.id} skipped: Exceeded max reattempts ({task.reattempt_count}/{TASK_MAX_REATTEMPT})."
                    )
                    # Mark as failed permanently if skipped here
                    self.update_task(
                        task.id,
                        "failed",
                        reflections=f"Skipped: Exceeded max reattempts ({task.reattempt_count}).",
                    )
                    continue  # Skip this task

                # --- Check dependencies ---
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
        # Sort by priority (descending) then creation time (ascending)
        return sorted(available, key=lambda t: (-t.priority, t.created_at))[0]

    def update_task(
        self,
        task_id: str,
        status: str,
        result: Any = None,
        reflections: Optional[str] = None,
        # No reattempt_count here, it's handled internally by agent
    ):
        """Updates basic task info like status, result, reflections. Saves queue."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            # Avoid unnecessary saves if nothing changed
            if (
                task.status == status
                and task.result == result
                and task.reflections == reflections
            ):
                return

            log.info(f"Updating task {task_id} status to: {status}")
            task.status = status
            if status in ["completed", "failed"] and not task.completed_at:
                task.completed_at = datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat()
            if result is not None:
                task.result = result
            if reflections is not None:
                task.reflections = reflections

            # Reset plan fields if task becomes pending again (e.g., manual reset not covered here)
            if status == "pending":
                task.plan = None
                task.current_step_index = 0
                task.cumulative_findings = ""
                # Reset reattempt count if manually set back to pending? Maybe not - let agent control this.

            # Clear plan details if completed or failed
            if status in ["completed", "failed"]:
                task.plan = None  # Don't need plan after completion/failure
                task.current_step_index = 0

            self.save_queue()
        else:
            log.error(f"Cannot update task - ID '{task_id}' not found.")

    # --- NEW: Method to specifically update task state for re-attempt ---
    def prepare_task_for_reattempt(self, task_id: str, lesson_learned: str) -> bool:
        """Resets a task's state for a re-attempt after failure."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            log.info(
                f"Preparing task {task_id} for re-attempt (Current count: {task.reattempt_count})."
            )

            task.reattempt_count += 1
            task.status = "planning"  # Go directly back to planning
            task.plan = None
            task.current_step_index = 0
            # Append the lesson learned to findings for context in the next planning phase
            task.cumulative_findings += f"\n--- Task Re-attempting (Attempt {task.reattempt_count}) ---\nLesson Learned: {lesson_learned}\n--- Resetting Plan ---\n"
            task.completed_at = None  # Clear completion time as it's restarting
            # Keep existing reflections (failure reason) but maybe append reattempt info?
            task.reflections = (
                task.reflections or ""
            ) + f"\n[Reattempting task - Attempt {task.reattempt_count}]"

            self.save_queue()
            log.info(
                f"Task {task.id} state reset for re-attempt {task.reattempt_count}. New status: planning."
            )
            return True
        else:
            log.error(f"Cannot prepare task for re-attempt - ID '{task_id}' not found.")
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
                f"Updating task {task_id} priority from {task.priority} to {new_priority}."
            )
            task.priority = new_priority
            self.save_queue()
            return True
        else:
            log.error(f"Cannot set priority - Task ID '{task_id}' not found.")
            return False

    def get_dependent_tasks(self, task_id: str) -> List[Task]:
        """Returns a list of tasks that depend on the given task_id."""
        if not task_id:
            return []
        dependent = [task for task in self.tasks.values() if task_id in task.depends_on]
        log.debug(f"Found {len(dependent)} tasks dependent on {task_id}.")
        return sorted(dependent, key=lambda t: (-t.priority, t.created_at))

    def get_tasks_by_status(self, statuses: List[str]) -> List[Task]:
        """Returns a list of tasks matching the provided statuses."""
        if not statuses:
            return []
        return [task for task in self.tasks.values() if task.status in statuses]

    def get_all_tasks_structured(self) -> Dict[str, List[Dict[str, Any]]]:
        """Returns tasks categorized by status and formatted for UI display."""
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

        # Process Pending
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

        # Process Planning
        planning_sorted = sorted(
            categorized["planning"], key=lambda t: (-t.priority, t.created_at)
        )
        output["planning"] = [
            {
                "ID": t.id,
                "Priority": t.priority,
                "Description": t.description,
                "Status": f"Planning (Attempt {t.reattempt_count + 1})",  # Show reattempt count
                "Created": t.created_at,
            }
            for t in planning_sorted
        ]

        # Process In Progress
        inprogress_sorted = sorted(
            categorized["in_progress"], key=lambda t: t.created_at
        )
        output["in_progress"] = [
            {
                "ID": t.id,
                "Priority": t.priority,
                "Description": t.description,
                "Status": (
                    f"Step {t.current_step_index + 1}/{len(t.plan)}"
                    if (t.plan and t.current_step_index < len(t.plan))
                    else "In Progress"
                )
                + (
                    f" (Attempt {t.reattempt_count + 1})"
                    if t.reattempt_count > 0
                    else ""
                ),
                "Created": t.created_at,
            }
            for t in inprogress_sorted
        ]

        # Process Completed
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

        # Process Failed
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
            }  # Show how many reattempts occurred before final failure
            for t in failed_sorted
        ]

        return output
