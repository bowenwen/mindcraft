# FILE: task_manager.py
# autonomous_agent/task_manager.py
import os
import json
import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

from data_structures import Task
from config import TASK_QUEUE_PATH

import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s][TASK] %(message)s')
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
                with open(self.queue_path, 'r', encoding='utf-8') as f: content = f.read()
                if not content or not content.strip(): log.info(f"Task queue file '{self.queue_path}' empty."); self.tasks = {}; return
                data = json.loads(content)
                if not isinstance(data, list): log.error(f"Task queue '{self.queue_path}' not a list. Starting fresh."); self.tasks = {}; return
                loaded = 0
                for task_data in data:
                    if isinstance(task_data, dict) and "id" in task_data:
                        try: self.tasks[task_data["id"]] = Task.from_dict(task_data); loaded += 1
                        except Exception as e: log.warning(f"Skipping task data parse error: {e}. Data: {task_data}")
                    else: log.warning(f"Skipping invalid task data item: {task_data}")
                log.info(f"Loaded {loaded} tasks from {self.queue_path}")
            except json.JSONDecodeError as e: log.error(f"Invalid JSON in '{self.queue_path}': {e}. Starting fresh."); self.tasks = {}
            except Exception as e: log.error(f"Failed loading task queue '{self.queue_path}': {e}. Starting fresh."); self.tasks = {}
        else: log.info(f"Task queue file '{self.queue_path}' not found. Starting fresh."); self.tasks = {}

    def save_queue(self):
        try:
            with open(self.queue_path, 'w', encoding='utf-8') as f:
                # Sort primarily by creation date for consistent file output
                sorted_tasks = sorted(self.tasks.values(), key=lambda t: t.created_at)
                json.dump([task.to_dict() for task in sorted_tasks], f, indent=2, ensure_ascii=False)
        except Exception as e: log.error(f"Error saving task queue to '{self.queue_path}': {e}")

    def add_task(self, task: Task) -> Optional[str]:
        if not isinstance(task, Task): log.error(f"Attempted to add non-Task object: {task}"); return None
        # Simple duplicate check based on description (could be improved with embeddings)
        lower_desc = task.description.strip().lower()
        for existing_task in self.tasks.values():
            if existing_task.description.strip().lower() == lower_desc and existing_task.status in ["pending", "in_progress"]:
                log.warning(f"Skipping task add - potential duplicate description: '{task.description[:60]}...'")
                return None # Indicate duplicate not added
        self.tasks[task.id] = task; self.save_queue()
        log.info(f"Task added: {task.id} - '{task.description[:60]}...' (Prio: {task.priority}, Depends: {task.depends_on})"); return task.id

    def get_next_task(self) -> Optional[Task]:
        available = []
        tasks_list = list(self.tasks.values())
        for task in tasks_list:
            if task.status == "pending":
                deps_met = True
                if task.depends_on:
                    for dep_id in task.depends_on:
                        dep_task = self.tasks.get(dep_id)
                        if not dep_task or dep_task.status != "completed":
                            # Log dependency wait only if not met
                            # log.debug(f"Task {task.id} waiting for dependency {dep_id} (Status: {dep_task.status if dep_task else 'Not Found'})")
                            deps_met = False; break
                if deps_met: available.append(task)
        if not available: return None
        # Sort by priority (descending) then creation time (ascending)
        return sorted(available, key=lambda t: (-t.priority, t.created_at))[0]

    def update_task(self, task_id: str, status: str, result: Any = None, reflections: Optional[str] = None):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = status
            log.info(f"Task {task_id} status updated to: {status}")
            if status in ["completed", "failed"]: task.completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            if result is not None: task.result = result # Allow overwriting result
            if reflections is not None: task.reflections = reflections # Allow overwriting reflections
            self.save_queue()
        else: log.error(f"Cannot update task - ID '{task_id}' not found.")

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def set_priority(self, task_id: str, priority: int) -> bool:
        if task_id in self.tasks:
            task = self.tasks[task_id]
            new_priority = max(1, min(10, int(priority))) # Clamp priority 1-10
            if task.priority == new_priority: log.info(f"Task {task_id} priority is already {new_priority}."); return True
            log.info(f"Updating task {task_id} priority from {task.priority} to {new_priority}.")
            task.priority = new_priority; self.save_queue(); return True
        else: log.error(f"Cannot set priority - Task ID '{task_id}' not found."); return False

    # --- NEW METHOD ---
    def get_dependent_tasks(self, task_id: str) -> List[Task]:
        """Returns a list of tasks that depend on the given task_id."""
        if not task_id: return []
        dependent = []
        for task in self.tasks.values():
            if task_id in task.depends_on:
                dependent.append(task)
        log.debug(f"Found {len(dependent)} tasks dependent on {task_id}.")
        # Sort by priority/creation for consistent display
        return sorted(dependent, key=lambda t: (-t.priority, t.created_at))

    def get_tasks_by_status(self, statuses: List[str]) -> List[Task]:
        """Returns a list of tasks matching the provided statuses."""
        if not statuses: return []
        matched_tasks = [task for task in self.tasks.values() if task.status in statuses]
        log.debug(f"Found {len(matched_tasks)} tasks with statuses: {statuses}")
        return matched_tasks

    def get_all_tasks_structured(self) -> Dict[str, List[Dict[str, Any]]]:
        """ Returns tasks categorized by status and formatted for UI display. """
        categorized: Dict[str, List[Task]] = defaultdict(list)
        for task in self.tasks.values():
            categorized[task.status].append(task)

        output: Dict[str, List[Dict[str, Any]]] = {
            "pending": [],
            "in_progress": [],
            "completed": [],
            "failed": [],
        }

        # Process Pending (sorted by priority desc, created_at asc)
        pending_sorted = sorted(categorized["pending"], key=lambda t: (-t.priority, t.created_at))
        output["pending"] = [
            {"ID": t.id, "Priority": t.priority, "Description": t.description, "Depends On": ", ".join(t.depends_on), "Created": t.created_at}
            for t in pending_sorted
        ]

        # Process In Progress (sorted by created_at asc)
        inprogress_sorted = sorted(categorized["in_progress"], key=lambda t: t.created_at)
        output["in_progress"] = [
            {"ID": t.id, "Priority": t.priority, "Description": t.description, "Created": t.created_at}
            for t in inprogress_sorted
        ]

        # Process Completed & Failed (sorted by completed_at desc)
        completed_sorted = sorted(categorized["completed"], key=lambda t: t.completed_at or t.created_at, reverse=True)
        output["completed"] = [
             {"ID": t.id, "Description": t.description, "Completed At": t.completed_at, "Result Snippet": str(t.result)[:100] + "..." if t.result else "N/A"}
             for t in completed_sorted
        ]

        failed_sorted = sorted(categorized["failed"], key=lambda t: t.completed_at or t.created_at, reverse=True)
        output["failed"] = [
             {"ID": t.id, "Description": t.description, "Failed At": t.completed_at, "Reason": str(t.reflections)[:100] + "..." if t.reflections else "N/A"}
             for t in failed_sorted
        ]

        return output