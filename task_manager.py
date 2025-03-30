# autonomous_agent/task_manager.py
import os
import json
import datetime
from typing import List, Dict, Any, Optional

# Import Task from data_structures
from data_structures import Task
# Import specific config value needed
from config import TASK_QUEUE_PATH

# Suggestion: Use logging module
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s][TASK] %(message)s')
log = logging.getLogger(__name__)


class TaskQueue:
    """Manages the persistence and retrieval of tasks."""
    def __init__(self, queue_path: str = TASK_QUEUE_PATH):
        self.queue_path = queue_path
        self.tasks: Dict[str, Task] = {} # {task_id: Task object}
        self.load_queue()

    def load_queue(self):
        """Loads tasks from the JSON file."""
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
        """Saves the current task list to the JSON file."""
        try:
            with open(self.queue_path, 'w', encoding='utf-8') as f:
                sorted_tasks = sorted(self.tasks.values(), key=lambda t: t.created_at)
                json.dump([task.to_dict() for task in sorted_tasks], f, indent=2, ensure_ascii=False)
        except Exception as e: log.error(f"Error saving task queue to '{self.queue_path}': {e}")

    def add_task(self, task: Task) -> Optional[str]:
        """Adds a task to the queue and saves."""
        if not isinstance(task, Task): log.error(f"Attempted to add non-Task object: {task}"); return None
        self.tasks[task.id] = task; self.save_queue()
        log.info(f"Task added: {task.id} - '{task.description[:60]}...' (Prio: {task.priority}, Depends: {task.depends_on})"); return task.id

    def get_next_task(self) -> Optional[Task]:
        """Gets the highest priority 'pending' task whose dependencies are 'completed'."""
        available = []
        tasks_list = list(self.tasks.values()) # Stable list for iteration
        for task in tasks_list:
            if task.status == "pending":
                deps_met = True
                if task.depends_on: # Only check if dependencies exist
                    for dep_id in task.depends_on:
                        dep_task = self.tasks.get(dep_id)
                        # Dependency must exist AND be completed
                        if not dep_task or dep_task.status != "completed":
                            # log.debug(f"Task {task.id} dependency {dep_id} not met (status: {dep_task.status if dep_task else 'Not Found'}).") # Verbose Debug
                            deps_met = False; break
                if deps_met:
                    available.append(task)
        if not available: return None
        # Sort by priority (desc), then creation time (asc)
        return sorted(available, key=lambda t: (-t.priority, t.created_at))[0]

    def update_task(self, task_id: str, status: str, result: Any = None, reflections: Optional[str] = None):
        """Updates the status and optionally result/reflections of a task."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status == status and result is None and reflections is None: return # Avoid redundant saves/logs if nothing changed
            task.status = status
            log.info(f"Task {task_id} status updated to: {status}")
            if status in ["completed", "failed"]: task.completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            if result is not None: task.result = result
            if reflections is not None: task.reflections = reflections
            self.save_queue() # Save after any update
        else: log.error(f"Cannot update task - ID '{task_id}' not found.")

    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieves a task by its ID."""
        return self.tasks.get(task_id)