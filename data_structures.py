# autonomous_agent/data_structures.py
import uuid
import datetime
from typing import List, Dict, Any, Optional

class Task:
    """Represents a single task with description, priority, status, and dependencies."""
    def __init__(self, description: str, priority: int = 1, depends_on: Optional[List[str]] = None):
        self.id = str(uuid.uuid4())
        self.description = description
        self.priority = max(1, min(10, priority)) # Ensure priority is 1-10
        self.depends_on = depends_on if isinstance(depends_on, list) else ([depends_on] if isinstance(depends_on, str) else [])
        self.status = "pending" # pending, in_progress, completed, failed
        self.created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.completed_at = None
        self.result = None # Store final answer or outcome
        self.reflections = None # Store LLM reflections on task execution

    def to_dict(self) -> Dict[str, Any]:
        """Serializes task object to a dictionary."""
        return {
            "id": self.id, "description": self.description, "priority": self.priority,
            "depends_on": self.depends_on, "status": self.status, "created_at": self.created_at,
            "completed_at": self.completed_at, "result": self.result, "reflections": self.reflections
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Deserializes task object from a dictionary."""
        task = cls(
            description=data.get("description", "No description provided"),
            priority=data.get("priority", 1), depends_on=data.get("depends_on")
        )
        # Restore all fields
        task.id = data.get("id", task.id)
        task.status = data.get("status", "pending")
        task.created_at = data.get("created_at", task.created_at)
        task.completed_at = data.get("completed_at")
        task.result = data.get("result")
        task.reflections = data.get("reflections")
        return task