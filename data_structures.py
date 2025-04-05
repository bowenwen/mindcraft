# autonomous_agent/data_structures.py
import uuid
import datetime
from typing import List, Dict, Any, Optional


class Task:
    """Represents a single task with description, priority, status, and dependencies."""

    def __init__(
        self,
        description: str,
        priority: int = 1,
        depends_on: Optional[List[str]] = None,
    ):
        self.id = str(uuid.uuid4())
        self.description = description
        self.priority = max(1, min(10, priority))  # Ensure priority is 1-10
        self.depends_on = (
            depends_on
            if isinstance(depends_on, list)
            else ([depends_on] if isinstance(depends_on, str) else [])
        )
        self.status = "pending"  # pending, planning, in_progress, completed, failed
        self.created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.completed_at = None
        self.result = None  # Store final answer or outcome
        self.reflections = None  # Store LLM reflections on task execution

        # --- Task Planning Fields ---
        self.plan: Optional[List[str]] = None  # The generated list of steps
        self.current_step_index: int = 0  # Index of the next step to execute (0-based)
        self.cumulative_findings: str = ""  # Summary of results from completed steps

        # --- NEW Field for Task Re-attempts ---
        self.reattempt_count: int = (
            0  # Number of times this task has been fully restarted
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes task object to a dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority,
            "depends_on": self.depends_on,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "reflections": self.reflections,
            "plan": self.plan,
            "current_step_index": self.current_step_index,
            "cumulative_findings": self.cumulative_findings,
            "reattempt_count": self.reattempt_count,  # <<<--- Serialize new field
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Deserializes task object from a dictionary."""
        task = cls(
            description=data.get("description", "No description provided"),
            priority=data.get("priority", 1),
            depends_on=data.get("depends_on"),
        )
        # Restore all fields
        task.id = data.get("id", task.id)
        task.status = data.get("status", "pending")
        task.created_at = data.get("created_at", task.created_at)
        task.completed_at = data.get("completed_at")
        task.result = data.get("result")
        task.reflections = data.get("reflections")
        task.plan = data.get("plan")
        task.current_step_index = data.get("current_step_index", 0)
        task.cumulative_findings = data.get("cumulative_findings", "")
        task.reattempt_count = data.get(
            "reattempt_count", 0
        )  # <<<--- Deserialize new field (default 0)

        return task
