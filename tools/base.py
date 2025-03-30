# autonomous_agent/tools/base.py
from typing import Dict, Any

class Tool:
    """Base class for tools the agent can use."""
    def __init__(self, name: str, description: str):
        if not name or not description:
            raise ValueError("Tool name and description cannot be empty.")
        self.name = name
        self.description = description

    def run(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the tool's action with the given parameters.
        Returns a dictionary containing the results or an error.
        """
        raise NotImplementedError("Tool subclass must implement run method.")