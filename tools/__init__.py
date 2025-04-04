# FILE: tools/__init__.py
# autonomous_agent/tools/__init__.py
from typing import Dict

# Import tool implementations
from .base import Tool
from .web_tool import WebTool
from .memory_tool import MemoryTool
from .file_tool import FileTool
from .status_tool import StatusTool  # <<<--- ADDED IMPORT

# Instantiate the consolidated tools
AVAILABLE_TOOLS: Dict[str, Tool] = {
    "web": WebTool(),
    "memory": MemoryTool(),
    "file": FileTool(),
    "status": StatusTool(),  # <<<--- ADDED INSTANCE
    # Add other future tools here
}


def load_tools() -> Dict[str, Tool]:
    """Returns a dictionary of available tool instances."""
    print(f"[SETUP] Loading tools: {list(AVAILABLE_TOOLS.keys())}")
    return AVAILABLE_TOOLS
