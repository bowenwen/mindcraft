# FILE: tools/__init__.py
# autonomous_agent/tools/__init__.py
from typing import Dict

# Import tool implementations
from .base import Tool
from .web_tool import WebTool            # <<<--- Use new WebTool
from .memory_tool import MemoryTool      # <<<--- Use new MemoryTool
from .file_tool import FileTool          # <<<--- Use new FileTool

# Instantiate the consolidated tools
AVAILABLE_TOOLS: Dict[str, Tool] = {
    "web": WebTool(),
    "memory": MemoryTool(),
    "file": FileTool(),
    # Add other future tools here
}

def load_tools() -> Dict[str, Tool]:
    """Returns a dictionary of available tool instances."""
    print(f"[SETUP] Loading tools: {list(AVAILABLE_TOOLS.keys())}")
    return AVAILABLE_TOOLS