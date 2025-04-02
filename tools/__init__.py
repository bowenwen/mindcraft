# FILE: tools/__init__.py
# autonomous_agent/tools/__init__.py
from typing import Dict

# Import tool implementations
from .base import Tool
from .web_search import WebSearchTool
from .web_browse import WebBrowserTool
from .memory_retrieval import MemoryRetrievalTool
from .artifact_writer import ArtifactWriterTool  # <<<--- ADD IMPORT
from .artifact_reader import ArtifactReaderTool  # <<<--- ADD IMPORT

# Suggestion: Automatically discover tools or use a registration pattern?
# For now, manually list them.
AVAILABLE_TOOLS: Dict[str, Tool] = {
    "web_search": WebSearchTool(),
    "web_browse": WebBrowserTool(),
    "memory_retrieval": MemoryRetrievalTool(),
    "artifact_writer": ArtifactWriterTool(),  # <<<--- ADD NEW TOOL INSTANCE
    "artifact_reader": ArtifactReaderTool(),  # <<<--- ADD NEW TOOL INSTANCE
    # Add other tools here: e.g., "code_interpreter": CodeInterpreterTool()
}


def load_tools() -> Dict[str, Tool]:
    """Returns a dictionary of available tool instances."""
    # Could add logic here to dynamically load tools if needed
    print(f"[SETUP] Loading tools: {list(AVAILABLE_TOOLS.keys())}")
    return AVAILABLE_TOOLS
