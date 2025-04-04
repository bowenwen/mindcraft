# FILE: tools/memory_tool.py
import json
import traceback
from typing import Dict, Any, Optional, List

from .base import Tool
from memory import AgentMemory
from utils import format_relative_time
import config

import logging

log = logging.getLogger("TOOL_Memory")


class MemoryTool(Tool):
    """
    Interacts with the agent's long-term memory.
    Requires 'action' parameter: 'search' or 'write'.
    """

    def __init__(self):
        super().__init__(
            name="memory",
            description=(
                "Interacts with the agent's long-term memory. Requires 'action' parameter. "
                "Actions: "
                "'search' (requires 'query', optional 'n_results'): Searches memory using semantic query. Returns relevant memories. "
                "'write' (requires 'content'): Adds the provided text content to memory with a specific tag. Returns status and memory ID."
            ),
        )
        # Memory instance and identity are passed during run()

    def _run_search(
        self,
        query: str,
        n_results: int,
        memory_instance: AgentMemory,
        identity_statement: str,
    ) -> Dict[str, Any]:
        """Executes a semantic search query against the agent's memory."""
        log.info(f"Executing Memory Search: '{query}' (n_results={n_results})")
        try:
            retrieved_memories, separated_suggestion = (
                memory_instance.retrieve_and_rerank_memories(
                    query=query,
                    task_description="Fulfilling a direct memory retrieval request.",
                    context=f"Memory tool 'search' action query: {query}",
                    identity_statement=identity_statement,  # Pass identity
                    n_results=n_results * 2,  # Fetch more candidates for reranking
                    n_final=n_results,
                )
            )

            formatted_results = []
            if retrieved_memories:
                for i, mem in enumerate(retrieved_memories):
                    meta = mem.get("metadata", {})
                    mem_type = meta.get("type", "memory")
                    dist_str = (
                        f"{mem.get('distance', -1.0):.3f}"
                        if mem.get("distance") is not None
                        else "N/A"
                    )
                    relative_time = format_relative_time(meta.get("timestamp"))
                    # Limit snippet length for tool output consistency
                    content_snippet = (
                        mem.get("content", "N/A")[:500].replace("\n", " ") + "..."
                    )

                    formatted_results.append(
                        {
                            "rank": i + 1,
                            "relative_time": relative_time,
                            "type": mem_type,
                            "distance": dist_str,
                            "content_snippet": content_snippet,
                            "memory_id": mem.get("id", "N/A"),
                        }
                    )
                log.info(
                    f"Memory search action found {len(formatted_results)} relevant memories."
                )
                return {
                    "status": "success",
                    "action": "search",
                    "retrieved_memories": formatted_results,
                    "message": f"Memory search completed, found {len(formatted_results)} results.",
                }
            else:
                log.info(
                    "Memory search action found no relevant results for the query."
                )
                message = "No relevant memories found for the search query."
                # Inform if a suggestion was filtered out during search
                if separated_suggestion:
                    message += " Note: A potentially relevant user suggestion to 'move on' was identified during the search but is not included in results."
                return {
                    "status": "success",
                    "action": "search",
                    "retrieved_memories": [],
                    "message": message,
                }

        except Exception as e:
            log.exception(f"Unexpected error during memory tool 'search' action: {e}")
            return {"error": f"Unexpected error during memory search: {e}"}

    def _run_write(self, content: str, memory_instance: AgentMemory) -> Dict[str, Any]:
        """Writes the provided content to the agent's memory."""
        log.info(f"Executing Memory Write: '{content[:100]}...'")
        if not content or not isinstance(content, str) or not content.strip():
            return {"error": "Content for memory write cannot be empty."}

        try:
            # Add specific metadata to indicate this memory was added via the tool
            metadata = {
                "type": "agent_explicit_memory_write",
                "source_tool": "memory_tool",
                "source_action": "write",
            }
            memory_id = memory_instance.add_memory(content=content, metadata=metadata)

            if memory_id:
                log.info(f"Successfully wrote content to memory. ID: {memory_id}")
                return {
                    "status": "success",
                    "action": "write",
                    "memory_id": memory_id,
                    "message": "Content successfully added to memory.",
                }
            else:
                log.error("Failed to add content to memory (add_memory returned None).")
                return {
                    "error": "Failed to write to memory. Check agent logs for details."
                }
        except Exception as e:
            log.exception(f"Unexpected error during memory tool 'write' action: {e}")
            return {"error": f"Unexpected error writing to memory: {e}"}

    def run(
        self,
        parameters: Dict[str, Any],
        memory_instance: Optional[AgentMemory] = None,
        identity_statement: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs either the search or write action based on parameters."""
        if not memory_instance:
            log.error("Memory instance was not provided to MemoryTool.")
            return {"error": "Memory Tool internal error: Memory instance missing."}

        if not isinstance(parameters, dict):
            log.error(f"Invalid params type: {type(parameters)}")
            return {
                "error": "Invalid parameters format. Expected a JSON object (dict) with an 'action' key."
            }

        action = parameters.get("action")

        if action == "search":
            if not identity_statement:
                log.error(
                    "Agent identity statement was not provided for 'search' action."
                )
                return {
                    "error": "Memory Tool internal error: Agent identity missing for search."
                }
            query = parameters.get("query")
            if not query or not isinstance(query, str) or not query.strip():
                return {
                    "error": "Missing or invalid 'query' parameter for 'search' action."
                }
            try:
                n_results = int(
                    parameters.get("n_results", 10)
                )  # Use a reasonable default
                n_results = max(3, min(25, n_results))  # Clamp results
            except (ValueError, TypeError):
                log.warning(
                    f"Invalid n_results value: {parameters.get('n_results')}. Using default 10."
                )
                n_results = 10

            return self._run_search(
                query, n_results, memory_instance, identity_statement
            )

        elif action == "write":
            content = parameters.get("content")
            # Check for content specifically for write action
            if content is None or not isinstance(
                content, str
            ):  # Allow empty string technically, but maybe enforce non-empty?
                return {
                    "error": "Missing or invalid 'content' parameter (must be a string) for 'write' action."
                }
            # if not content.strip(): # Optionally enforce non-empty, non-whitespace content
            #     return {"error": "Content parameter cannot be empty or whitespace for 'write' action."}
            return self._run_write(content, memory_instance)

        else:
            log.error(f"Invalid action specified for memory tool: '{action}'")
            return {"error": "Invalid 'action' specified. Must be 'search' or 'write'."}
