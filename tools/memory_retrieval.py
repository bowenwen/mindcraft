# FILE: tools/memory_retrieval.py
# autonomous_agent/tools/memory_retrieval.py
import json
import traceback
from typing import Dict, Any, Optional, List

# Import base class, agent memory, and config
from .base import Tool
from memory import AgentMemory
from utils import format_relative_time
import config

# Use logging module
import logging
log = logging.getLogger("TOOL_MemoryRetrieval")


class MemoryRetrievalTool(Tool):
    """
    Tool for searching the agent's memory based on a semantic query.
    Retrieves relevant memories considering relevance, recency, and novelty.
    """
    def __init__(self):
        super().__init__(
            name="memory_retrieval",
            description=(
                "Searches the agent's long-term memory for information relevant to a given query. "
                "Useful for recalling past findings, errors, reflections, or specific details needed for the current task. "
                "Returns a ranked list of relevant memories with relative timestamps. "
                "Parameters: query (str, required), n_results (int, optional, default=10)"
            )
        )
        # The AgentMemory instance and identity_statement are passed during run()

    # --- MODIFIED: Added identity_statement parameter ---
    def run(self, parameters: Dict[str, Any], memory_instance: Optional[AgentMemory] = None, identity_statement: Optional[str] = None) -> Dict[str, Any]:
        """Executes a semantic search query against the agent's memory."""
        if not memory_instance:
            log.error("Memory instance was not provided to MemoryRetrievalTool.")
            return {"error": "Memory Retrieval Tool internal error: Memory instance missing."}
        # --- ADDED: Check for identity statement ---
        if not identity_statement:
            log.error("Agent identity statement was not provided to MemoryRetrievalTool.")
            return {"error": "Memory Retrieval Tool internal error: Agent identity missing."}

        if not isinstance(parameters, dict):
            log.error(f"Invalid params type: {type(parameters)}")
            return {"error": "Invalid parameters format. Expected a JSON object (dict)."}

        query = parameters.get("query")
        if not query or not isinstance(query, str) or not query.strip():
            return {"error": "Missing or invalid 'query' parameter (must be a non-empty string)."}

        try:
            n_results = int(parameters.get("n_results", 10))
            n_results = max(3, min(25, n_results))
        except ValueError:
             log.warning(f"Invalid n_results value: {parameters.get('n_results')}. Using default 10.")
             n_results = 10

        log.info(f"Executing Memory Search: '{query}' (n_results={n_results})")

        try:
            # --- MODIFIED: Pass identity_statement to the memory method ---
            retrieved_memories, separated_suggestion = memory_instance.retrieve_and_rerank_memories(
                query=query,
                task_description="Fulfilling a direct memory retrieval request.",
                context=f"Memory retrieval tool query: {query}",
                identity_statement=identity_statement, # <<<--- Pass identity
                n_results=n_results * 2,
                n_final=n_results
            )

            # --- Format the results for the LLM ---
            formatted_results = []
            if retrieved_memories:
                for i, mem in enumerate(retrieved_memories):
                    meta = mem.get('metadata', {})
                    mem_type = meta.get('type', 'memory')
                    dist_str = f"{mem.get('distance', -1.0):.3f}" if mem.get('distance') is not None else "N/A"
                    relative_time = format_relative_time(meta.get('timestamp'))
                    content_snippet = mem.get('content', 'N/A')[:500].replace('\n', ' ') + "..."

                    formatted_results.append({
                        "rank": i + 1,
                        "relative_time": relative_time,
                        "type": mem_type,
                        "distance": dist_str,
                        "content_snippet": content_snippet,
                        "memory_id": mem.get('id', 'N/A')
                    })
                log.info(f"Memory retrieval tool found {len(formatted_results)} relevant memories.")
                return {"retrieved_memories": formatted_results}
            else:
                log.info("Memory retrieval tool found no relevant results for the query.")
                message = "No relevant memories found."
                if separated_suggestion:
                    message += " Note: A potentially relevant user suggestion to 'move on' was also identified during the search."
                return {"retrieved_memories": [], "message": message}

        except Exception as e:
            log.exception(f"Unexpected error during memory retrieval tool execution: {e}")
            return {"error": f"Unexpected error during memory search: {e}"}