# FILE: tools/status_tool.py
# autonomous_agent/tools/status_tool.py
import datetime
from typing import Dict, Any, List
import logging
import os  # For os.path.join

from .base import Tool

# Note: This tool doesn't directly interact with config or utils,
# it receives the necessary data from the agent during the run call.

log = logging.getLogger("TOOL_Status")


class StatusTool(Tool):
    """
    Provides a summary of the agent's current status, including time, connectivity,
    identity, task queues, and memory summary. Takes no parameters.
    """

    def __init__(self):
        super().__init__(
            name="status",
            description=(
                "Provides a snapshot of the agent's current operational status. "
                "Includes current time, connectivity checks (Ollama, SearXNG), "
                "agent identity, task queue summary (pending, in-progress, recent completed/failed), "
                "and memory usage summary. "
                "**This tool takes NO parameters.**"
            ),
        )

    def _format_task_list(self, tasks: List[Dict[str, Any]], max_items: int = 5) -> str:
        """Formats a list of tasks for the status report."""
        if not tasks:
            return "  None"
        lines = []
        for i, task in enumerate(tasks[:max_items]):
            desc_snippet = task.get("Description", "N/A")[:70]
            prio_str = (
                f" (Prio: {task.get('Priority', 'N/A')})" if "Priority" in task else ""
            )
            id_str = f" (ID: {task.get('ID', 'N/A')[:8]}...)"
            lines.append(f"  - {desc_snippet}...{prio_str}{id_str}")
        if len(tasks) > max_items:
            lines.append(f"  ... and {len(tasks) - max_items} more.")
        return "\n".join(lines)

    def run(
        self,
        agent_identity: str,
        task_queue_summary: Dict[str, List[Dict[str, Any]]],
        memory_summary: Dict[str, int],
        ollama_status: str,
        searxng_status: str,
    ) -> Dict[str, Any]:
        """
        Constructs the status report string from the provided data.
        The agent gathers this data and passes it here.
        """
        log.info("Executing Status Report generation...")
        try:
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            now_local = now_utc.astimezone()  # Convert to local time for display

            report_parts = []
            report_parts.append(f"## Agent Status Report")
            report_parts.append(
                f"**Generated:** {now_local.strftime('%Y-%m-%d %H:%M:%S %Z')} ({now_utc.isoformat()})"
            )
            report_parts.append("\n**Connectivity:**")
            report_parts.append(f"- Ollama API: {ollama_status}")
            report_parts.append(f"- SearXNG API: {searxng_status}")

            report_parts.append("\n**Identity:**")
            report_parts.append(f"{agent_identity}")

            report_parts.append("\n**Task Queue Summary:**")
            # Pending (Highest priority first)
            pending_tasks = task_queue_summary.get("pending", [])
            report_parts.append(f"- Pending ({len(pending_tasks)}):")
            report_parts.append(self._format_task_list(pending_tasks, 5))

            # In Progress
            in_progress_tasks = task_queue_summary.get("in_progress", [])
            report_parts.append(f"- In Progress ({len(in_progress_tasks)}):")
            report_parts.append(
                self._format_task_list(in_progress_tasks, 1)
            )  # Usually only 1

            # Recently Completed (Most recent first)
            completed_tasks = task_queue_summary.get("completed", [])
            report_parts.append(f"- Recently Completed ({len(completed_tasks)} total):")
            report_parts.append(self._format_task_list(completed_tasks, 3))

            # Recently Failed (Most recent first)
            failed_tasks = task_queue_summary.get("failed", [])
            report_parts.append(f"- Recently Failed ({len(failed_tasks)} total):")
            report_parts.append(self._format_task_list(failed_tasks, 3))

            report_parts.append("\n**Memory Summary (by type):**")
            if memory_summary:
                # Sort by count descending for relevance
                sorted_mem_summary = sorted(
                    memory_summary.items(), key=lambda item: item[1], reverse=True
                )
                for mem_type, count in sorted_mem_summary:
                    report_parts.append(f"- {mem_type}: {count}")
            else:
                report_parts.append("  No memory summary available.")

            # Note: Chat history is not included as tools don't have direct access to UI state.

            report_content = "\n".join(report_parts)
            log.info("Status report generated successfully.")

            return {
                "status": "success",
                "action": "status_report",  # Use a specific action name
                "report_content": report_content,
                "message": "Agent status report generated.",
            }

        except Exception as e:
            log.exception("Unexpected error generating status report.")
            return {"error": f"Failed to generate status report: {e}"}
