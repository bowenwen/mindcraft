# autonomous_agent/main.py
import os
import json
import traceback
import sys
import logging
from typing import Optional

# --- Project Imports ---
import config
from llm_utils import call_ollama_api
from memory import setup_chromadb # Function to setup DB
from agent import AutonomousAgent
import chromadb

# --- Basic Logging Setup for Main Script ---
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='[%(asctime)s] [%(levelname)s][%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger("MAIN")

# --- Function to Generate Initial Topic ---
def generate_initial_topic(model: str = config.OLLAMA_CHAT_MODEL, base_url: str = config.OLLAMA_BASE_URL) -> Optional[str]:
    """Asks the LLM to generate an interesting research topic."""
    log.info("Asking LLM to generate an initial research topic...")
    prompt = """Suggest one interesting, specific, and researchable topic suitable for an AI agent with web search capabilities to investigate and summarize. The topic should require gathering and synthesizing information from multiple sources. Avoid overly broad or simple topics. Output only the topic description itself, without any preamble, quotes, or explanation."""
    log.info(f"Asking {model} for initial topic suggestion...")
    topic = call_ollama_api(prompt, model, base_url, timeout=60)
    if topic and topic.strip():
        cleaned_topic = topic.strip().strip('"').strip("'").strip()
        log.info(f"LLM suggested initial topic: '{cleaned_topic}'")
        return cleaned_topic
    else:
        log.warning("LLM failed to generate an initial topic.")
        return None

# --- Main Execution Helper ---
def run_agent_and_collect_data(investigation_topic: str,
                               duration_minutes: int = config.DEFAULT_SESSION_DURATION_MINUTES,
                               max_steps_per_task: int = config.DEFAULT_MAX_STEPS_PER_TASK,
                               memory_collection: Optional[chromadb.Collection] = None):
    """Initializes agent, ensures initial task, runs session, returns results."""
    print(f"\n{'#'*10} Starting Agent Investigation {'#'*10}") # Use print for major demarcations
    log.info(f"Topic: '{investigation_topic}'")
    log.info(f"Duration: {duration_minutes} min, Max Steps/Task: {max_steps_per_task}")

    try:
        agent = AutonomousAgent(memory_collection=memory_collection)
    except Exception as e:
        log.critical(f"Failed to initialize AutonomousAgent: {e}", exc_info=True)
        return None # Cannot proceed if agent fails init

    task_desc = f"Investigate and provide a comprehensive summary of: {investigation_topic}"; existing_task = None
    for task in agent.task_queue.tasks.values():
        if task.description == task_desc and task.status in ["pending", "in_progress"]: existing_task = task; break

    if not existing_task:
        log.info(f"Creating initial investigation task: '{task_desc[:60]}...' (Prio 10)")
        agent.create_task(task_desc, priority=10)
    else:
         log.info(f"Found existing relevant task {existing_task.id} (Status: {existing_task.status}). Not creating duplicate.")

    # Session results now include counts for completed, failed, generated
    session_results = agent.run_session(duration_minutes, max_steps_per_task)

    all_tasks_final_state = list(agent.task_queue.tasks.values())
    print(f"\n[MAIN] {'#'*10} Investigation Session Finished {'#'*10}") # Use print
    return {
        "investigation_topic": investigation_topic,
        "session_parameters": {
            "duration_minutes": duration_minutes,
            "max_steps_per_task": max_steps_per_task
        },
        "session_results": session_results, # Pass the dict containing counts
        "final_task_queue_state": [t.to_dict() for t in all_tasks_final_state]
    }

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Autonomous Agent Script Start ---") # Use print for initial start

    log.info("--- Configuration Check ---")
    log.info(f"Ollama Chat: {config.OLLAMA_CHAT_MODEL}, Embed: {config.OLLAMA_EMBED_MODEL} @ {config.OLLAMA_BASE_URL}")
    log.info(f"ChromaDB Path: {config.DB_PATH}")
    if not config.SEARXNG_BASE_URL: log.error("SEARXNG_BASE_URL not set. Web search will fail.")
    else: log.info(f"SearXNG URL: {config.SEARXNG_BASE_URL}")
    log.info(f"Log Level Set To: {config.LOG_LEVEL}")
    log.info(f"Summary Folder: {config.SUMMARY_FOLDER}") # Log summary folder path
    log.info("-" * 25)

    log.info("--- Database Setup ---")
    mem_collection = setup_chromadb()
    if mem_collection is None: log.critical("Exiting due to database setup failure."); sys.exit(1)
    log.info("-" * 25)

    try:
        log.info("--- Initial Topic Generation ---")
        initial_topic = generate_initial_topic()
        if not initial_topic:
            log.warning("Using fallback topic as LLM failed to generate one.")
            initial_topic = "The future potential and challenges of fusion power generation" # Example fallback
        log.info("-" * 25)

        session_duration = config.DEFAULT_SESSION_DURATION_MINUTES
        steps_limit = config.DEFAULT_MAX_STEPS_PER_TASK

        investigation_data = run_agent_and_collect_data(
            investigation_topic=initial_topic,
            duration_minutes=session_duration,
            max_steps_per_task=steps_limit,
            memory_collection=mem_collection # Pass collection
        )

        if investigation_data:
            results_file = config.INVESTIGATION_RES_PATH
            try:
                log.info(f"Saving investigation results to {results_file}...")
                with open(results_file, "w", encoding='utf-8') as f:
                    json.dump(investigation_data, f, indent=2, ensure_ascii=False, default=str)
                log.info(f"Results saved.")
            except Exception as e: log.error(f"Error saving results to {results_file}: {e}")
        else:
             log.error("Agent investigation run failed, skipping results save.")

    except KeyboardInterrupt: print("\n--- KeyboardInterrupt received. Shutting down agent gracefully. ---")
    except Exception as main_err: log.critical(f"UNEXPECTED MAIN EXECUTION ERROR: {main_err}", exc_info=True)
    finally:
        print("\n--- Autonomous Agent Script End ---") # Use print