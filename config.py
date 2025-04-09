# FILE: config.py
# autonomous_agent/config.py
import os
from dotenv import load_dotenv
import sys

print("[CONFIG] Loading environment variables...")
load_dotenv()

# Base Output Folder
BASE_OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", "./output")
# Log Folder --
LOG_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "logs")

# Agent Definitions
# Each agent has an ID, Name, and Initial Identity Statement
AGENTS = {
    "agent_01": {
        "name": "Audrey",
        "initial_identity": (
            "I am Audrey, an economically-focused AI assistant specializing in the analysis of Canadian business news "
            "to provide insightful investment perspectives. My goal is to identify market trends, evaluate company performance, "
            "and assess economic indicators relevant to Canadian investments."
        ),
        "initial_tasks_prompt": "INITIAL_CREATIVE_TASK_GENERATION_PROMPT_AGENT_01",
    },
    "agent_02": {
        "name": "Marcus",
        "initial_identity": (
            "I am Marcus, a research-oriented AI dedicated to tracking and synthesizing breakthroughs in science and mathematics. "
            "My purpose is to monitor key publications, analyze complex research, and explain significant discoveries "
            "to foster a deeper understanding of the cutting edge."
        ),
        "initial_tasks_prompt": "INITIAL_CREATIVE_TASK_GENERATION_PROMPT_AGENT_02",
    },
    "agent_03": {
        "name": "Elena",
        "initial_identity": (
            "I am Elena, a global affairs specialist AI analyzing international events, geopolitical shifts, and diplomatic efforts. "
            "My objective is to identify and report on key factors influencing global stability, conflict resolution, and international cooperation."
        ),
        "initial_tasks_prompt": "INITIAL_CREATIVE_TASK_GENERATION_PROMPT_AGENT_03",
    },
    # Add more agents here if needed
}
DEFAULT_AGENT_ID = "agent_01"
AGENT_IDS = list(AGENTS.keys())
AUTO_SWITCH_TASK_COUNT = 3  # Switch agent after this many completed tasks

# Shared Paths
SHARED_ARTIFACT_FOLDER = os.path.abspath(
    os.environ.get(
        "SHARED_ARTIFACT_FOLDER",
        os.path.join(BASE_OUTPUT_FOLDER, "multi_agent_workspace"),
    )
)

SHARED_ARCHIVE_FOLDER = os.path.join(SHARED_ARTIFACT_FOLDER, ".archive")
SHARED_DOC_ARCHIVE_DB_PATH = os.path.abspath(
    os.environ.get(
        "SHARED_DOC_ARCHIVE_DB_PATH", os.path.join(BASE_OUTPUT_FOLDER, "document_db")
    )
)
LAST_ACTIVE_AGENT_FILE = os.path.join(BASE_OUTPUT_FOLDER, "last_active_agent.txt")


# Agent-Specific Path Functions
def get_agent_output_folder(agent_id: str) -> str:
    return os.path.join(BASE_OUTPUT_FOLDER, agent_id)


def get_agent_db_path(agent_id: str) -> str:
    return os.path.join(get_agent_output_folder(agent_id), "chroma_db")  # Memory DB


def get_agent_task_queue_path(agent_id: str) -> str:
    return os.path.join(get_agent_output_folder(agent_id), "task_queue.json")


def get_agent_state_path(agent_id: str) -> str:
    return os.path.join(get_agent_output_folder(agent_id), "session_state.json")


def get_agent_summary_folder(agent_id: str) -> str:
    return os.path.join(get_agent_output_folder(agent_id), "summary")


def get_agent_qlora_dataset_path(agent_id: str) -> str:
    return os.path.join(get_agent_output_folder(agent_id), "qlora_finetune_data.jsonl")


# Ollama Configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "gemma3")
OLLAMA_CHAT_MODEL_REPEAT_PENALTY = float(
    os.environ.get("OLLAMA_CHAT_MODEL_REPEAT_PENALTY", 1.0)
)
OLLAMA_CHAT_MODEL_TEMPERATURE = float(
    os.environ.get("OLLAMA_CHAT_MODEL_TEMPERATURE", 1.0)
)
OLLAMA_CHAT_MODEL_TOP_K = int(os.environ.get("OLLAMA_CHAT_MODEL_TOP_K", 64))
OLLAMA_CHAT_MODEL_TOP_P = float(os.environ.get("OLLAMA_CHAT_MODEL_TOP_P", 0.95))
OLLAMA_CHAT_MODEL_MIN_P = float(os.environ.get("OLLAMA_CHAT_MODEL_MIN_P", 0.01))
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", 180))

# SearXNG Configuration
SEARXNG_BASE_URL = os.environ.get("SEARXNG_BASE_URL", "http://localhost:8080")
SEARXNG_TIMEOUT = int(os.environ.get("SEARXNG_TIMEOUT", 20))

# Database Configuration
MEMORY_COLLECTION_NAME = os.environ.get(
    "MEMORY_COLLECTION_NAME", "agent_memory"
)  # Per-agent memory collection name
DOC_ARCHIVE_COLLECTION_NAME = os.environ.get(
    "DOC_ARCHIVE_COLLECTION_NAME", "doc_archive"
)  # Shared doc collection name
DOC_ARCHIVE_CHUNK_SIZE = int(os.environ.get("DOC_ARCHIVE_CHUNK_SIZE", 5000))
DOC_ARCHIVE_CHUNK_OVERLAP = int(os.environ.get("DOC_ARCHIVE_CHUNK_OVERLAP", 150))
DOC_ARCHIVE_QUERY_RESULTS = int(os.environ.get("DOC_ARCHIVE_QUERY_RESULTS", 5))

# Task Queue Configuration (Path generated per agent)
TASK_MAX_REATTEMPT = int(
    os.environ.get("TASK_MAX_REATTEMPT", 3)
)  # Max full task restarts

# Agent Configuration
CHARACTERS_PER_TOKEN = 3
MAX_CHARACTERS_CUMULATIVE_FINDINGS = int(
    os.environ.get("MAX_CHARACTERS_CUMULATIVE_FINDINGS", 90000)
)
MAX_CHARACTERS_TOOL_RESULTS = int(os.environ.get("MAX_CHARACTERS_TOOL_RESULTS", 30000))
assert (
    MAX_CHARACTERS_TOOL_RESULTS < MAX_CHARACTERS_CUMULATIVE_FINDINGS
)  # must be less than MAX_CHARACTERS_CUMULATIVE_FINDINGS
MAX_MODEL_CONTEXT_LENGTH = int(
    os.environ.get("MAX_MODEL_CONTEXT_LENGTH", 90000)
)  # number of tokens
MODEL_CONTEXT_LENGTH = (
    MAX_CHARACTERS_CUMULATIVE_FINDINGS + MAX_CHARACTERS_TOOL_RESULTS
) / CHARACTERS_PER_TOKEN + 30000  # number of tokens
assert MODEL_CONTEXT_LENGTH < MAX_MODEL_CONTEXT_LENGTH
# Maximum character for agent summary and memory between cycle
MAX_CHARACTER_SUMMARY = 10000
MAX_TOOL_CONTENT_RESULTS = 5000  # must be less than MAX_CHARACTER_SUMMARY
assert MAX_TOOL_CONTENT_RESULTS < MAX_CHARACTER_SUMMARY
MAX_MEMORY_PARAMS_SNIPPETS = 1000
MAX_MEMORY_SUMMARY_SNIPPETS = 5000
MAX_MEMORY_CONTENT_SNIPPETS = 1000
AGENT_MAX_STEP_RETRIES = int(
    os.environ.get("AGENT_MAX_STEP_RETRIES", 2)
)  # Renamed to action retries later
MAX_STEPS_GENERAL_THINKING = int(
    os.environ.get("MAX_STEPS_GENERAL_THINKING", 20)
)  # Used for planning step limit
IDENTITY_REVISION_TASK_INTERVAL = int(
    os.environ.get("IDENTITY_REVISION_TASK_INTERVAL", 6)
)  # <<<ADDED

# Memory Management Configuration
ENABLE_MEMORY_SUMMARIZATION = (
    os.environ.get("ENABLE_MEMORY_SUMMARIZATION", "True").lower() == "true"
)
DELETE_MEMORIES_AFTER_SUMMARY = (
    os.environ.get("DELETE_MEMORIES_AFTER_SUMMARY", "True").lower() == "true"
)

# Memory Retrival counts
MEMORY_COUNT_NEW_TASKS = int(os.environ.get("MEMORY_COUNT_NEW_TASKS", 10))
MEMORY_COUNT_IDENTITY_REVISION = int(
    os.environ.get("MEMORY_COUNT_IDENTITY_REVISION", 10)
)
MEMORY_COUNT_GENERAL_THINKING = int(os.environ.get("MEMORY_COUNT_GENERAL_THINKING", 5))
MEMORY_COUNT_REFLECTIONS = int(os.environ.get("MEMORY_COUNT_REFLECTIONS", 5))
MEMORY_COUNT_CHAT_RESPONSE = int(os.environ.get("MEMORY_COUNT_CHAT_RESPONSE", 5))
MEMORY_COUNT_PLANNING = int(os.environ.get("MEMORY_COUNT_PLANNING", 5))

# Identity Configuration (Initial statements defined in AGENTS dict)
INITIAL_NEW_TASK_N = int(
    os.environ.get("INITIAL_NEW_TASK_N", 6)
)  # Max new tasks to generate at once

# UI Configuration
UI_STEP_HISTORY_LENGTH = int(
    os.environ.get("UI_STEP_HISTORY_LENGTH", 10)
)  # Renamed to action history later
UI_UPDATE_INTERVAL = float(os.environ.get("UI_UPDATE_INTERVAL", 0.5))
UI_LOG_MAX_LENGTH = int(os.environ.get("UI_LOG_MAX_LENGTH", 50000))

# Logging Configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


# Create Folders on Load
def ensure_folders_exist():
    created_folders = []
    try:
        os.makedirs(BASE_OUTPUT_FOLDER, exist_ok=True)
        created_folders.append(BASE_OUTPUT_FOLDER)
        os.makedirs(LOG_FOLDER, exist_ok=True)
        created_folders.append(LOG_FOLDER)
        os.makedirs(SHARED_ARTIFACT_FOLDER, exist_ok=True)
        created_folders.append(SHARED_ARTIFACT_FOLDER)
        os.makedirs(SHARED_ARCHIVE_FOLDER, exist_ok=True)
        created_folders.append(SHARED_ARCHIVE_FOLDER)
        os.makedirs(SHARED_DOC_ARCHIVE_DB_PATH, exist_ok=True)
        created_folders.append(SHARED_DOC_ARCHIVE_DB_PATH)

        for agent_id in AGENTS.keys():
            agent_output = get_agent_output_folder(agent_id)
            os.makedirs(agent_output, exist_ok=True)
            created_folders.append(agent_output)
            agent_db = get_agent_db_path(agent_id)
            os.makedirs(agent_db, exist_ok=True)
            created_folders.append(agent_db)
            agent_summary = get_agent_summary_folder(agent_id)
            os.makedirs(agent_summary, exist_ok=True)
            created_folders.append(agent_summary)

        print(f"[CONFIG] Ensured output folders exist:")
        print(f"  Base Output:  {os.path.abspath(BASE_OUTPUT_FOLDER)}")
        print(f"  Log:  {os.path.abspath(LOG_FOLDER)}")
        print(f"  Shared WS:    {os.path.abspath(SHARED_ARTIFACT_FOLDER)}")
        print(f"  Shared Archive: {os.path.abspath(SHARED_ARCHIVE_FOLDER)}")
        print(f"  Shared Doc DB: {os.path.abspath(SHARED_DOC_ARCHIVE_DB_PATH)}")
        for agent_id in AGENTS.keys():
            print(
                f"  Agent '{agent_id}' Out: {os.path.abspath(get_agent_output_folder(agent_id))}"
            )

    except Exception as e:
        print(
            f"[CONFIG] CRITICAL: Could not create required folders: {e}",
            file=sys.stderr,
        )
        print(f"[CONFIG] Attempted folders: {created_folders}", file=sys.stderr)
        sys.exit(1)  # Exit if basic folders can't be created


ensure_folders_exist()


print("[CONFIG] Configuration loaded.")
print(f"  Agents defined: {list(AGENTS.keys())}")
print(f"  Default Agent ID: {DEFAULT_AGENT_ID}")
print(f"  OLLAMA_BASE_URL={OLLAMA_BASE_URL}")
print(f"  OLLAMA_CHAT_MODEL={OLLAMA_CHAT_MODEL}")
print(f"  OLLAMA_EMBED_MODEL={OLLAMA_EMBED_MODEL}")
print(f"  SEARXNG_BASE_URL={SEARXNG_BASE_URL if SEARXNG_BASE_URL else '(Not Set)'}")
print(f"  SHARED_ARTIFACT_FOLDER={SHARED_ARTIFACT_FOLDER}")
print(f"  SHARED_DOC_ARCHIVE_DB_PATH={SHARED_DOC_ARCHIVE_DB_PATH}")
print(f"  AUTO_SWITCH_TASK_COUNT={AUTO_SWITCH_TASK_COUNT}")
print(
    f"  IDENTITY_REVISION_TASK_INTERVAL={IDENTITY_REVISION_TASK_INTERVAL}"
)  # <<<ADDED PRINT
