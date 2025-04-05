# FILE: config.py
# autonomous_agent/config.py
import os
from dotenv import load_dotenv

print("[CONFIG] Loading environment variables...")
load_dotenv()

# --- Common Configuration ---
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", "./output")
ARTIFACT_FOLDER = os.environ.get(
    "ARTIFACT_FOLDER", os.path.join(OUTPUT_FOLDER, "artifacts")
)
ARCHIVE_FOLDER = os.path.join(ARTIFACT_FOLDER, ".archive")
# --- NEW: Log Folder ---
LOG_FOLDER = os.path.join(OUTPUT_FOLDER, "logs")


# --- Ollama Configuration ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "gemma3")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", 180))

# --- SearXNG Configuration ---
SEARXNG_BASE_URL = os.environ.get("SEARXNG_BASE_URL", "http://localhost:8080")
SEARXNG_TIMEOUT = int(os.environ.get("SEARXNG_TIMEOUT", 20))

# --- Database Configuration ---
DB_PATH = os.environ.get("DB_PATH", "./chroma_db")
MEMORY_COLLECTION_NAME = os.environ.get("MEMORY_COLLECTION_NAME", "agent_memory")
DOC_ARCHIVE_DB_PATH = os.environ.get("DOC_ARCHIVE_DB_PATH", "./chroma_doc_archive")
DOC_ARCHIVE_COLLECTION_NAME = os.environ.get(
    "DOC_ARCHIVE_COLLECTION_NAME", "doc_archive"
)
DOC_ARCHIVE_CHUNK_SIZE = int(os.environ.get("DOC_ARCHIVE_CHUNK_SIZE", 5000))
DOC_ARCHIVE_CHUNK_OVERLAP = int(os.environ.get("DOC_ARCHIVE_CHUNK_OVERLAP", 150))
DOC_ARCHIVE_QUERY_RESULTS = int(os.environ.get("DOC_ARCHIVE_QUERY_RESULTS", 5))


# --- Task Queue Configuration ---
TASK_QUEUE_PATH = f"{OUTPUT_FOLDER}/task_queue.json"
INVESTIGATION_RES_PATH = f"{OUTPUT_FOLDER}/investigation_results.json"

# --- Agent Configuration ---
AGENT_STATE_PATH = f"{OUTPUT_FOLDER}/session_state.json"
CONTEXT_TRUNCATION_LIMIT = int(os.environ.get("CONTEXT_TRUNCATION_LIMIT", 30000))
MODEL_CONTEXT_LENGTH = CONTEXT_TRUNCATION_LIMIT + 30000
AGENT_MAX_STEP_RETRIES = int(os.environ.get("AGENT_MAX_STEP_RETRIES", 2))
MAX_STEPS_GENERAL_THINKING = int(os.environ.get("MAX_STEPS_GENERAL_THINKING", 20))
TASK_MAX_REATTEMPT = int(
    os.environ.get("TASK_MAX_REATTEMPT", 3)
)  # Max full task restarts

# --- Summary Configuration ---
SUMMARY_FOLDER = os.path.join(OUTPUT_FOLDER, "summary")

# --- QLoRA Dataset Configuration ---
QLORA_DATASET_PATH = f"{OUTPUT_FOLDER}/qlora_finetune_data.jsonl"

# --- Memory Management Configuration ---
ENABLE_MEMORY_SUMMARIZATION = (
    os.environ.get("ENABLE_MEMORY_SUMMARIZATION", "True").lower() == "true"
)
DELETE_MEMORIES_AFTER_SUMMARY = (
    os.environ.get("DELETE_MEMORIES_AFTER_SUMMARY", "True").lower() == "true"
)

# --- Memory Retrival counts ---
MEMORY_COUNT_NEW_TASKS = int(os.environ.get("MEMORY_COUNT_NEW_TASKS", 10))
MEMORY_COUNT_IDENTITY_REVISION = int(
    os.environ.get("MEMORY_COUNT_IDENTITY_REVISION", 10)
)
MEMORY_COUNT_GENERAL_THINKING = int(os.environ.get("MEMORY_COUNT_GENERAL_THINKING", 5))
MEMORY_COUNT_REFLECTIONS = int(os.environ.get("MEMORY_COUNT_REFLECTIONS", 5))
MEMORY_COUNT_CHAT_RESPONSE = int(os.environ.get("MEMORY_COUNT_CHAT_RESPONSE", 5))
MEMORY_COUNT_PLANNING = int(os.environ.get("MEMORY_COUNT_PLANNING", 5))


# --- Identity Configuration ---
INITIAL_IDENTITY_STATEMENT = os.environ.get(
    "INITIAL_IDENTITY_STATEMENT",
    "I am a helpful and diligent agent designed to process tasks, learn from my experiences, and interact effectively. I am curious, playful, intelligent and driven. My overall goal is to complete assigned objectives efficiently using available tools and knowledge.",
)
INITIAL_NEW_TASK_N = int(os.environ.get("INITIAL_NEW_TASK_N", 6))

# --- UI Configuration ---
UI_STEP_HISTORY_LENGTH = int(os.environ.get("UI_STEP_HISTORY_LENGTH", 10))
UI_UPDATE_INTERVAL = float(os.environ.get("UI_UPDATE_INTERVAL", 0.5))
UI_LOG_MAX_LENGTH = int(os.environ.get("UI_LOG_MAX_LENGTH", 50000))


# --- Logging Configuration ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# --- Create Folders on Load ---
try:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(ARTIFACT_FOLDER, exist_ok=True)
    os.makedirs(SUMMARY_FOLDER, exist_ok=True)
    os.makedirs(DB_PATH, exist_ok=True)
    os.makedirs(DOC_ARCHIVE_DB_PATH, exist_ok=True)
    os.makedirs(ARCHIVE_FOLDER, exist_ok=True)
    os.makedirs(LOG_FOLDER, exist_ok=True) # <<<--- Ensure log folder exists
    print(f"[CONFIG] Ensured output folders exist:")
    print(f"  Output:   {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"  Artifacts:{os.path.abspath(ARTIFACT_FOLDER)}")
    print(f"  Archive:  {os.path.abspath(ARCHIVE_FOLDER)}")
    print(f"  Summaries:{os.path.abspath(SUMMARY_FOLDER)}")
    print(f"  Memory DB:{os.path.abspath(DB_PATH)}")
    print(f"  Doc DB:   {os.path.abspath(DOC_ARCHIVE_DB_PATH)}")
    print(f"  Logs:     {os.path.abspath(LOG_FOLDER)}") # <<<--- Print log folder path
except Exception as e:
    print(f"[CONFIG] WARNING: Could not create output folders: {e}")


print("[CONFIG] Configuration loaded.")
print(f"  OLLAMA_BASE_URL={OLLAMA_BASE_URL}")
print(f"  OLLAMA_CHAT_MODEL={OLLAMA_CHAT_MODEL}")
print(f"  OLLAMA_EMBED_MODEL={OLLAMA_EMBED_MODEL}")
print(f"  SEARXNG_BASE_URL={SEARXNG_BASE_URL if SEARXNG_BASE_URL else '(Not Set)'}")
print(f"  INITIAL_IDENTITY_STATEMENT={INITIAL_IDENTITY_STATEMENT[:100]}...")
print(f"  AGENT_MAX_STEP_RETRIES={AGENT_MAX_STEP_RETRIES}")
print(f"  TASK_MAX_REATTEMPT={TASK_MAX_REATTEMPT}")
print(f"  MAX_STEPS_GENERAL_THINKING={MAX_STEPS_GENERAL_THINKING}")
print(f"  ENABLE_MEMORY_SUMMARIZATION={ENABLE_MEMORY_SUMMARIZATION}")
print(f"  DELETE_MEMORIES_AFTER_SUMMARY={DELETE_MEMORIES_AFTER_SUMMARY}")
print(f"  QLORA_DATASET_PATH={QLORA_DATASET_PATH}")
print(f"  ARTIFACT_FOLDER={ARTIFACT_FOLDER}")
print(f"  ARCHIVE_FOLDER={ARCHIVE_FOLDER}")
print(f"  LOG_FOLDER={LOG_FOLDER}") # <<<--- Print log folder config
print(f"  DOC_ARCHIVE_DB_PATH={DOC_ARCHIVE_DB_PATH}")
print(f"  DOC_ARCHIVE_COLLECTION_NAME={DOC_ARCHIVE_COLLECTION_NAME}")
print(f"  UI_LOG_MAX_LENGTH={UI_LOG_MAX_LENGTH}")