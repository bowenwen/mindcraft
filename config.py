# autonomous_agent/config.py
import os
from dotenv import load_dotenv

print("[CONFIG] Loading environment variables...")
load_dotenv()

# --- Common Configuration ---
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", "./output")

# --- Ollama Configuration ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "gemma3")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", 120))

# --- SearXNG Configuration ---
SEARXNG_BASE_URL = os.environ.get("SEARXNG_BASE_URL", "http://localhost:8080")
SEARXNG_TIMEOUT = int(os.environ.get("SEARXNG_TIMEOUT", 20))

# --- Database Configuration ---
DB_PATH = os.environ.get("DB_PATH", "./chroma_db")
MEMORY_COLLECTION_NAME = os.environ.get("MEMORY_COLLECTION_NAME", "agent_memory")

# --- Task Queue Configuration ---
TASK_QUEUE_PATH = f"{OUTPUT_FOLDER}/task_queue.json"
INVESTIGATION_RES_PATH = f"{OUTPUT_FOLDER}/investigation_results.json"

# --- Agent Configuration ---
AGENT_STATE_PATH = f"{OUTPUT_FOLDER}/session_state.json"
DEFAULT_MAX_STEPS_PER_TASK = int(os.environ.get("DEFAULT_MAX_STEPS_PER_TASK", 8))
DEFAULT_SESSION_DURATION_MINUTES = int(
    os.environ.get("DEFAULT_SESSION_DURATION_MINUTES", 30)
)
CONTEXT_TRUNCATION_LIMIT = int(os.environ.get("CONTEXT_TRUNCATION_LIMIT", 40000))
MODEL_CONTEXT_LENGTH = CONTEXT_TRUNCATION_LIMIT + 10000 # Allow buffer for prompt overhead
# --- NEW: Error Handling Configuration ---
AGENT_MAX_STEP_RETRIES = int(os.environ.get("AGENT_MAX_STEP_RETRIES", 2)) # Max retries for a *single step* before failing the task

# --- Summary Configuration ---
SUMMARY_FOLDER = os.environ.get("SUMMARY_FOLDER", "./output/summary")

# --- NEW: QLoRA Dataset Configuration ---
QLORA_DATASET_PATH = f"{OUTPUT_FOLDER}/qlora_finetune_data.jsonl" # Changed extension

# --- NEW: Memory Management Configuration ---
ENABLE_MEMORY_SUMMARIZATION = os.environ.get("ENABLE_MEMORY_SUMMARIZATION", "True").lower() == "true"
DELETE_MEMORIES_AFTER_SUMMARY = os.environ.get("DELETE_MEMORIES_AFTER_SUMMARY", "True").lower() == "false"

# --- NEW: Identity Configuration ---
INITIAL_IDENTITY_STATEMENT = os.environ.get(
    "INITIAL_IDENTITY_STATEMENT",
    "I am a helpful and diligent AI assistant designed to process tasks, learn from my experiences, and interact effectively. My goal is to complete assigned objectives efficiently using available tools and knowledge."
)
IDENTITY_REVISION_MEMORY_COUNT = int(os.environ.get("IDENTITY_REVISION_MEMORY_COUNT", 15))

# --- Logging Configuration ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

print("[CONFIG] Configuration loaded.")
# (Optionally print other loaded configs)
print(f"  INITIAL_IDENTITY_STATEMENT={INITIAL_IDENTITY_STATEMENT[:100]}...") # Print snippet
print(f"  AGENT_MAX_STEP_RETRIES={AGENT_MAX_STEP_RETRIES}")
print(f"  ENABLE_MEMORY_SUMMARIZATION={ENABLE_MEMORY_SUMMARIZATION}")
print(f"  DELETE_MEMORIES_AFTER_SUMMARY={DELETE_MEMORIES_AFTER_SUMMARY}")
print(f"  QLORA_DATASET_PATH={QLORA_DATASET_PATH}")