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
MODEL_CONTEXT_LENGTH = CONTEXT_TRUNCATION_LIMIT + 10000

# --- Summary Configuration ---
SUMMARY_FOLDER = os.environ.get("SUMMARY_FOLDER", "./output/summary")

# --- NEW: QLoRA Dataset Configuration ---
QLORA_DATASET_PATH = f"{OUTPUT_FOLDER}/qlora_finetune_data.json"

# --- Logging Configuration ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

print("[CONFIG] Configuration loaded.")
# (Optionally print other loaded configs)
print(f"  QLORA_DATASET_PATH={QLORA_DATASET_PATH}")
