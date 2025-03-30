# autonomous_agent/memory.py
import uuid
import datetime
import json
import traceback
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import ollama # Direct client for pre-checks

# Import necessary config values
from config import (
    OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, DB_PATH, MEMORY_COLLECTION_NAME
)

# Suggestion: Use logging module
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s][MEMORY] %(message)s')
log = logging.getLogger(__name__)


# --- ChromaDB Setup ---
# Moved setup into a function to handle potential errors gracefully at startup
def setup_chromadb() -> Optional[chromadb.Collection]:
    """Initializes ChromaDB client and collection."""
    log.info(f"Initializing ChromaDB client at path: {DB_PATH}")
    try:
        vector_db = chromadb.PersistentClient(path=DB_PATH)
        log.info(f"Getting or creating ChromaDB collection '{MEMORY_COLLECTION_NAME}'")

        # Initialize embedding function (error if Ollama URL/model invalid from Chroma's perspective)
        embedding_function = OllamaEmbeddingFunction(
            url=f"{OLLAMA_BASE_URL}/api/embeddings",
            model_name=OLLAMA_EMBED_MODEL
        )

        memory_collection = vector_db.get_or_create_collection(
            name=MEMORY_COLLECTION_NAME,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        log.info("ChromaDB collection ready.")
        return memory_collection
    except Exception as e:
        log.critical(f"Could not initialize ChromaDB at {DB_PATH}.")
        log.critical(f"  Potential Issues: Ollama server/model, path permissions, DB errors.")
        log.critical(f"  Error details: {e}", exc_info=True) # Log stack trace
        return None

# --- Agent Memory Class ---
class AgentMemory:
    """Manages interaction with the ChromaDB vector store for agent memories."""
    def __init__(self, collection: chromadb.Collection):
        if collection is None:
             raise ValueError("ChromaDB collection cannot be None for AgentMemory.")
        self.collection = collection
        # Use Ollama client *only* for pre-checks
        try:
             self.ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
        except Exception as e:
             log.error(f"Failed to initialize Ollama client for pre-checks: {e}")
             self.ollama_client = None # Allow continuation but pre-check will fail
        self.ollama_embed_model = OLLAMA_EMBED_MODEL

    def _test_embedding_generation(self, text_content: str) -> bool:
        """Attempts to generate an embedding via Ollama client to pre-check."""
        if not self.ollama_client:
             log.warning("_test_embedding_generation: Ollama client not initialized, skipping pre-check.")
             return True # Assume okay if client failed init, let Chroma handle it
        if not text_content: log.warning("_test_embedding_generation: Empty content passed."); return False
        try:
            # log.debug(f"Testing embedding for: {text_content[:60]}...") # Very Verbose
            response = self.ollama_client.embeddings(model=self.ollama_embed_model, prompt=text_content)
            if response and isinstance(response.get('embedding'), list) and len(response['embedding']) > 0: return True
            else: log.error(f"Ollama client returned empty/invalid embedding. Response: {response}"); return False
        except Exception as e:
            log.error(f"Failed embedding test call to Ollama. Error: {e}")
            log.error(f"  Content Snippet: {text_content[:100]}...")
            log.error(f"  Troubleshoot: Check Ollama server at {OLLAMA_BASE_URL}, model '{self.ollama_embed_model}' availability, content length.")
            return False

    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Adds content to ChromaDB after pre-checking embedding generation."""
        if not content or not isinstance(content, str) or not content.strip(): log.warning("Skipped adding empty memory content."); return None
        if not self._test_embedding_generation(content): log.error(f"Embedding pre-check failed. Skipping add to ChromaDB for content: {content[:100]}..."); return None

        memory_id = str(uuid.uuid4()); metadata = metadata if isinstance(metadata, dict) else {}; metadata["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            # Ensure metadata values are simple types for ChromaDB
            cleaned_metadata = {k: (str(v) if not isinstance(v, (str, int, float, bool)) else v) for k, v in metadata.items()}
            # Let ChromaDB handle the actual embedding using the configured function
            self.collection.add(documents=[content], metadatas=[cleaned_metadata], ids=[memory_id])
            log.info(f"Memory added: ID {memory_id} (Type: {metadata.get('type', 'N/A')})")
            return memory_id
        except Exception as e:
            log.error(f"Failed during collection.add operation. Error: {e}"); log.error(f"  Content: {content[:100]}..."); log.exception("Add Memory Traceback:") # Log stack trace
            return None

    def retrieve_raw_candidates(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Retrieves raw candidate memories based on vector similarity."""
        if not query or not isinstance(query, str) or not query.strip() or n_results <= 0: return []
        # log.debug(f"Retrieving {n_results} candidates for query: '{query[:50]}...'") # Verbose
        try:
            # Ensure n_results doesn't exceed the number of items in the collection
            collection_count = self.collection.count()
            actual_n_results = min(n_results, collection_count) if collection_count > 0 else 0

            if actual_n_results == 0:
                 # log.debug("Collection is empty, returning no candidates.") # Verbose
                 return []

            results = self.collection.query(query_texts=[query], n_results=actual_n_results, include=['metadatas', 'documents', 'distances'])
            memories = []
            # Robust check of ChromaDB response structure
            if results and isinstance(results.get("ids"), list) and results["ids"] and \
               isinstance(results.get("documents"), list) and results.get("documents") is not None and \
               isinstance(results.get("metadatas"), list) and results.get("metadatas") is not None and \
               isinstance(results.get("distances"), list) and results.get("distances") is not None and \
               len(results["ids"][0]) == len(results["documents"][0]) == len(results["metadatas"][0]) == len(results["distances"][0]):
                 for i, doc_id in enumerate(results["ids"][0]):
                     doc = results["documents"][0][i]; meta = results["metadatas"][0][i]; dist = results["distances"][0][i]
                     if doc is not None and isinstance(meta, dict) and isinstance(dist, (float, int)):
                         memories.append({"id": doc_id, "content": doc, "metadata": meta, "distance": dist})
            # log.debug(f"Retrieved {len(memories)} raw candidates.") # Verbose
            return memories
        except Exception as e: log.exception(f"Error retrieving raw memories from ChromaDB: {e}"); return [] # Log stack trace