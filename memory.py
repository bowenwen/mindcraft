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
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s][MEMORY] %(message)s')
log = logging.getLogger(__name__)


# --- ChromaDB Setup ---
def setup_chromadb() -> Optional[chromadb.Collection]:
    """Initializes ChromaDB client and collection."""
    log.info(f"Initializing ChromaDB client at path: {DB_PATH}")
    try:
        # Explicitly adding setting to potentially mitigate issues on some systems
        # See: https://docs.trychroma.com/troubleshooting#sqlite
        settings = chromadb.Settings(anonymized_telemetry=False) # Disable telemetry
        # settings = chromadb.Settings(allow_reset=True) # Use allow_reset carefully if needed for debugging resets
        vector_db = chromadb.PersistentClient(path=DB_PATH, settings=settings)

        log.info(f"Getting or creating ChromaDB collection '{MEMORY_COLLECTION_NAME}'")

        # Initialize embedding function
        embedding_function = OllamaEmbeddingFunction(
            url=f"{OLLAMA_BASE_URL}/api/embeddings",
            model_name=OLLAMA_EMBED_MODEL
        )

        memory_collection = vector_db.get_or_create_collection(
            name=MEMORY_COLLECTION_NAME,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"} # Cosine similarity is often good for embeddings
        )
        log.info("ChromaDB collection ready.")
        return memory_collection
    except Exception as e:
        log.critical(f"Could not initialize ChromaDB at {DB_PATH}.")
        log.critical(f"  Potential Issues: Ollama server/model, path permissions, DB state, telemetry conflicts.")
        log.critical(f"  Ensure '.env' includes 'ANONYMIZED_TELEMETRY=False'")
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

    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Adds content to ChromaDB after pre-checking embedding generation."""
        if not content or not isinstance(content, str) or not content.strip(): log.warning("Skipped adding empty memory content."); return None
        if not self._test_embedding_generation(content): log.error(f"Embedding pre-check failed. Skipping add to ChromaDB for content: {content[:100]}..."); return None

        memory_id = str(uuid.uuid4()); metadata = metadata if isinstance(metadata, dict) else {}; metadata["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            # Ensure metadata values are simple types compatible with ChromaDB
            cleaned_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    cleaned_metadata[k] = v
                elif isinstance(v, list) and all(isinstance(item, (str, int, float, bool)) for item in v):
                    cleaned_metadata[k] = json.dumps(v) # Store lists as JSON strings if needed by filter
                else:
                    # Convert other types to string, log warning
                    log.debug(f"Converting metadata value '{k}' of type {type(v)} to string.")
                    cleaned_metadata[k] = str(v)

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
            collection_count = self.collection.count()
            actual_n_results = min(n_results, collection_count) if collection_count > 0 else 0
            if actual_n_results == 0: return []

            results = self.collection.query(query_texts=[query], n_results=actual_n_results, include=['metadatas', 'documents', 'distances'])
            memories = []
            # Check response structure defensively
            if results and all(k in results for k in ['ids', 'documents', 'metadatas', 'distances']) and \
               isinstance(results['ids'], list) and len(results['ids']) > 0 and \
               isinstance(results['documents'], list) and len(results['documents']) > 0 and \
               isinstance(results['metadatas'], list) and len(results['metadatas']) > 0 and \
               isinstance(results['distances'], list) and len(results['distances']) > 0 and \
               len(results['ids'][0]) == len(results['documents'][0]) == len(results['metadatas'][0]) == len(results['distances'][0]):

                for i, doc_id in enumerate(results["ids"][0]):
                     doc = results["documents"][0][i]; meta = results["metadatas"][0][i]; dist = results["distances"][0][i]
                     if doc is not None and isinstance(meta, dict) and isinstance(dist, (float, int)):
                         memories.append({"id": doc_id, "content": doc, "metadata": meta, "distance": dist})
            else:
                 log.warning(f"Unexpected structure in ChromaDB query results: {results}")

            # log.debug(f"Retrieved {len(memories)} raw candidates.") # Verbose
            return memories
        except Exception as e: log.exception(f"Error retrieving raw memories from ChromaDB: {e}"); return [] # Log stack trace

    def get_memories_by_metadata(self, filter_dict: Dict[str, Any], include_vectors: bool = False) -> List[Dict[str, Any]]:
        """Retrieves memories matching specific metadata filters."""
        if not filter_dict: log.warning("get_memories_by_metadata called with empty filter."); return []
        log.debug(f"Getting memories with metadata filter: {filter_dict}")
        include_fields = ['metadatas', 'documents']
        if include_vectors: include_fields.append('embeddings')

        try:
            # Ensure filter values are basic types Chroma expects
            cleaned_filter = {}
            for k, v in filter_dict.items():
                if isinstance(v, (str, int, float, bool)):
                    cleaned_filter[k] = v
                else:
                    log.warning(f"Metadata filter value for '{k}' is not a simple type ({type(v)}). Skipping key.")

            if not cleaned_filter:
                 log.error("Metadata filter became empty after cleaning. Cannot query.")
                 return []

            # Using collection.get with a 'where' clause
            results = self.collection.get(
                where=cleaned_filter,
                include=include_fields
            )

            memories = []
            if results and results.get('ids'):
                num_results = len(results['ids'])
                log.debug(f"Found {num_results} memories matching filter.")
                for i in range(num_results):
                    mem_data = {
                        "id": results['ids'][i],
                        "content": results['documents'][i] if results.get('documents') else None,
                        "metadata": results['metadatas'][i] if results.get('metadatas') else None,
                    }
                    if include_vectors and results.get('embeddings'):
                        mem_data["embedding"] = results['embeddings'][i]
                    memories.append(mem_data)
            else:
                log.debug("No memories found matching the metadata filter.")
            return memories
        except Exception as e:
            log.exception(f"Error getting memories by metadata ({filter_dict}): {e}")
            return []

    def delete_memories(self, memory_ids: List[str]) -> bool:
        """Deletes memories from the collection by their IDs."""
        if not memory_ids: log.warning("delete_memories called with empty ID list."); return False
        log.info(f"Attempting to delete {len(memory_ids)} memories...")
        try:
            # Ensure IDs are strings
            ids_to_delete = [str(mem_id) for mem_id in memory_ids]
            if not ids_to_delete:
                log.warning("No valid string IDs found to delete.")
                return False

            self.collection.delete(ids=ids_to_delete)
            log.info(f"Successfully deleted {len(ids_to_delete)} memories.")
            return True
        except Exception as e:
            log.exception(f"Error deleting memories from ChromaDB: {e}")
            return False