# autonomous_agent/memory.py
import uuid
import datetime
import json
import traceback
from typing import List, Dict, Any, Optional
from collections import Counter # <<<--- ADD IMPORT

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
# ... (setup_chromadb unchanged) ...
def setup_chromadb() -> Optional[chromadb.Collection]:
    """Initializes ChromaDB client and collection."""
    log.info(f"Initializing ChromaDB client at path: {DB_PATH}")
    try:
        settings = chromadb.Settings(anonymized_telemetry=False)
        vector_db = chromadb.PersistentClient(path=DB_PATH, settings=settings)
        log.info(f"Getting or creating ChromaDB collection '{MEMORY_COLLECTION_NAME}'")
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
        log.critical(f"Could not initialize ChromaDB at {DB_PATH}.", exc_info=True)
        return None

# --- Agent Memory Class ---
class AgentMemory:
    """Manages interaction with the ChromaDB vector store for agent memories."""
    def __init__(self, collection: chromadb.Collection):
        if collection is None:
             raise ValueError("ChromaDB collection cannot be None for AgentMemory.")
        self.collection = collection
        try:
             self.ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
        except Exception as e:
             log.error(f"Failed to initialize Ollama client for pre-checks: {e}")
             self.ollama_client = None
        self.ollama_embed_model = OLLAMA_EMBED_MODEL

    def _test_embedding_generation(self, text_content: str) -> bool:
        # ... (implementation unchanged) ...
        if not self.ollama_client: log.warning("_test_embedding_generation: Ollama client not initialized, skipping pre-check."); return True
        if not text_content: log.warning("_test_embedding_generation: Empty content passed."); return False
        try:
            response = self.ollama_client.embeddings(model=self.ollama_embed_model, prompt=text_content)
            if response and isinstance(response.get('embedding'), list) and len(response['embedding']) > 0: return True
            else: log.error(f"Ollama client returned empty/invalid embedding. Response: {response}"); return False
        except Exception as e:
            log.error(f"Failed embedding test call to Ollama. Error: {e}")
            log.error(f"  Content Snippet: {text_content[:100]}...")
            return False

    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        # ... (implementation unchanged) ...
        if not content or not isinstance(content, str) or not content.strip(): log.warning("Skipped adding empty memory content."); return None
        if not self._test_embedding_generation(content): log.error(f"Embedding pre-check failed. Skipping add to ChromaDB for content: {content[:100]}..."); return None
        memory_id = str(uuid.uuid4()); metadata = metadata if isinstance(metadata, dict) else {}; metadata["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            cleaned_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)): cleaned_metadata[k] = v
                elif isinstance(v, list) and all(isinstance(item, (str, int, float, bool)) for item in v): cleaned_metadata[k] = json.dumps(v)
                else: log.debug(f"Converting metadata value '{k}' of type {type(v)} to string."); cleaned_metadata[k] = str(v)

            self.collection.add(documents=[content], metadatas=[cleaned_metadata], ids=[memory_id])
            log.info(f"Memory added: ID {memory_id} (Type: {metadata.get('type', 'N/A')})")
            return memory_id
        except Exception as e: log.error(f"Failed during collection.add operation. Error: {e}"); log.error(f"  Content: {content[:100]}..."); log.exception("Add Memory Traceback:"); return None

    def retrieve_raw_candidates(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        # ... (implementation unchanged) ...
        if not query or not isinstance(query, str) or not query.strip() or n_results <= 0: return []
        try:
            collection_count = self.collection.count(); actual_n_results = min(n_results, collection_count) if collection_count > 0 else 0
            if actual_n_results == 0: return []
            results = self.collection.query(query_texts=[query], n_results=actual_n_results, include=['metadatas', 'documents', 'distances'])
            memories = []
            if results and all(k in results for k in ['ids', 'documents', 'metadatas', 'distances']) and \
               isinstance(results.get('ids'), list) and len(results['ids']) > 0 and len(results['ids'][0]) == len(results['documents'][0]) == len(results['metadatas'][0]) == len(results['distances'][0]):
                for i, doc_id in enumerate(results["ids"][0]):
                     doc = results["documents"][0][i]; meta = results["metadatas"][0][i]; dist = results["distances"][0][i]
                     if doc is not None and isinstance(meta, dict) and isinstance(dist, (float, int)):
                         memories.append({"id": doc_id, "content": doc, "metadata": meta, "distance": dist})
            else: log.warning(f"Unexpected structure in ChromaDB query results: {results}")
            return memories
        except Exception as e: log.exception(f"Error retrieving raw memories from ChromaDB: {e}"); return []

    def get_memories_by_metadata(self, filter_dict: Dict[str, Any], limit: Optional[int] = None, include_vectors: bool = False) -> List[Dict[str, Any]]:
        """Retrieves memories matching specific metadata filters, optionally limited."""
        # ... (filtering logic remains largely the same, add limit) ...
        if not filter_dict: log.warning("get_memories_by_metadata called with empty filter."); return []
        log.debug(f"Getting memories with metadata filter: {filter_dict}{f', limit: {limit}' if limit else ''}")
        include_fields = ['metadatas', 'documents']
        if include_vectors: include_fields.append('embeddings')
        try:
            cleaned_filter = {};
            for k, v in filter_dict.items():
                if isinstance(v, (str, int, float, bool)): cleaned_filter[k] = v
                else: log.warning(f"Metadata filter value for '{k}' is not a simple type ({type(v)}). Skipping key.")
            if not cleaned_filter: log.error("Metadata filter became empty after cleaning. Cannot query."); return []

            get_args = {"where": cleaned_filter, "include": include_fields}
            if limit is not None and isinstance(limit, int) and limit > 0:
                get_args["limit"] = limit

            results = self.collection.get(**get_args) # Use dictionary unpacking

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
                    if include_vectors and results.get('embeddings'): mem_data["embedding"] = results['embeddings'][i]
                    memories.append(mem_data)
            else: log.debug("No memories found matching the metadata filter.")
            # Sort by timestamp descending if available in metadata
            try: memories.sort(key=lambda m: m.get('metadata', {}).get('timestamp', '0'), reverse=True)
            except: pass # Ignore sorting errors
            return memories
        except Exception as e: log.exception(f"Error getting memories by metadata ({filter_dict}): {e}"); return []

    def delete_memories(self, memory_ids: List[str]) -> bool:
        # ... (implementation unchanged) ...
        if not memory_ids: log.warning("delete_memories called with empty ID list."); return False
        log.info(f"Attempting to delete {len(memory_ids)} memories...")
        try:
            ids_to_delete = [str(mem_id) for mem_id in memory_ids if mem_id]
            if not ids_to_delete: log.warning("No valid string IDs found to delete."); return False
            self.collection.delete(ids=ids_to_delete)
            log.info(f"Successfully deleted {len(ids_to_delete)} memories."); return True
        except Exception as e: log.exception(f"Error deleting memories from ChromaDB: {e}"); return False

    # --- NEW METHOD ---
    def get_memory_summary(self) -> Dict[str, int]:
        """Returns a count of memories grouped by their 'type' metadata."""
        log.debug("Getting memory summary by type...")
        summary = Counter()
        try:
            # Fetch only metadata for all items (more efficient than getting all docs)
            # Note: collection.get() without IDs/where might be inefficient for huge DBs
            # Consider sampling if performance becomes an issue.
            all_memories = self.collection.get(include=['metadatas'])
            if all_memories and all_memories.get('metadatas'):
                for metadata in all_memories['metadatas']:
                    mem_type = metadata.get('type', 'unknown') if isinstance(metadata, dict) else 'malformed_meta'
                    summary[mem_type] += 1
                log.info(f"Memory summary generated: {len(summary)} types found.")
            else:
                log.warning("Could not retrieve metadata for memory summary.")
        except Exception as e:
            log.exception(f"Error generating memory summary: {e}")
        return dict(summary)

    # --- NEW METHOD ---
    def get_general_memories(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent memories not directly tied to a task ID (e.g., reflections, chat)."""
        log.debug(f"Getting {limit} general memories...")
        # This is tricky with current filtering. Option 1: Get all recent and filter in Python.
        # Option 2: Try a ChromaDB filter if supported (`task_id` does not exist).
        # Let's try Option 1 for simplicity first. Fetch more and filter.
        fetch_limit = limit * 3 # Fetch more to increase chance of finding general ones
        try:
            # Get recent memories regardless of metadata
            results = self.collection.get(limit=fetch_limit, include=['metadatas', 'documents'])
            general_memories = []
            if results and results.get('ids'):
                num_results = len(results['ids'])
                for i in range(num_results):
                    metadata = results['metadatas'][i] if results.get('metadatas') else None
                    # Check if 'task_id' metadata key exists
                    if isinstance(metadata, dict) and 'task_id' not in metadata:
                         general_memories.append({
                             "id": results['ids'][i],
                             "content": results['documents'][i] if results.get('documents') else None,
                             "metadata": metadata,
                         })
                         if len(general_memories) >= limit:
                             break # Stop once we have enough general memories

                # Sort by timestamp descending
                try: general_memories.sort(key=lambda m: m.get('metadata', {}).get('timestamp', '0'), reverse=True)
                except: pass
                log.info(f"Found {len(general_memories)} general memories (limit {limit}).")
                return general_memories
            else:
                log.debug("No general memories found.")
                return []
        except Exception as e:
            log.exception(f"Error retrieving general memories: {e}")
            return []