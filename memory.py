# FILE: memory.py
# autonomous_agent/memory.py
import uuid
import datetime
import json
import traceback
import re  # <<<--- ADD IMPORT
from typing import List, Dict, Any, Optional, Tuple  # <<<--- ADD Tuple
from collections import Counter

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import ollama  # Direct client for pre-checks

# Import necessary config values
from config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, DB_PATH, MEMORY_COLLECTION_NAME

# Import utilities needed for moved logic
from utils import call_ollama_api, format_relative_time  # <<<--- ADD IMPORTS
import prompts  # <<<--- IMPORT PROMPTS MODULE

# Suggestion: Use logging module
import logging

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s][%(levelname)s][MEMORY] %(message)s"
)
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
            url=f"{OLLAMA_BASE_URL}/api/embeddings", model_name=OLLAMA_EMBED_MODEL
        )
        memory_collection = vector_db.get_or_create_collection(
            name=MEMORY_COLLECTION_NAME,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("ChromaDB collection ready.")
        return memory_collection
    except Exception as e:
        log.critical(f"Could not initialize ChromaDB at {DB_PATH}.", exc_info=True)
        return None


# --- Agent Memory Class ---
class AgentMemory:
    """Manages interaction with the ChromaDB vector store for agent memories."""

    def __init__(
        self,
        collection: chromadb.Collection,
        ollama_chat_model: str,
        ollama_base_url: str,
    ):  # <<<--- ADD OLLAMA PARAMS
        if collection is None:
            raise ValueError("ChromaDB collection cannot be None for AgentMemory.")
        self.collection = collection
        self.ollama_chat_model = ollama_chat_model  # <<<--- STORE OLLAMA CHAT MODEL
        self.ollama_base_url = ollama_base_url  # <<<--- STORE OLLAMA BASE URL
        try:
            self.ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
        except Exception as e:
            log.error(f"Failed to initialize Ollama client for pre-checks: {e}")
            self.ollama_client = None
        self.ollama_embed_model = OLLAMA_EMBED_MODEL

    def _test_embedding_generation(self, text_content: str) -> bool:
        # ... (implementation unchanged) ...
        if not self.ollama_client:
            log.warning(
                "_test_embedding_generation: Ollama client not initialized, skipping pre-check."
            )
            return True
        if not text_content:
            log.warning("_test_embedding_generation: Empty content passed.")
            return False
        try:
            response = self.ollama_client.embeddings(
                model=self.ollama_embed_model, prompt=text_content
            )
            if (
                response
                and isinstance(response.get("embedding"), list)
                and len(response["embedding"]) > 0
            ):
                return True
            else:
                log.error(
                    f"Ollama client returned empty/invalid embedding. Response: {response}"
                )
                return False
        except Exception as e:
            log.error(f"Failed embedding test call to Ollama. Error: {e}")
            log.error(f"  Content Snippet: {text_content[:100]}...")
            return False

    def add_memory(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        # ... (implementation unchanged) ...
        if not content or not isinstance(content, str) or not content.strip():
            log.warning("Skipped adding empty memory content.")
            return None
        # Embedding pre-check happens here using self.ollama_client and self.ollama_embed_model
        if not self._test_embedding_generation(content):
            log.error(
                f"Embedding pre-check failed. Skipping add to ChromaDB for content: {content[:100]}..."
            )
            return None
        memory_id = str(uuid.uuid4())
        metadata = metadata if isinstance(metadata, dict) else {}
        metadata["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            cleaned_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    cleaned_metadata[k] = v
                elif isinstance(v, list) and all(
                    isinstance(item, (str, int, float, bool)) for item in v
                ):
                    cleaned_metadata[k] = json.dumps(v)
                else:
                    log.debug(
                        f"Converting metadata value '{k}' of type {type(v)} to string."
                    )
                    cleaned_metadata[k] = str(v)

            self.collection.add(
                documents=[content], metadatas=[cleaned_metadata], ids=[memory_id]
            )
            log.info(
                f"Memory added: ID {memory_id} (Type: {metadata.get('type', 'N/A')})"
            )
            return memory_id
        except Exception as e:
            log.error(f"Failed during collection.add operation. Error: {e}")
            log.error(f"  Content: {content[:100]}...")
            log.exception("Add Memory Traceback:")
            return None

    def retrieve_raw_candidates(
        self, query: str, n_results: int = 10
    ) -> List[Dict[str, Any]]:
        # ... (implementation unchanged) ...
        if (
            not query
            or not isinstance(query, str)
            or not query.strip()
            or n_results <= 0
        ):
            return []
        try:
            collection_count = self.collection.count()
            actual_n_results = (
                min(n_results, collection_count) if collection_count > 0 else 0
            )
            if actual_n_results == 0:
                log.debug("Memory collection is empty, cannot retrieve candidates.")
                return []  # Added log
            results = self.collection.query(
                query_texts=[query],
                n_results=actual_n_results,
                include=["metadatas", "documents", "distances"],
            )
            memories = []
            if (
                results
                and all(
                    k in results for k in ["ids", "documents", "metadatas", "distances"]
                )
                and isinstance(results.get("ids"), list)
                and len(results["ids"]) > 0
                and len(results["ids"][0])
                == len(results["documents"][0])
                == len(results["metadatas"][0])
                == len(results["distances"][0])
            ):
                for i, doc_id in enumerate(results["ids"][0]):
                    doc = results["documents"][0][i]
                    meta = results["metadatas"][0][i]
                    dist = results["distances"][0][i]
                    if (
                        doc is not None
                        and isinstance(meta, dict)
                        and isinstance(dist, (float, int))
                    ):
                        memories.append(
                            {
                                "id": doc_id,
                                "content": doc,
                                "metadata": meta,
                                "distance": dist,
                            }
                        )
            else:
                log.warning(
                    f"Unexpected structure in ChromaDB query results: {results}"
                )
            return memories
        except Exception as e:
            log.exception(f"Error retrieving raw memories from ChromaDB: {e}")
            return []

    # --- UPDATED to use prompt from prompts.py ---
    def retrieve_and_rerank_memories(
        self,
        query: str,
        task_description: str,
        context: str,
        identity_statement: str,  # <<<--- ADDED identity_statement PARAM
        n_results: int = 10,
        n_final: int = 4,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Retrieves candidate memories, separates user suggestions, and asks LLM
        to re-rank remaining candidates based on relevance, recency, and novelty.
        Returns the re-ranked memories and any separated user suggestion content.
        """
        user_suggestion_content: Optional[str] = None
        if not query or not isinstance(query, str) or not query.strip():
            return [], user_suggestion_content
        log.info(f"Retrieving & re-ranking memories. Query: '{query[:50]}...'")

        # Use self.retrieve_raw_candidates
        candidates = self.retrieve_raw_candidates(query, n_results)
        if not candidates:
            log.info("No initial memory candidates found for re-ranking.")
            return [], user_suggestion_content

        # --- Separate user suggestion memory and filter candidates ---
        filtered_candidates = []
        for mem in candidates:
            mem_type = mem.get("metadata", {}).get("type")
            if mem_type == "user_suggestion_move_on" and not user_suggestion_content:
                user_suggestion_content = mem.get("content")
                log.info("Separated 'user_suggestion_move_on' memory.")
            else:
                filtered_candidates.append(mem)

        if not filtered_candidates:
            log.info("No candidates remain after filtering suggestion.")
            return [], user_suggestion_content

        if len(filtered_candidates) <= n_final:
            log.info(
                f"Fewer candidates ({len(filtered_candidates)}) than requested ({n_final}) after filtering. Returning all (sorted by distance)."
            )
            return (
                sorted(
                    filtered_candidates, key=lambda m: m.get("distance", float("inf"))
                ),
                user_suggestion_content,
            )

        # --- Prepare prompt for LLM re-ranking ---
        candidate_details_list = []
        for idx, mem in enumerate(filtered_candidates):
            meta_info = f"Type: {mem['metadata'].get('type', 'N/A')}"
            dist_info = f"Distance: {mem.get('distance', 'N/A'):.4f}"
            # Use format_relative_time utility
            relative_time = format_relative_time(mem["metadata"].get("timestamp"))
            content_snippet = mem["content"][:200].replace("\n", " ") + "..."
            candidate_details_list.append(
                f"--- Memory Index {idx} [{relative_time}] ({dist_info}) ---\nMetadata: {meta_info}\nContent Snippet: {content_snippet}\n---"
            )

        candidate_details_str = "\n".join(candidate_details_list)

        # --- Use prompt from prompts.py ---
        rerank_prompt = prompts.RERANK_MEMORIES_PROMPT.format(
            identity_statement=identity_statement,
            task_description=task_description,
            query=query,
            context=context[-1000:],  # Limit context size for prompt
            candidate_details=candidate_details_str,
            n_final=n_final,
        )
        # ---

        # Use self.ollama_chat_model and self.ollama_base_url
        log.info(
            f"Asking {self.ollama_chat_model} to re-rank {len(filtered_candidates)} memories down to {n_final} (considering recency/novelty)..."
        )
        # Use call_ollama_api utility
        rerank_response = call_ollama_api(
            rerank_prompt, self.ollama_chat_model, self.ollama_base_url, timeout=90
        )

        fallback_memories = sorted(
            filtered_candidates, key=lambda m: m.get("distance", float("inf"))
        )[:n_final]

        if not rerank_response:
            log.warning("LLM re-ranking failed. Falling back to top N by similarity.")
            return fallback_memories, user_suggestion_content
        try:
            matches = re.findall(r"\d+", rerank_response)
            if not matches:
                log.warning(
                    f"LLM re-ranking response ('{rerank_response}') had no numbers. Falling back to top N."
                )
                return fallback_memories, user_suggestion_content
            selected_indices = [int(idx_str) for idx_str in matches]
            valid_indices = [
                idx for idx in selected_indices if 0 <= idx < len(filtered_candidates)
            ]
            valid_indices = list(dict.fromkeys(valid_indices))  # Remove duplicates

            if not valid_indices:
                log.warning(
                    f"LLM re-ranking response ('{rerank_response}') had invalid indices. Falling back to top N."
                )
                return fallback_memories, user_suggestion_content

            final_indices = valid_indices[:n_final]
            log.info(f"LLM selected memory indices for re-ranking: {final_indices}")
            reranked_memories = [filtered_candidates[i] for i in final_indices]
            return reranked_memories, user_suggestion_content

        except ValueError as e:
            log.error(
                f"Error parsing LLM re-ranking indices ('{rerank_response}'): {e}. Falling back to top N."
            )
            return fallback_memories, user_suggestion_content
        except Exception as e:
            log.exception(
                f"Unexpected error parsing LLM re-ranking response ('{rerank_response}'): {e}. Falling back to top N."
            )
            return fallback_memories, user_suggestion_content

    def get_memories_by_metadata(
        self,
        filter_dict: Dict[str, Any],
        limit: Optional[int] = None,
        include_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        # ... (implementation unchanged) ...
        if not filter_dict:
            log.warning("get_memories_by_metadata called with empty filter.")
            return []
        log.debug(
            f"Getting memories with metadata filter: {filter_dict}{f', limit: {limit}' if limit else ''}"
        )
        include_fields = ["metadatas", "documents"]
        if include_vectors:
            include_fields.append("embeddings")
        try:
            get_args = {"where": filter_dict, "include": include_fields}
            if limit is not None and isinstance(limit, int) and limit > 0:
                get_args["limit"] = limit

            # retrieve results from memory chroma db, then unpack items
            results = self.collection.get(**get_args)
            memories = []
            if results and results.get("ids"):
                num_results = len(results["ids"])
                log.debug(f"Found {num_results} memories matching filter.")
                for i in range(num_results):
                    mem_data = {
                        "id": results["ids"][i],
                        "content": (
                            results["documents"][i]
                            if results.get("documents")
                            else None
                        ),
                        "metadata": (
                            results["metadatas"][i]
                            if results.get("metadatas")
                            else None
                        ),
                    }
                    if include_vectors and results.get("embeddings"):
                        mem_data["embedding"] = results["embeddings"][i]
                    memories.append(mem_data)
            else:
                log.debug("No memories found matching the metadata filter.")
            # Sort by timestamp descending if available in metadata
            try:
                memories.sort(
                    key=lambda m: m.get("metadata", {}).get("timestamp", "0"),
                    reverse=True,
                )
            except:
                pass  # Ignore sorting errors
            return memories
        except Exception as e:
            log.exception(f"Error getting memories by metadata ({filter_dict}): {e}")
            return []

    def delete_memories(self, memory_ids: List[str]) -> bool:
        # ... (implementation unchanged) ...
        if not memory_ids:
            log.warning("delete_memories called with empty ID list.")
            return False
        log.info(f"Attempting to delete {len(memory_ids)} memories...")
        try:
            ids_to_delete = [str(mem_id) for mem_id in memory_ids if mem_id]
            if not ids_to_delete:
                log.warning("No valid string IDs found to delete.")
                return False
            self.collection.delete(ids=ids_to_delete)
            log.info(f"Successfully deleted {len(ids_to_delete)} memories.")
            return True
        except Exception as e:
            log.exception(f"Error deleting memories from ChromaDB: {e}")
            return False

    def get_memory_summary(self) -> Dict[str, int]:
        # ... (implementation unchanged) ...
        log.debug("Getting memory summary by type...")
        summary = Counter()
        try:
            all_memories = self.collection.get(include=["metadatas"])
            if all_memories and all_memories.get("metadatas"):
                for metadata in all_memories["metadatas"]:
                    mem_type = (
                        metadata.get("type", "unknown")
                        if isinstance(metadata, dict)
                        else "malformed_meta"
                    )
                    summary[mem_type] += 1
                log.info(f"Memory summary generated: {len(summary)} types found.")
            else:
                log.warning("Could not retrieve metadata for memory summary.")
        except Exception as e:
            log.exception(f"Error generating memory summary: {e}")
        return dict(summary)

    def get_general_memories(self, limit: int = 50) -> List[Dict[str, Any]]:
        # ... (implementation unchanged) ...
        log.debug(f"Getting {limit} general memories...")
        fetch_limit = limit * 3
        try:
            results = self.collection.get(
                limit=fetch_limit, include=["metadatas", "documents"]
            )
            general_memories = []
            if results and results.get("ids"):
                num_results = len(results["ids"])
                for i in range(num_results):
                    metadata = (
                        results["metadatas"][i] if results.get("metadatas") else None
                    )
                    if isinstance(metadata, dict) and "task_id" not in metadata:
                        general_memories.append(
                            {
                                "id": results["ids"][i],
                                "content": (
                                    results["documents"][i]
                                    if results.get("documents")
                                    else None
                                ),
                                "metadata": metadata,
                            }
                        )
                        if len(general_memories) >= limit:
                            break

                try:
                    general_memories.sort(
                        key=lambda m: m.get("metadata", {}).get("timestamp", "0"),
                        reverse=True,
                    )
                except:
                    pass
                log.info(
                    f"Found {len(general_memories)} general memories (limit {limit})."
                )
                return general_memories
            else:
                log.debug("No general memories found.")
                return []
        except Exception as e:
            log.exception(f"Error retrieving general memories: {e}")
            return []
