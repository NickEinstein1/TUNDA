"""
Vector memory module using ChromaDB for long-term recall (RAG).
"""

import logging
import time
import threading
from typing import List, Dict, Any, Optional
import os

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None

from ..utils.config import config

logger = logging.getLogger(__name__)

class VectorMemory:
    """Manages long-term memory using vector embeddings."""
    
    def __init__(self, persistence_path: str = "data/memory_db"):
        self.enabled = config.memory.enabled
        self.collection = None
        self.pending = []
        self.pending_lock = threading.Lock()
        self.batch_embeddings = config.memory.batch_embeddings
        self.batch_size = config.memory.batch_size
        self.batch_flush_seconds = config.memory.batch_flush_seconds
        self.max_items = config.memory.max_items
        self.prune_to = config.memory.prune_to
        self.prune_strategy = config.memory.prune_strategy
        self.importance_threshold = config.memory.importance_threshold
        
        if not self.enabled:
            logger.info("Vector memory disabled in config")
            return
            
        if chromadb is None:
            logger.warning("ChromaDB not installed. Vector memory disabled.")
            self.enabled = False
            return
            
        try:
            # Initialize client
            self.client = chromadb.PersistentClient(path=persistence_path)
            
            # Use a lightweight embedding model
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="conversation_history",
                embedding_function=self.embedding_fn
            )
            logger.info(f"Vector memory initialized at {persistence_path}")
            if self.batch_embeddings:
                self._start_flush_thread()
            
        except Exception as e:
            logger.error(f"Failed to initialize vector memory: {e}")
            self.enabled = False
    
    def _start_flush_thread(self):
        thread = threading.Thread(target=self._flush_loop, daemon=True)
        thread.start()

    def _flush_loop(self):
        while True:
            time.sleep(self.batch_flush_seconds)
            self.flush_pending()

    def add_memory(self, text: str, metadata: Dict[str, Any] = None):
        """Add a memory to the vector database."""
        if not self.enabled or not self.collection:
            return
            
        try:
            # Ensure metadata values are strings, ints, floats, or bools
            clean_metadata = {}
            if metadata:
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_metadata[k] = v
                    else:
                        clean_metadata[k] = str(v)
            
            # Add timestamp
            clean_metadata["timestamp"] = time.time()
            importance = self._score_memory(text, clean_metadata)
            if importance < self.importance_threshold:
                return
            clean_metadata["importance"] = importance
            clean_metadata["topic"] = self._extract_topic(text)
            
            if self.batch_embeddings:
                with self.pending_lock:
                    self.pending.append((text, clean_metadata, f"mem_{time.time_ns()}"))
                    if len(self.pending) >= self.batch_size:
                        self.flush_pending()
            else:
                self.collection.add(
                    documents=[text],
                    metadatas=[clean_metadata],
                    ids=[f"mem_{time.time_ns()}"]
                )
                logger.debug(f"Added memory: {text[:50]}...")
                self._prune_if_needed()
            
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")

    def flush_pending(self):
        if not self.pending:
            return
        with self.pending_lock:
            batch = self.pending[:]
            self.pending = []
        try:
            docs = [item[0] for item in batch]
            metas = [item[1] for item in batch]
            ids = [item[2] for item in batch]
            self.collection.add(documents=docs, metadatas=metas, ids=ids)
            logger.debug(f"Added {len(docs)} memories in batch")
            self._prune_if_needed()
        except Exception as e:
            logger.error(f"Failed to flush memory batch: {e}")

    def retrieve_relevant(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for a query."""
        if not self.enabled or not self.collection:
            return []
    
    def _extract_topic(self, text: str) -> str:
        tokens = [t.strip(".,!?;:").lower() for t in text.split()]
        stopwords = {
            "the", "and", "a", "to", "of", "in", "is", "it", "for", "on", "with", "that",
            "this", "i", "you", "we", "they", "me", "my", "your", "our", "at"
        }
        keywords = [t for t in tokens if t and t not in stopwords]
        return keywords[0] if keywords else "general"

    def _score_memory(self, text: str, metadata: Dict[str, Any]) -> float:
        score = 0.1
        if metadata.get("emotion") in {"sad", "angry", "anxious"}:
            score += 0.3
        if "name" in text.lower():
            score += 0.2
        score += min(0.4, len(text) / 400.0)
        return min(1.0, score)

    def _prune_if_needed(self):
        if self.max_items <= 0:
            return
        try:
            count = self.collection.count()
            if count <= self.max_items:
                return
            target = max(0, self.prune_to)
            results = self.collection.get(include=["metadatas", "documents", "ids"])
            ids = results.get("ids", [])
            metas = results.get("metadatas", [])
            if not ids:
                return
            scored = []
            for mem_id, meta in zip(ids, metas):
                importance = 0.0
                if isinstance(meta, dict):
                    importance = float(meta.get("importance", 0.0))
                scored.append((importance, mem_id, meta))
            scored.sort(key=lambda x: x[0], reverse=True)

            if self.prune_strategy == "per_topic":
                keep_ids = []
                topic_counts = {}
                per_topic_limit = max(1, target // max(1, len(scored)))
                for importance, mem_id, meta in scored:
                    topic = (meta or {}).get("topic", "general")
                    if topic_counts.get(topic, 0) < per_topic_limit:
                        keep_ids.append(mem_id)
                        topic_counts[topic] = topic_counts.get(topic, 0) + 1
            else:
                keep_ids = [mem_id for _, mem_id, _ in scored[:target]]

            delete_ids = [mem_id for mem_id in ids if mem_id not in set(keep_ids)]
            if delete_ids:
                self.collection.delete(ids=delete_ids)
                logger.info(f"Pruned {len(delete_ids)} memories")
        except Exception as e:
            logger.error(f"Memory pruning failed: {e}")
            
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            memories = []
            if results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    memories.append({
                        "text": doc,
                        "metadata": meta,
                        "distance": results["distances"][0][i] if results["distances"] else 0.0
                    })
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []

    def clear(self):
        """Clear all memories."""
        if self.collection:
            try:
                self.client.delete_collection("conversation_history")
            except:
                pass
