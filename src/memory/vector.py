"""
Vector memory module using ChromaDB for long-term recall (RAG).
"""

import logging
import time
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
            
        except Exception as e:
            logger.error(f"Failed to initialize vector memory: {e}")
            self.enabled = False

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
            
            self.collection.add(
                documents=[text],
                metadatas=[clean_metadata],
                ids=[f"mem_{time.time_ns()}"]
            )
            logger.debug(f"Added memory: {text[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")

    def retrieve_relevant(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for a query."""
        if not self.enabled or not self.collection:
            return []
            
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
