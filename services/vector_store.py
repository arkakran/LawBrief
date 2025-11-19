import numpy as np
import faiss
from typing import List, Dict, Any
from loguru import logger

from utils.embeddings import EmbeddingService


class LocalVectorStore:

    def __init__(self, m: int = 32, ef_construction: int = 64, ef_search: int = 64):

        self.embedding_service = EmbeddingService()
        self.index = None
        self.chunks: List[Dict[str, Any]] = []

        self.m = int(m)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)

        # Use actual embedding dimension to avoid mismatch errors
        self.dimension = self.embedding_service.dimension


    def create_index(self, chunks: List[Dict[str, Any]]):
        """Create the FAISS HNSW index from chunk list."""
        logger.info(f"Creating FAISS HNSW index for {len(chunks)} chunks")

        self.chunks = chunks

        # Extract text for embedding
        texts = [ch.get("text", "") for ch in chunks]
        embeddings = self.embedding_service.embed_batch(texts)

        if not embeddings:
            raise RuntimeError("Embedding generation failed — cannot build vector index")

        embeddings_np = np.array(embeddings).astype("float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_np)

        try:
            # Create index
            index = faiss.IndexHNSWFlat(self.dimension, self.m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search

            index.add(embeddings_np)
            self.index = index

            logger.info(f"FAISS index created with {self.index.ntotal} vectors")

        except Exception as e:
            logger.error(f"FAISS index creation failed: {e}")
            raise RuntimeError("Error creating FAISS index. Try reinstalling faiss-cpu.") from e


    def search(self, query: str, top_k: int = 60) -> List[Dict[str, Any]]:
        """Search for similar chunks using ANN."""
        if self.index is None:
            raise ValueError("Index not created — call create_index() first.")

        query_emb = self.embedding_service.embed_text(query)
        query_np = np.array([query_emb]).astype("float32")
        faiss.normalize_L2(query_np)

        try:
            scores, idxs = self.index.search(query_np, min(top_k, len(self.chunks)))
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []

        results = []
        for score, i in zip(scores[0], idxs[0]):
            if 0 <= i < len(self.chunks):
                results.append({
                    "text": self.chunks[i]["text"],
                    "metadata": self.chunks[i]["metadata"],
                    "score": float(score)
                })

        return results


    def search_multiple_queries(self, queries: List[str], top_k_per_query: int = 20) -> List[Dict[str, Any]]:
        """
        Perform multiple search queries and return combined unique results.
        Highest similarity scores sorted first.
        """
        logger.info(f"Running {len(queries)} retrieval queries")

        all_results: List[Dict[str, Any]] = []
        seen_ids = set()

        for q in queries:
            hits = self.search(q, top_k=top_k_per_query)

            for item in hits:
                chunk_id = item["metadata"].get("chunk_id")
                if chunk_id and chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    all_results.append(item)

        # Sort by FAISS similarity score
        all_results.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"Total unique retrieved chunks: {len(all_results)}")

        return all_results
