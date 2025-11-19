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

        # embedding dimension
        self.dimension = self.embedding_service.dimension

    # NORMALIZED TEXT FOR EMBEDDING (MAJOR FIX)
    def _build_embedding_text(self, chunk: Dict[str, Any]) -> str:
        """
        Build richer embedding text to improve retrieval.
        This prevents wrongly retrieving neutral segments.
        """
        meta = chunk.get("metadata", {}) or {}

        stance = meta.get("stance", "neutral")
        concepts = meta.get("legal_concepts", [])
        arg_type = meta.get("argument_type", "other")

        # Build metadata-aware embedding text
        emb_text = (
            f"{chunk.get('text','')}\n"
            f"STANCE: {stance}\n"
            f"ARGUMENT_TYPE: {arg_type}\n"
            f"LEGAL_CONCEPTS: {', '.join(concepts)}"
        )

        return emb_text.strip()

    # CREATE INDEX
    def create_index(self, chunks: List[Dict[str, Any]]):
        logger.info(f"Creating FAISS HNSW index for {len(chunks)} chunks")

        if not chunks:
            raise RuntimeError("No chunks to index")

        self.chunks = chunks

        # Build embedding texts
        emb_texts = [self._build_embedding_text(ch) for ch in chunks]

        # Create embeddings
        try:
            embeddings = self.embedding_service.embed_batch(emb_texts)
        except Exception as e:
            logger.error(f"Embed batch failed: {e}")
            raise RuntimeError("Embedding failed")

        if not embeddings:
            raise RuntimeError("Embeddings returned empty")

        emb_np = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(emb_np)  # important for cosine similarity

        try:
            index = faiss.IndexHNSWFlat(self.dimension, self.m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search

            index.add(emb_np)
            self.index = index

            logger.info(f"FAISS index created with {index.ntotal} vectors")

        except Exception as e:
            logger.error(f"FAISS index creation error: {e}")
            raise RuntimeError("Could not create FAISS index")

    # SEARCH
    def search(self, query: str, top_k: int = 60) -> List[Dict[str, Any]]:
        if self.index is None:
            raise ValueError("FAISS index not created")

        if not query or len(query.strip()) == 0:
            logger.warning("Empty query received")
            return []

        # embed query
        try:
            q_emb = self.embedding_service.embed_text(query)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return []

        q_np = np.array([q_emb], dtype="float32")
        faiss.normalize_L2(q_np)

        try:
            scores, idxs = self.index.search(q_np, min(top_k, len(self.chunks)))
        except Exception as e:
            logger.error(f"FAISS search failure: {e}")
            return []

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if 0 <= idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx]["text"],
                    "metadata": self.chunks[idx]["metadata"],
                    "score": float(score)
                })

        return results

    # MULTI-QUERY SEARCH
    def search_multiple_queries(self, queries: List[str], top_k_per_query: int = 20) -> List[Dict[str, Any]]:
        logger.info(f"Running {len(queries)} queries for retrieval")

        all_results: List[Dict[str, Any]] = []
        seen_ids = set()

        for q in queries:
            hits = self.search(q, top_k_per_query)

            for item in hits:
                cid = item["metadata"].get("chunk_id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    all_results.append(item)

        # Sort by score descending
        all_results.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"Retrieved {len(all_results)} unique chunks after merging queries")

        return all_results
