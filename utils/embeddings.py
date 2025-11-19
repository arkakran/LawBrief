from sentence_transformers import SentenceTransformer
from typing import List
from loguru import logger
import os


class EmbeddingService:
    """Generates sentence embeddings using SentenceTransformers."""

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.device = device or os.getenv("EMBEDDING_DEVICE", "cpu")

        logger.info(f"Loading embedding model: {self.model_name} on device={self.device}")
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded. dimension={self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.model_name}': {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        if text is None:
            return [0.0] * getattr(self, "dimension", 384)
        emb = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return emb.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        if not texts:
            return []
        embs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=(len(texts) > 50))
        return embs.tolist()
