"""
embeddings.py — Sentence-transformer embeddings + FAISS index for retrieval.

Builds a FAISS index from text chunks at startup.
Provides fast similarity search for incoming queries.
"""

import logging

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Model for generating embeddings (small, fast, good quality)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class StatuteIndex:
    """FAISS-backed vector index for CCPA statute chunks."""

    def __init__(self):
        self.model: SentenceTransformer | None = None
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[str] = []
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def build(self, chunks: list[str]) -> None:
        """Build the FAISS index from text chunks. Called once at startup."""
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        self.chunks = chunks
        embeddings = self.model.encode(
            chunks,
            show_progress_bar=True,
            normalize_embeddings=True,  # for cosine similarity via inner product
            batch_size=32,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # Build FAISS index using inner product (cosine sim with normalized vectors)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        self._is_ready = True
        logger.info(
            f"FAISS index built: {self.index.ntotal} vectors, dim={dimension}"
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve the top_k most relevant chunks for a query."""
        if not self._is_ready:
            raise RuntimeError("Index not built. Call build() first.")

        # Encode query
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks) and idx >= 0:
                results.append(self.chunks[idx])
                logger.debug(
                    f"  Chunk {idx} (score={scores[0][i]:.4f}): "
                    f"{self.chunks[idx][:80]}..."
                )

        return results


# Global singleton — built once at startup, reused for all requests
statute_index = StatuteIndex()
