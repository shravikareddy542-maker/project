"""
FAISS-based fast cosine similarity matcher.
Builds an IndexFlatIP (inner-product = cosine sim for L2-normalised vectors)
and maps FAISS row ids → (guest_id, embedding_id, guest_name).
"""
import os
import pickle
import numpy as np
import faiss

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class FAISSMatcher:
    """
    Maintains a FAISS inner-product index for fast cosine similarity search
    on L2-normalised ArcFace embeddings.
    """

    def __init__(self, dim: int = None):
        self.dim = dim or config.EMBEDDING_DIM
        self.index: faiss.IndexFlatIP | None = None
        # row_id → {guest_id, embedding_id, name}
        self.id_map: list[dict] = []

    # ───────────── Build / Rebuild ────────────────────────

    def build_index(self, records: list[dict]):
        """
        Build index from database records.
        Each record: {embedding_id, guest_id, embedding: np.ndarray, name}
        """
        self.id_map = []
        if not records:
            self.index = faiss.IndexFlatIP(self.dim)
            return

        vecs = np.stack([r["embedding"] for r in records]).astype(np.float32)
        # Ensure L2 normalised
        faiss.normalize_L2(vecs)

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vecs)

        self.id_map = [
            {
                "guest_id": r["guest_id"],
                "embedding_id": r["embedding_id"],
                "name": r["name"],
            }
            for r in records
        ]

    def search(self, query: np.ndarray, top_k: int = None) -> list[dict]:
        """
        Search for closest matches.
        Returns list of {name, guest_id, embedding_id, score} sorted by score desc.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        top_k = top_k or config.TOP_K
        top_k = min(top_k, self.index.ntotal)

        q = query.reshape(1, -1).astype(np.float32).copy()
        faiss.normalize_L2(q)

        scores, indices = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = self.id_map[idx].copy()
            entry["score"] = float(score)
            results.append(entry)
        return results

    def match(self, query: np.ndarray,
              threshold: float = None) -> tuple[str | None, float, int | None]:
        """
        Convenience: return (guest_name, score, guest_id) or (None, 0.0, None).
        """
        threshold = threshold or config.RECOGNITION_THRESHOLD
        results = self.search(query, top_k=1)
        if results and results[0]["score"] >= threshold:
            r = results[0]
            return r["name"], r["score"], r["guest_id"]
        return None, 0.0, None

    # ───────────── Persistence ────────────────────────────

    def save(self, index_path: str = None, map_path: str = None):
        index_path = index_path or config.FAISS_INDEX_PATH
        map_path = map_path or config.FAISS_MAP_PATH
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, index_path)
        with open(map_path, "wb") as f:
            pickle.dump(self.id_map, f)

    def load(self, index_path: str = None, map_path: str = None) -> bool:
        index_path = index_path or config.FAISS_INDEX_PATH
        map_path = map_path or config.FAISS_MAP_PATH
        if not os.path.isfile(index_path) or not os.path.isfile(map_path):
            return False
        self.index = faiss.read_index(index_path)
        with open(map_path, "rb") as f:
            self.id_map = pickle.load(f)
        return True

    @property
    def total(self) -> int:
        return self.index.ntotal if self.index else 0
