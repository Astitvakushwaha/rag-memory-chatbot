import os
import pickle

import faiss
import numpy as np


class VectorMemory:
    """
    Thin wrapper around a FAISS flat index so the chatbot can remember
    past turns and pull the most relevant ones back up later.

    Nothing fancy here - IndexFlatL2 is enough for a few thousand
    conversation turns. If this ever needs to scale to millions of
    entries, swap it for IndexIVFFlat or similar.
    """

    def __init__(self, dim, index_path="data/memory_index"):
        self.dim = dim
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []      # raw text tied to each vector, same order as the index
        self.metadata = []   # timestamps, session ids, whatever you want to tag on

        if os.path.exists(f"{index_path}.faiss"):
            self.load()

    def add(self, vectors, texts, metadata=None):
        vectors = np.array(vectors).astype("float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        self.index.add(vectors)
        self.texts.extend(texts)

        if metadata is None:
            metadata = [{} for _ in texts]
        self.metadata.extend(metadata)

    def search(self, query_vector, k=3):
        if self.index.ntotal == 0:
            return []

        query_vector = np.array([query_vector]).astype("float32")
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "score": float(dist),
            })
        return results

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, f"{self.index_path}.faiss")
        with open(f"{self.index_path}_meta.pkl", "wb") as f:
            pickle.dump({"texts": self.texts, "metadata": self.metadata}, f)

    def load(self):
        self.index = faiss.read_index(f"{self.index_path}.faiss")
        with open(f"{self.index_path}_meta.pkl", "rb") as f:
            data = pickle.load(f)
        self.texts = data["texts"]
        self.metadata = data["metadata"]

    def __len__(self):
        return self.index.ntotal
