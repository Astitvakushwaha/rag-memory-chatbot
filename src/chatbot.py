import datetime

from src.embedder import Embedder
from src.generator import ResponseGenerator
from src.memory import VectorMemory


class MemoryChatbot:
    def __init__(self, index_path="data/memory_index", top_k=3):
        self.embedder = Embedder()
        self.memory = VectorMemory(dim=self.embedder.dim, index_path=index_path)
        self.generator = ResponseGenerator()
        self.top_k = top_k

    def chat(self, user_input):
        query_vec = self.embedder.encode(user_input)

        # pull back whatever's relevant from earlier turns
        retrieved = self.memory.search(query_vec, k=self.top_k)
        context_texts = [r["text"] for r in retrieved]

        response = self.generator.generate(user_input, context_texts)

        # store this turn so future questions can find it too
        self.memory.add(
            [query_vec],
            [f"User said: {user_input} | Bot replied: {response}"],
            [{"timestamp": datetime.datetime.now().isoformat()}],
        )

        return response, retrieved

    def save_memory(self):
        self.memory.save()
