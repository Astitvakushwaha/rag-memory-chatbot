from sentence_transformers import SentenceTransformer


class Embedder:
    """Turns text into vectors so we can compare meaning, not just words."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        try:
            self.dim = self.model.get_embedding_dimension()
        except AttributeError:
            # older sentence-transformers versions use this name instead
            self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, text):
        # works for a single string or a list of strings
        return self.model.encode(text, show_progress_bar=False)