

from src.embedder import Embedder
from src.memory import VectorMemory

DATA_PATH = "data/dialogs.csv"


def load_dialogs(path):
   
    questions, answers = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue  # skip anything that doesn't look like question,answer
            q, a = parts[0].strip(), parts[1].strip()
            if not q or not a:
                continue
            questions.append(q)
            answers.append(a)
    return questions, answers


def main():
    questions, answers = load_dialogs(DATA_PATH)
    print(f"loaded {len(questions)} rows from {DATA_PATH}")

    embedder = Embedder()
    memory = VectorMemory(dim=embedder.dim, index_path="data/memory_index")

    texts = [f"User asked: {q} | Bot answered: {a}" for q, a in zip(questions, answers)]

    print("embedding dataset, this can take a minute on CPU...")
    vectors = embedder.encode(texts)

    memory.add(vectors, texts)
    memory.save()

    print(f"memory store built with {len(memory)} entries -> data/memory_index.faiss")


if __name__ == "__main__":
    main()