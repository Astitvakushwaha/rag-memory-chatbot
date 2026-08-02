# Conversational Agent with Long-term Memory (RAG)

A chatbot that remembers past conversations. Instead of forgetting everything
once a session ends, every turn gets embedded and stored in a FAISS vector
index. When you say something new, the bot searches that index for related
past turns and uses them as extra context before generating a reply — basic
retrieval-augmented generation (RAG), applied to conversation memory instead
of documents.

## How it works

1. Every message you type gets converted into a vector using
   `sentence-transformers` (`all-MiniLM-L6-v2`).
2. That vector is used to search a FAISS index of everything said so far
   (including a big batch of conversations loaded from a Kaggle dataset up
   front).
3. The top matching past turns get stuffed into the prompt as context.
4. `google/flan-t5-base` generates the reply using that context.
5. The new turn gets added back into the FAISS index, so memory keeps
   growing the longer you chat.

## Project structure

```
rag-memory-chatbot/
├── data/
│   └── dialogs.csv              <- you add this (see Dataset section)
├── src/
│   ├── embedder.py              # wraps sentence-transformers
│   ├── memory.py                # FAISS vector store
│   ├── generator.py             # flan-t5 response generation
│   └── chatbot.py                # ties it all together
├── build_memory_from_dataset.py # bootstraps memory from the Kaggle csv
├── main.py                       # run this to chat
└── requirements.txt
```

## Dataset

This project uses a small conversational dataset from Kaggle to pre-fill the
bot's memory. Any two-column (question, answer) chatbot dataset works —
this was built and tested against:

**Simple Dialogs for Chatbot** — https://www.kaggle.com/datasets/grafstor/simple-dialogs-for-chatbot

1. Download the CSV from the link above (you'll need a free Kaggle account).
2. Rename it to `dialogs.csv` and drop it into the `data/` folder.
3. If your CSV has different column names/order, just adjust the two lines
   in `build_memory_from_dataset.py` that read `df.iloc[:, 0]` / `df.iloc[:, 1]`.

## Setup (Windows, VS Code)

Open the project folder in VS Code, then open a terminal (`` Ctrl+` ``) and run:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

First run will download the embedding + generation models from Hugging Face
(a few hundred MB total), so make sure you're online the first time.

## Usage

Build the memory from the dataset (do this once):

```powershell
python build_memory_from_dataset.py
```

Then start chatting:

```powershell
python main.py
```

Type normally, and `quit` to exit (this also saves any new memories from the
session back to disk so they're there next time you run it).

## Notes / possible improvements

- Swapping `flan-t5-base` for a bigger model will give noticeably better
  replies at the cost of speed/RAM.
- `IndexFlatL2` is fine for a few thousand entries; for anything bigger,
  FAISS has `IndexIVFFlat` which is faster to search.
- No API keys needed anywhere — everything runs locally.
