# Conversational Agent with Long-term Memory (RAG)

A chatbot that remembers past conversations. Instead of forgetting everything
once a session ends, every turn gets embedded and stored in a FAISS vector
index. When you say something new, the bot searches that index for related
past turns and uses them as extra context before generating a reply — basic
retrieval-augmented generation (RAG), applied to conversation memory instead
of documents.

## Demo

![Chatbot demo](screenshots/demo.png)

*(Streamlit UI showing a live conversation. Each reply has an expandable
"used N memories" section showing exactly which past turns were retrieved
to answer the current message.)*

## How it works

1. Every message you type gets converted into a vector using
   `sentence-transformers` (`all-MiniLM-L6-v2`).
2. That vector is used to search a FAISS index of everything said so far
   (including a big batch of conversations loaded from a Kaggle dataset up
   front).
3. The