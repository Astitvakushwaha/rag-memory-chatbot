import streamlit as st

from src.chatbot import MemoryChatbot

st.set_page_config(page_title="Memory Chatbot", page_icon="🧠")

st.title("🧠 Conversational Agent with Long-term Memory")
st.caption("RAG-based chatbot - retrieves relevant past turns from a FAISS vector store before replying.")


@st.cache_resource
def load_bot():
    return MemoryChatbot()


bot = load_bot()

if "history" not in st.session_state:
    st.session_state.history = []

# replay past messages in this session
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn.get("retrieved"):
            with st.expander(f"used {len(turn['retrieved'])} memories"):
                for r in turn["retrieved"]:
                    st.write(f"- {r['text']}")

user_input = st.chat_input("Type a message...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("thinking..."):
            response, retrieved = bot.chat(user_input)
        st.write(response)
        if retrieved:
            with st.expander(f"used {len(retrieved)} memories"):
                for r in retrieved:
                    st.write(f"- {r['text']}")

    st.session_state.history.append({
        "role": "assistant",
        "content": response,
        "retrieved": retrieved,
    })

    bot.save_memory()

st.sidebar.header("About")
st.sidebar.write(
    "Every message you send gets embedded and stored. When you send a new "
    "message, the bot searches past turns for related context and uses it "
    "to answer - that's the 'long-term memory' part."
)
if st.sidebar.button("Clear chat display"):
    st.session_state.history = []
    st.rerun()