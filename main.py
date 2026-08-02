from src.chatbot import MemoryChatbot


def main():
    print("loading models, give it a few seconds...\n")
    bot = MemoryChatbot()
    print("Chatbot ready. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            bot.save_memory()
            print("Bot: saved everything we talked about, bye!")
            break

        response, retrieved = bot.chat(user_input)
        print(f"Bot: {response}")

        if retrieved:
            print(f"   (used {len(retrieved)} related memory from earlier conversations)")


if __name__ == "__main__":
    main()
