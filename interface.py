import argparse
from model_loader import load_model
from chat_memory import ChatMemory

def main():
    parser = argparse.ArgumentParser(description="Local CLI Chatbot using Hugging Face models")
    parser.add_argument("--model", default="google/flan-t5-large", help="Hugging Face model name")
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, or GPU id")
    parser.add_argument("--max_messages", type=int, default=5, help="Sliding window size in messages")
    args = parser.parse_args()

    generator, tokenizer = load_model(model_name=args.model, device=args.device)
    memory = ChatMemory(max_messages=args.max_messages)

    print("🤖 Local CLI Chatbot — type your message and press Enter. Type /exit to quit.\n")

    try:
        while True:
            user_input = input("User: ").strip()
            if not user_input:
                continue

            if user_input.lower() == "/exit":
                print("Exiting chatbot. Goodbye!")
                break

            if user_input.lower() == "/clear":
                memory.clear()
                print("Memory cleared.")
                continue

            memory.add_user(user_input)

            # ✨ STRONG prompt for style control
            prompt = (
                "You are a friendly and knowledgeable assistant. "
                "Always respond in full, polite sentences. "
                "Use natural human tone.\n\n"
                "Examples:\n"
                "User: What is the capital of France?\n"
                "Bot: The capital of France is Paris.\n\n"
                "User: What about Italy?\n"
                "Bot: The capital of Italy is Rome.\n\n"
                "User: Hello!\n"
                "Bot: Hello there! How are you today?\n\n"
                f"User: {user_input}\nBot:"
            )

            result = generator(
                prompt,
                max_new_tokens=100,
                num_return_sequences=1
            )

            bot_reply = result[0]["generated_text"].strip()

            # Post-process
            if not bot_reply.endswith(('.', '!', '?')):
                bot_reply += '.'

            memory.add_bot(bot_reply)
            print(f"Bot: {bot_reply}\n")

    except KeyboardInterrupt:
        print("\nExiting chatbot. Goodbye!")

if __name__ == "__main__":
    main()
