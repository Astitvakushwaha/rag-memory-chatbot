def generate(self, query, memory_context):
        if memory_context:
            context_str = "\n".join(f"- {c}" for c in memory_context)
            prompt = (
                "You are a chatbot having a live conversation. Below are notes "
                "from earlier conversations that might be relevant. Use them "
                "only as background - do NOT copy or repeat their wording. "
                "Write a brand new, natural reply to the current message.\n\n"
                f"Background notes:\n{context_str}\n\n"
                f"Current message from the user: {query}\n\n"
                "Write only your reply, in your own words:"
            )
        else:
            prompt = (
                "You are a chatbot having a live conversation. Reply naturally "
                f"to this message:\n\n{query}\n\nYour reply:"
            )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()