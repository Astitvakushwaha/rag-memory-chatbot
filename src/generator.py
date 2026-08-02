from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class ResponseGenerator:
    """
    Generates a reply given the current message plus whatever the memory
    search pulled up. flan-t5-base is small enough to run on CPU and
    follows instructions reasonably well for a project like this.

    Loading the model/tokenizer directly instead of going through
    transformers' pipeline() helper - some recent transformers versions
    don't register text2text-generation in the pipeline task registry,
    so this sidesteps that entirely.
    """

    def __init__(self, model_name="google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, query, memory_context):
        if memory_context:
            context_str = "\n".join(f"- {c}" for c in memory_context)
            prompt = (
                "You are a chatbot with memory of earlier conversations. "
                "Use the relevant past context below if it helps answer the "
                "current message, otherwise ignore it.\n\n"
                f"Relevant past context:\n{context_str}\n\n"
                f"Current message: {query}\n"
                "Reply:"
            )
        else:
            prompt = f"Current message: {query}\nReply:"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()