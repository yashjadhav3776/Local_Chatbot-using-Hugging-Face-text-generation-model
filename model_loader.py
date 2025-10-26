from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)

def load_model(model_name: str = "google/flan-t5-large", device: int = -1):
    """
    Loads a T5-based question answering model.
    """
    logging.info(f"Loading model: {model_name} (device={device})")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    logging.info("✅ Model loaded successfully.")
    return generator, tokenizer
