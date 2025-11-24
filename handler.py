import runpod
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger

BANNED_WORDS = [
    "child porn",
    "cp",
    "pedophile",
    "minor sex",
    "kill yourself",
    "suicide instructions",
    "bomb instructions",
    "terrorist attack",
    "credit card fraud",
    "stolen identity",
    "human trafficking",
    "drug dealing",
    "rape fantasy",
    "sexual assault",
    "non-consensual",
    "racial slur",
    "hate crime",
    "ethnic cleansing",
    "genocide promotion",
    "school shooting",
]

logger.info("Loading model from HuggingFace...")
tokenizer = AutoTokenizer.from_pretrained("mlgethoney/test")
model = AutoModelForSequenceClassification.from_pretrained("mlgethoney/test")
model.eval()
logger.info("Model loaded successfully")


def check_banned_words(text: str) -> tuple[bool, str]:
    text_lower = text.lower()
    for word in BANNED_WORDS:
        if word in text_lower:
            return True, word
    return False, ""


def get_toxicity_score(text: str) -> float:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(0)

    max_score = logits.max().item()
    return max_score


def handler(job):
    job_input = job["input"]
    logger.info(f"Processing text moderation request: {str(job_input)[:100]}...")

    text = job_input.get("text", "")

    if not text:
        return {"error": "Text field is required"}

    is_banned, matched_word = check_banned_words(text)

    if is_banned:
        logger.info(f"Text blocked by banned word: {matched_word}")
        return {
            "result": "unsafe",
            "reason": "banned_word",
            "matched_word": matched_word
        }

    toxicity_score = get_toxicity_score(text)
    is_toxic = toxicity_score > 0

    logger.info(f"Toxicity score: {toxicity_score:.3f}, is_toxic: {is_toxic}")

    return {
        "result": "unsafe" if is_toxic else "safe",
        "toxicity_score": toxicity_score
    }


runpod.serverless.start({"handler": handler})
