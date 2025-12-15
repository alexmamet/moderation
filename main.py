import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger
import re

app = FastAPI()

DEATH_TRIGGERS = [
    "kys", "ky$", "k.y.s",
    "killurself", "kill-yourself", "kill_yourself",
    "unaliveyourself", "unalive-yourself",
    "offyourself", "off-yourself",
    "endyoursefl", "end-yourself",
    "sewer-slide", "sewerslide", "sudoku-yourself",
    "suicide", "die"
]

DEATH_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(word) for word in DEATH_TRIGGERS) + r')\b',
    re.IGNORECASE
)

MODEL_NAME = "mlgethoney/test"

CATEGORY_INDICES = {
    "S1": [0, 4],
    "S2": [1],
    "S3": [5, 7],
    "S4": [6, 2],
    "S11": [3],
}

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

logger.info("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()
logger.info(f"Model loaded on {device}")


class ModerationRequest(BaseModel):
    text: str


class ModerationResponse(BaseModel):
    result: str
    categories: list[str]



def check_death_triggers(text: str) -> bool:
    return bool(DEATH_PATTERN.search(text))


def get_category_scores(text: str) -> dict[str, float]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = inputs.to(device)

    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(0)

    scores = {}
    for category, indices in CATEGORY_INDICES.items():
        scores[category] = max(logits[idx].item() for idx in indices)

    return scores


@app.post("/moderate", response_model=ModerationResponse)
def moderate(request: ModerationRequest):
    text = request.text
    logger.info(f"Processing text ({len(text)} chars)")

    scores = get_category_scores(text)
    blocked_categories = [cat for cat, score in scores.items() if score > 0]

    if check_death_triggers(text):
        logger.info("Death triggers detected, adding S11")
        if "S11" not in blocked_categories:
            blocked_categories.append("S11")

    result = "unsafe" if blocked_categories else "safe"

    logger.info(f"Result: {result}, categories: {blocked_categories}")

    return ModerationResponse(result=result, categories=blocked_categories)


@app.get("/health")
def health():
    return {"status": "ok", "device": device}
