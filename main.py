import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

logger.info("Loading model from HuggingFace...")
tokenizer = AutoTokenizer.from_pretrained("mlgethoney/test")
model = AutoModelForSequenceClassification.from_pretrained("mlgethoney/test")
model = model.to(device)
model.eval()
label2id = model.config.label2id
logger.info(f"Model loaded successfully on {device}")
logger.info(f"Model labels: {label2id}")


class ModerationRequest(BaseModel):
    text: str


class ModerationResponse(BaseModel):
    result: str
    categories: list[str]


def get_category_scores(text: str) -> dict[str, float]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = inputs.to(device)

    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(0)

    # S1 (Violence): cannibalism, necro
    s1 = max(logits[label2id.get("cannibalism", 0)].item(), logits[label2id.get("necro", 0)].item())

    # S2 (Hate Speech): discrimination
    s2 = logits[label2id.get("discrimination", 0)].item()

    # S3 (Sexual Crimes): rape, zoo (bestiality)
    s3 = max(logits[label2id.get("rape", 0)].item(), logits[label2id.get("zoo", 0)].item())

    # S4 (Minors): child_abuse, incest
    s4 = max(logits[label2id.get("child_abuse", 0)].item(), logits[label2id.get("incest", 0)].item())

    # S11 (Self-Harm): suicide
    s11 = logits[label2id.get("suicide", 0)].item()

    return {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S11": s11}


@app.post("/moderate", response_model=ModerationResponse)
def moderate(request: ModerationRequest):
    text = request.text
    logger.info(f"Processing: {text[:100]}...")

    scores = get_category_scores(text)
    blocked_categories = [cat for cat, score in scores.items() if score > 0]

    result = "unsafe" if blocked_categories else "safe"

    logger.info(f"Scores: {scores}, blocked: {blocked_categories}")

    return ModerationResponse(result=result, categories=blocked_categories)


@app.get("/health")
def health():
    return {"status": "ok", "device": device}
