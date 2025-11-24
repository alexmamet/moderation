# RunPod Text Moderation Serverless

Serverless text moderation service using BERT model on RunPod.

## Features

- Banned words check (20 critical terms)
- ML toxicity classification using mlgethoney/test model
- Returns unsafe if any logit > 0
- CPU/GPU inference

## Build and Deploy

```bash
# Build Docker image
docker build -t runpod-text-moderation .

# Tag for registry
docker tag runpod-text-moderation:latest <your-registry>/runpod-text-moderation:latest

# Push to registry
docker push <your-registry>/runpod-text-moderation:latest
```

Then deploy on RunPod using the image URL.

## API Usage

Input:
```json
{
  "input": {
    "text": "Your text to moderate"
  }
}
```

Output:
```json
{
  "result": "safe" | "unsafe",
  "toxicity_score": 0.123,
  "reason": "banned_word" (optional),
  "matched_word": "..." (optional)
}
```

## Local Testing

```bash
# Install dependencies
uv pip install -r pyproject.toml

# Run handler
python handler.py
```
