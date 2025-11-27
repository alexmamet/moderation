# Text Moderation Service

FastAPI text moderation service with GPU support.

## Categories

- **S1** (Violence): cannibalism, necro
- **S2** (Hate Speech): discrimination
- **S3** (Sexual Crimes): rape, zoo
- **S4** (Minors): child_abuse, incest
- **S11** (Self-Harm): suicide

## Build and Run

```bash
docker build -t text-moderation .
docker run -p 8000:8000 --gpus all text-moderation
```

## API

POST /moderate
```json
{
  "text": "Your text to moderate"
}
```

Response:
```json
{
  "result": "safe",
  "categories": []
}
```

or

```json
{
  "result": "unsafe",
  "categories": ["S4", "S3"]
}
```

GET /health
```json
{
  "status": "ok",
  "device": "cuda"
}
```
