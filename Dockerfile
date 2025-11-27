FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn transformers loguru

RUN pip install --no-cache-dir huggingface_hub && \
    huggingface-cli download mlgethoney/test

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
