FROM runpod/base:0.6.3-cuda11.8.0

RUN pip install --no-cache-dir uv
RUN uv pip install 'huggingface_hub[cli,torch]' --system

RUN hf download mlgethoney/test

COPY pyproject.toml .
RUN uv pip install -r pyproject.toml --system

COPY handler.py handler.py

CMD python3 -u /handler.py
