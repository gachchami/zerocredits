FROM python:3.11-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

RUN pip wheel \
    --no-cache-dir \
    --wheel-dir /wheels \
    -r requirements.txt


FROM debian:bookworm-slim AS model

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       curl \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /models \
    && curl -L \
       --fail \
       --retry 3 \
       --retry-delay 2 \
       -o /models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
       https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf


FROM python:3.11-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOCAL_MODEL_PATH=/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    LOCAL_THREADS=2 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

COPY --from=builder /wheels /wheels

RUN pip install \
    --no-cache-dir \
    /wheels/* \
    && rm -rf /wheels

COPY --from=model /models /models

COPY main.py ./main.py
COPY zerocredits ./zerocredits

ENTRYPOINT ["python", "main.py"]
