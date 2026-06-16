FROM node:24-bookworm

ARG HF_MODEL_ID=snunlp/KR-SBERT-V40K-klueNLI-augSTS
ARG MODEL_DIR=krsbert-onnx

WORKDIR /app

ENV NODE_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:${PATH}"
ENV DEFAULT_MODEL_DIR="./${MODEL_DIR}"

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 \
    python3-pip \
    python3-venv \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
  && pip install --upgrade pip \
  && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
  && pip install --no-cache-dir optimum-onnx onnx onnxruntime

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY embed-core.js embed.js server.js ./
COPY public ./public

RUN optimum-cli export onnx \
  --model "${HF_MODEL_ID}" \
  --task feature-extraction \
  --library-name transformers \
  "${MODEL_DIR}"

EXPOSE 3000

CMD ["node", "server.js", "--host", "0.0.0.0", "--port", "3000"]
