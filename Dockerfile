FROM node:24-bookworm

WORKDIR /app

ENV NODE_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:${PATH}"
ENV DEFAULT_MODEL_DIR="./models/krsbert-onnx"

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

RUN mkdir -p /app/models

EXPOSE 3000

CMD ["node", "server.js", "--host", "0.0.0.0", "--port", "3000"]
