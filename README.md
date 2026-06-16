# Local KR-SBERT Embeddings

This project generates sentence embeddings locally with:

```text
snunlp/KR-SBERT-V40K-klueNLI-augSTS
```

It does not use the Hugging Face Inference API, `HF_TOKEN`, or any remote inference service at runtime. Runtime inference uses the local ONNX model in `./krsbert-onnx`.

## First-Time Setup

If this is your first run, you may not have these directories yet:

```text
node_modules/
krsbert-onnx/
```

That is expected. Create them with the steps below.

### 1. Install Node.js Dependencies

Run this from the project directory:

```bash
npm install
```

This creates:

```text
node_modules/
package-lock.json
```

The required packages are already listed in `package.json`:

```json
{
  "type": "module",
  "dependencies": {
    "@xenova/transformers": "^2.17.2",
    "clipboardy": "^5.3.1"
  }
}
```

### 2. Install the ONNX Export Tooling

You need `optimum-cli` to export the Hugging Face model to ONNX:

```bash
pip install optimum-onnx onnx onnxruntime
```

If `optimum-cli` is still unavailable, also run:

```bash
pip install optimum
```

### 3. Export the Model

The original requested export command is:

```bash
optimum-cli export onnx \
  --model snunlp/KR-SBERT-V40K-klueNLI-augSTS \
  --task feature-extraction \
  krsbert-onnx
```

For this model, current Optimum versions may try to load it through the `sentence_transformers` wrapper and fail with:

```text
AttributeError: property 'config' of 'SentenceTransformer' object has no setter
```

If that happens, use the verified working command:

```bash
optimum-cli export onnx \
  --model snunlp/KR-SBERT-V40K-klueNLI-augSTS \
  --task feature-extraction \
  --library-name transformers \
  krsbert-onnx
```

This creates:

```text
krsbert-onnx/
  model.onnx
  config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  vocab.txt
```

The export step downloads the model once. After that, `embed.js` runs inference from local files.

## Run

Generate an embedding:

```bash
node embed.js "페라트라정"
```

Expected console output:

```text
Text      : 페라트라정
Model     : ./krsbert-onnx
Dimension : 768
Embedding : [-0.069932, 0.016134, -0.033339, -0.036370, 0.049995, 0.013386, -0.069461, -0.003972, 0.013122, 0.024493, ...]
Full embedding copied to clipboard.
```

Only the first 10 embedding values are printed. The full embedding is copied to the clipboard as a JSON array.

## Run as a Web Service

Start the local web service:

```bash
npm start
```

Then open:

```text
http://127.0.0.1:3000
```

If port `3000` is already in use, run on another port.

Any shell:

```bash
node server.js --port 3001
```

Short form:

```bash
node server.js -p 3001
```

Then open:

```text
http://127.0.0.1:3001
```

The page lets you:

- Enter text to embed.
- Select an exported local ONNX model.
- Add a new Hugging Face model by exporting it to a local directory.
- Copy the full embedding JSON array with a button.

The browser calls the local Node.js service at `/api/embed`. Inference still runs on local ONNX files.

## Docker

Build a Docker image under the Docker Hub namespace `catac1`:

```bash
docker build -t catac1/krsbert-embeddings:latest .
```

The Docker image does not include model weights. This keeps the image smaller and avoids downloading from Hugging Face during `docker build`.

Create a local folder for exported models:

PowerShell:

```powershell
mkdir models
docker run --rm -p 3000:3000 -v "${PWD}\models:/app/models" catac1/krsbert-embeddings:latest
```

cmd.exe:

```cmd
mkdir models
docker run --rm -p 3000:3000 -v "%cd%\models:/app/models" catac1/krsbert-embeddings:latest
```

Then open:

```text
http://127.0.0.1:3000
```

In the page, use **Add Model**:

```text
Hugging Face model ID : snunlp/KR-SBERT-V40K-klueNLI-augSTS
Volume output directory: ./models/krsbert-onnx
```

The container downloads and exports the model into `/app/models/krsbert-onnx`, which is mapped to your local `models/` folder.
The server rejects web-added model output paths outside `./models/` so Docker exports are not accidentally written only inside the container filesystem.

After export finishes, select `models/krsbert-onnx` in the model selector and generate embeddings.

Push the image to Docker Hub:

```bash
docker push catac1/krsbert-embeddings:latest
```

The web service inside Docker listens on `0.0.0.0:3000`, so Docker port mapping works with `-p 3000:3000`.

To run on another host port:

```bash
docker run --rm -p 3001:3000 -v "${PWD}\models:/app/models" catac1/krsbert-embeddings:latest
```

You can also use a named Docker volume instead of a local folder:

```bash
docker run --rm \
  -p 3000:3000 \
  -v embedding-models:/app/models \
  catac1/krsbert-embeddings:latest
```

When using that volume, add models from the page with output directories like:

```text
./models/my-model-onnx
```

## Use a Different Hugging Face Model

Export the other model into its own local ONNX directory:

```bash
optimum-cli export onnx \
  --model <hugging-face-model-id> \
  --task feature-extraction \
  --library-name transformers \
  other-model-onnx
```

Then choose that model in a single `embed.js` command:

```bash
node embed.js --model ./other-model-onnx "페라트라정"
```

The short form is also supported:

```bash
node embed.js -m ./other-model-onnx "페라트라정"
```

The model directory must be inside this project directory and contain:

```text
model.onnx
config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

## Troubleshooting

If `node_modules/` is missing:

```bash
npm install
```

If `krsbert-onnx/` is missing:

```bash
optimum-cli export onnx \
  --model snunlp/KR-SBERT-V40K-klueNLI-augSTS \
  --task feature-extraction \
  --library-name transformers \
  krsbert-onnx
```

If clipboard copying fails on Windows, rerun the command from a normal terminal:

```bash
node embed.js "페라트라정"
```
