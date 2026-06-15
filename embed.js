import fs from 'node:fs/promises';
import path from 'node:path';
import clipboardy from 'clipboardy';
import { AutoModel, AutoTokenizer, env } from '@xenova/transformers';

const MODEL_DIR = './krsbert-onnx';
const MODEL_ID = 'krsbert-onnx';
const REQUIRED_MODEL_FILES = [
  'model.onnx',
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
];

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = path.resolve('.');

async function assertLocalModelFiles(modelDir) {
  const missing = [];

  for (const file of REQUIRED_MODEL_FILES) {
    try {
      await fs.access(path.join(modelDir, file));
    } catch {
      missing.push(file);
    }
  }

  if (missing.length > 0) {
    throw new Error(
      [
        `Missing local model file(s) in ${modelDir}: ${missing.join(', ')}`,
        '',
        'Export the model locally first:',
        'optimum-cli export onnx \\',
        '  --model snunlp/KR-SBERT-V40K-klueNLI-augSTS \\',
        '  --task feature-extraction \\',
        '  krsbert-onnx',
      ].join('\n'),
    );
  }
}

function getTensorData(tensor) {
  if (!tensor || !tensor.data) {
    throw new Error('Expected model output tensor data was not found.');
  }

  return tensor.data;
}

function meanPool(lastHiddenState, attentionMask) {
  const hiddenData = getTensorData(lastHiddenState);
  const dims = lastHiddenState.dims;

  if (!Array.isArray(dims) || dims.length !== 3) {
    throw new Error(`Expected last_hidden_state dimensions [batch, tokens, hidden], got ${JSON.stringify(dims)}.`);
  }

  const [batchSize, sequenceLength, hiddenSize] = dims;

  if (batchSize !== 1) {
    throw new Error(`This script expects a single input text, got batch size ${batchSize}.`);
  }

  const maskData = attentionMask ? getTensorData(attentionMask) : null;
  const pooled = new Array(hiddenSize).fill(0);
  let validTokenCount = 0;

  // Sentence-BERT mean pooling: sum token embeddings only where attention_mask is 1,
  // so padding tokens do not affect the sentence representation.
  for (let tokenIndex = 0; tokenIndex < sequenceLength; tokenIndex += 1) {
    const isValidToken = maskData ? Number(maskData[tokenIndex]) === 1 : true;

    if (!isValidToken) {
      continue;
    }

    validTokenCount += 1;
    const tokenOffset = tokenIndex * hiddenSize;

    for (let hiddenIndex = 0; hiddenIndex < hiddenSize; hiddenIndex += 1) {
      pooled[hiddenIndex] += Number(hiddenData[tokenOffset + hiddenIndex]);
    }
  }

  if (validTokenCount === 0) {
    throw new Error('Tokenization produced no valid tokens to pool.');
  }

  return pooled.map((value) => value / validTokenCount);
}

function l2Normalize(vector) {
  // L2 normalization makes cosine similarity equivalent to dot product for
  // downstream vector search and nearest-neighbor comparisons.
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));

  if (!Number.isFinite(norm) || norm === 0) {
    throw new Error('Cannot normalize an empty or zero-valued embedding vector.');
  }

  return vector.map((value) => value / norm);
}

async function main() {
  const text = process.argv.slice(2).join(' ').trim();

  if (!text) {
    throw new Error('Usage: node embed.js "페라트라정"');
  }

  await assertLocalModelFiles(MODEL_DIR);

  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
    local_files_only: true,
  });
  const model = await AutoModel.from_pretrained(MODEL_ID, {
    local_files_only: true,
    quantized: false,
    model_file_name: '../model',
  });

  // Tokenization converts the input string into model tensors, including
  // attention_mask, which marks real tokens as 1 and padding as 0.
  const inputs = await tokenizer(text, {
    padding: true,
    truncation: true,
  });

  const output = await model(inputs);

  if (!output.last_hidden_state) {
    throw new Error('Model output did not include last_hidden_state.');
  }

  const embedding = l2Normalize(meanPool(output.last_hidden_state, inputs.attention_mask));
  const preview = embedding.slice(0, 10).map((value) => value.toFixed(6));

  // Clipboard export writes only the full JSON array, with no labels or extra text.
  await clipboardy.write(JSON.stringify(embedding));

  console.log(`Text      : ${text}`);
  console.log(`Model     : ${MODEL_DIR}`);
  console.log(`Dimension : ${embedding.length}`);
  console.log(`Embedding : [${preview.join(', ')}, ...]`);
  console.log('Full embedding copied to clipboard.');
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
