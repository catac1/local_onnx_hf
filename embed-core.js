import fs from 'node:fs/promises';
import path from 'node:path';
import { AutoModel, AutoTokenizer, env } from '@xenova/transformers';

export const DEFAULT_MODEL_DIR = process.env.DEFAULT_MODEL_DIR ?? './krsbert-onnx';

export const REQUIRED_MODEL_FILES = [
  'model.onnx',
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
];

const loadedModels = new Map();

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = path.resolve('.');

export function resolveModel(modelDir = DEFAULT_MODEL_DIR) {
  const absoluteModelDir = path.resolve(modelDir);
  const relativeModelId = path.relative(env.localModelPath, absoluteModelDir);

  if (relativeModelId.startsWith('..') || path.isAbsolute(relativeModelId)) {
    throw new Error('Model directory must be inside the current project directory.');
  }

  return {
    displayDir: modelDir,
    absoluteDir: absoluteModelDir,
    modelId: relativeModelId.replaceAll(path.sep, '/'),
  };
}

export async function assertLocalModelFiles(modelDir) {
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
        '  --model <hugging-face-model-id> \\',
        '  --task feature-extraction \\',
        '  --library-name transformers \\',
        `  ${modelDir}`,
      ].join('\n'),
    );
  }
}

export async function isLocalModelDir(modelDir) {
  try {
    await assertLocalModelFiles(modelDir);
    return true;
  } catch {
    return false;
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

async function loadLocalModel(modelRef) {
  const cached = loadedModels.get(modelRef.modelId);

  if (cached) {
    return cached;
  }

  await assertLocalModelFiles(modelRef.absoluteDir);

  const loaded = {
    tokenizer: await AutoTokenizer.from_pretrained(modelRef.modelId, {
      local_files_only: true,
    }),
    model: await AutoModel.from_pretrained(modelRef.modelId, {
      local_files_only: true,
      quantized: false,
      model_file_name: '../model',
    }),
  };

  loadedModels.set(modelRef.modelId, loaded);
  return loaded;
}

export async function createEmbedding(text, modelDir = DEFAULT_MODEL_DIR) {
  const normalizedText = text.trim();

  if (!normalizedText) {
    throw new Error('Input text is required.');
  }

  const modelRef = resolveModel(modelDir);
  const { tokenizer, model } = await loadLocalModel(modelRef);

  // Tokenization converts the input string into model tensors, including
  // attention_mask, which marks real tokens as 1 and padding as 0.
  const inputs = await tokenizer(normalizedText, {
    padding: true,
    truncation: true,
  });

  const output = await model(inputs);

  if (!output.last_hidden_state) {
    throw new Error('Model output did not include last_hidden_state.');
  }

  const embedding = l2Normalize(meanPool(output.last_hidden_state, inputs.attention_mask));

  return {
    text: normalizedText,
    modelDir: modelRef.displayDir,
    dimension: embedding.length,
    embedding,
    preview: embedding.slice(0, 10),
  };
}
