import clipboardy from 'clipboardy';
import { createEmbedding, DEFAULT_MODEL_DIR } from './embed-core.js';

function parseArgs(argv) {
  const textParts = [];
  let modelDir = DEFAULT_MODEL_DIR;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === '--help' || arg === '-h') {
      return { help: true, modelDir, text: '' };
    }

    if (arg === '--model' || arg === '-m') {
      const value = argv[index + 1];

      if (!value) {
        throw new Error('Missing value for --model.');
      }

      modelDir = value;
      index += 1;
      continue;
    }

    if (arg.startsWith('--model=')) {
      modelDir = arg.slice('--model='.length);
      continue;
    }

    textParts.push(arg);
  }

  return {
    help: false,
    modelDir,
    text: textParts.join(' ').trim(),
  };
}

function getUsage() {
  return [
    'Usage:',
    '  node embed.js "페라트라정"',
    '  node embed.js --model ./other-model-onnx "페라트라정"',
  ].join('\n');
}

async function main() {
  const { help, modelDir, text } = parseArgs(process.argv.slice(2));

  if (help) {
    console.log(getUsage());
    return;
  }

  if (!text) {
    throw new Error(getUsage());
  }

  const result = await createEmbedding(text, modelDir);
  const preview = result.preview.map((value) => value.toFixed(6));

  // Clipboard export writes only the full JSON array, with no labels or extra text.
  await clipboardy.write(JSON.stringify(result.embedding));

  console.log(`Text      : ${result.text}`);
  console.log(`Model     : ${result.modelDir}`);
  console.log(`Dimension : ${result.dimension}`);
  console.log(`Embedding : [${preview.join(', ')}, ...]`);
  console.log('Full embedding copied to clipboard.');
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
