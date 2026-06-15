import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { createEmbedding, DEFAULT_MODEL_DIR, isLocalModelDir, resolveModel } from './embed-core.js';

const HOST = '127.0.0.1';
const DEFAULT_PORT = 3000;
const PUBLIC_DIR = path.resolve('public');

const MIME_TYPES = new Map([
  ['.html', 'text/html; charset=utf-8'],
  ['.css', 'text/css; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.svg', 'image/svg+xml'],
]);

function parseArgs(argv) {
  let port = Number(process.env.PORT ?? DEFAULT_PORT);

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === '--help' || arg === '-h') {
      return { help: true, port };
    }

    if (arg === '--port' || arg === '-p') {
      const value = argv[index + 1];

      if (!value) {
        throw new Error('Missing value for --port.');
      }

      port = Number(value);
      index += 1;
      continue;
    }

    if (arg.startsWith('--port=')) {
      port = Number(arg.slice('--port='.length));
      continue;
    }

    throw new Error(`Unknown option: ${arg}`);
  }

  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error('Port must be an integer from 1 to 65535.');
  }

  return { help: false, port };
}

function getUsage() {
  return [
    'Usage:',
    '  node server.js',
    '  node server.js --port 3001',
    '  node server.js -p 3001',
  ].join('\n');
}

const { help, port: PORT } = parseArgs(process.argv.slice(2));

if (help) {
  console.log(getUsage());
  process.exit(0);
}

function sendJson(response, statusCode, payload) {
  response.writeHead(statusCode, {
    'content-type': 'application/json; charset=utf-8',
  });
  response.end(JSON.stringify(payload));
}

function getModelNameFromDir(modelDir) {
  return modelDir.replaceAll('\\', '/').replace(/^\.\//, '');
}

async function readJsonBody(request) {
  const chunks = [];

  for await (const chunk of request) {
    chunks.push(chunk);
  }

  const rawBody = Buffer.concat(chunks).toString('utf8');

  if (!rawBody) {
    return {};
  }

  try {
    return JSON.parse(rawBody);
  } catch {
    throw new Error('Request body must be valid JSON.');
  }
}

async function handleEmbed(request, response) {
  if (request.method !== 'POST') {
    sendJson(response, 405, { error: 'Use POST /api/embed.' });
    return;
  }

  const body = await readJsonBody(request);
  const text = typeof body.text === 'string' ? body.text : '';
  const modelDir = typeof body.modelDir === 'string' && body.modelDir.trim()
    ? body.modelDir.trim()
    : DEFAULT_MODEL_DIR;

  const result = await createEmbedding(text, modelDir);

  sendJson(response, 200, {
    text: result.text,
    model: result.modelDir,
    dimension: result.dimension,
    embedding: result.embedding,
    preview: result.preview,
  });
}

async function listModels() {
  const entries = await fs.readdir('.', { withFileTypes: true });
  const models = [];

  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue;
    }

    if (entry.name === 'node_modules' || entry.name === '.git') {
      continue;
    }

    const modelDir = `./${entry.name}`;

    if (await isLocalModelDir(path.resolve(modelDir))) {
      models.push({
        name: getModelNameFromDir(modelDir),
        dir: modelDir,
      });
    }
  }

  models.sort((left, right) => left.name.localeCompare(right.name));
  return models;
}

async function handleModels(request, response) {
  if (request.method === 'GET') {
    sendJson(response, 200, { models: await listModels() });
    return;
  }

  if (request.method !== 'POST') {
    sendJson(response, 405, { error: 'Use GET or POST /api/models.' });
    return;
  }

  const body = await readJsonBody(request);
  const modelId = typeof body.modelId === 'string' ? body.modelId.trim() : '';
  const outputDir = typeof body.outputDir === 'string' ? body.outputDir.trim() : '';

  if (!modelId) {
    throw new Error('Hugging Face model ID is required.');
  }

  if (!outputDir) {
    throw new Error('Output directory is required.');
  }

  const modelRef = resolveModel(outputDir);

  try {
    await fs.access(modelRef.absoluteDir);
    throw new Error(`Output directory already exists: ${outputDir}`);
  } catch (error) {
    if (error instanceof Error && 'code' in error && error.code !== 'ENOENT') {
      throw error;
    }
  }

  await exportModel(modelId, modelRef.absoluteDir);

  sendJson(response, 201, {
    model: {
      name: getModelNameFromDir(outputDir),
      dir: outputDir,
    },
    models: await listModels(),
  });
}

function exportModel(modelId, outputDir) {
  return new Promise((resolve, reject) => {
    const child = spawn('optimum-cli', [
      'export',
      'onnx',
      '--model',
      modelId,
      '--task',
      'feature-extraction',
      '--library-name',
      'transformers',
      outputDir,
    ], {
      shell: false,
      windowsHide: true,
    });

    const output = [];

    child.stdout.on('data', (chunk) => output.push(chunk.toString()));
    child.stderr.on('data', (chunk) => output.push(chunk.toString()));
    child.on('error', (error) => reject(error));
    child.on('close', (code) => {
      if (code === 0) {
        resolve();
        return;
      }

      reject(new Error(`Model export failed with exit code ${code}.\n${output.join('').trim()}`));
    });
  });
}

async function serveStatic(request, response) {
  const url = new URL(request.url, `http://${request.headers.host}`);
  const pathname = url.pathname === '/' ? '/index.html' : decodeURIComponent(url.pathname);
  const absolutePath = path.resolve(PUBLIC_DIR, `.${pathname}`);

  if (!absolutePath.startsWith(PUBLIC_DIR)) {
    response.writeHead(403);
    response.end('Forbidden');
    return;
  }

  try {
    const file = await fs.readFile(absolutePath);
    const contentType = MIME_TYPES.get(path.extname(absolutePath)) ?? 'application/octet-stream';

    response.writeHead(200, { 'content-type': contentType });
    response.end(file);
  } catch {
    response.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
    response.end('Not found');
  }
}

const server = http.createServer(async (request, response) => {
  try {
    if (request.url?.startsWith('/api/embed')) {
      await handleEmbed(request, response);
      return;
    }

    if (request.url?.startsWith('/api/models')) {
      await handleModels(request, response);
      return;
    }

    await serveStatic(request, response);
  } catch (error) {
    sendJson(response, 500, {
      error: error instanceof Error ? error.message : 'Unknown server error.',
    });
  }
});

server.on('error', (error) => {
  if (error.code === 'EADDRINUSE') {
    console.error(`Port ${PORT} is already in use.`);
    console.error(`Open http://${HOST}:${PORT} if the service is already running.`);
    console.error('Or start another instance on a different port:');
    console.error('  node server.js --port 3001');
    process.exitCode = 1;
    return;
  }

  console.error(error);
  process.exitCode = 1;
});

server.listen(PORT, HOST, () => {
  console.log(`Embedding web service running at http://${HOST}:${PORT}`);
});
