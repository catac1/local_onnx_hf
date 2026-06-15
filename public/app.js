const form = document.querySelector('#embed-form');
const modelForm = document.querySelector('#model-form');
const textInput = document.querySelector('#text-input');
const modelSelect = document.querySelector('#model-select');
const hfModelInput = document.querySelector('#hf-model-input');
const outputDirInput = document.querySelector('#output-dir-input');
const generateButton = document.querySelector('#generate-button');
const addModelButton = document.querySelector('#add-model-button');
const refreshModelsButton = document.querySelector('#refresh-models-button');
const copyButton = document.querySelector('#copy-button');
const statusOutput = document.querySelector('#status');
const dimensionOutput = document.querySelector('#dimension-output');
const modelOutput = document.querySelector('#model-output');
const previewOutput = document.querySelector('#preview-output');

let currentEmbeddingJson = '';

function selectedModelDir() {
  return modelSelect.value || './krsbert-onnx';
}

function setStatus(message, type = '') {
  statusOutput.textContent = message;
  statusOutput.className = `status${type ? ` ${type}` : ''}`;
}

function setBusy(isBusy) {
  generateButton.disabled = isBusy;
  generateButton.textContent = isBusy ? 'Generating...' : 'Generate';
}

function setModelBusy(isBusy) {
  addModelButton.disabled = isBusy;
  refreshModelsButton.disabled = isBusy;
  addModelButton.textContent = isBusy ? 'Adding...' : 'Add Model';
}

async function loadModels(preferredDir = '') {
  const response = await fetch('/api/models');
  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload.error || 'Could not load models.');
  }

  modelSelect.replaceChildren();

  for (const model of payload.models) {
    const option = document.createElement('option');
    option.value = model.dir;
    option.textContent = model.name;
    modelSelect.append(option);
  }

  if (preferredDir) {
    modelSelect.value = preferredDir;
  }

  if (!modelSelect.value && payload.models.length > 0) {
    modelSelect.value = payload.models[0].dir;
  }

  generateButton.disabled = payload.models.length === 0;
}

function renderResult(result) {
  currentEmbeddingJson = JSON.stringify(result.embedding);
  dimensionOutput.textContent = String(result.dimension);
  modelOutput.textContent = result.model;
  previewOutput.textContent = `[${result.preview.map((value) => value.toFixed(6)).join(', ')}, ...]`;
  copyButton.disabled = false;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  copyButton.disabled = true;
  currentEmbeddingJson = '';
  setBusy(true);
  setStatus('Running');

  try {
    const response = await fetch('/api/embed', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        text: textInput.value,
        modelDir: selectedModelDir(),
      }),
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || 'Embedding request failed.');
    }

    renderResult(payload);
    setStatus('Ready', 'success');
  } catch (error) {
    dimensionOutput.textContent = '-';
    modelOutput.textContent = '-';
    previewOutput.textContent = error instanceof Error ? error.message : 'Embedding request failed.';
    setStatus('Error', 'error');
  } finally {
    setBusy(false);
  }
});

modelForm.addEventListener('submit', async (event) => {
  event.preventDefault();

  setModelBusy(true);
  setStatus('Exporting');

  try {
    const response = await fetch('/api/models', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        modelId: hfModelInput.value,
        outputDir: outputDirInput.value,
      }),
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || 'Model export failed.');
    }

    await loadModels(payload.model.dir);
    setStatus('Model added', 'success');
  } catch (error) {
    previewOutput.textContent = error instanceof Error ? error.message : 'Model export failed.';
    setStatus('Error', 'error');
  } finally {
    setModelBusy(false);
  }
});

refreshModelsButton.addEventListener('click', async () => {
  try {
    await loadModels(selectedModelDir());
    setStatus('Ready', 'success');
  } catch (error) {
    previewOutput.textContent = error instanceof Error ? error.message : 'Could not load models.';
    setStatus('Error', 'error');
  }
});

copyButton.addEventListener('click', async () => {
  if (!currentEmbeddingJson) {
    return;
  }

  await navigator.clipboard.writeText(currentEmbeddingJson);
  setStatus('Copied', 'success');
});

loadModels().catch((error) => {
  generateButton.disabled = true;
  previewOutput.textContent = error instanceof Error ? error.message : 'Could not load models.';
  setStatus('Error', 'error');
});
