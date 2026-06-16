const form = document.querySelector('#embed-form');
const modelForm = document.querySelector('#model-form');
const textInput = document.querySelector('#text-input');
const modelSelect = document.querySelector('#model-select');
const hfModelInput = document.querySelector('#hf-model-input');
const outputDirInput = document.querySelector('#output-dir-input');
const generateButton = document.querySelector('#generate-button');
const addModelButton = document.querySelector('#add-model-button');
const refreshModelsButton = document.querySelector('#refresh-models-button');
const deleteModelButton = document.querySelector('#delete-model-button');
const copyButton = document.querySelector('#copy-button');
const statusOutput = document.querySelector('#status');
const dimensionOutput = document.querySelector('#dimension-output');
const modelOutput = document.querySelector('#model-output');
const previewOutput = document.querySelector('#preview-output');
const deleteDialog = document.querySelector('#delete-dialog');
const deleteDialogText = document.querySelector('#delete-dialog-text');
const deleteConfirmInput = document.querySelector('#delete-confirm-input');
const confirmDeleteButton = document.querySelector('#confirm-delete-button');

let currentEmbeddingJson = '';
let currentModels = [];
let isGenerating = false;

function selectedModelDir() {
  return modelSelect.value;
}

function selectedModel() {
  return currentModels.find((model) => model.dir === selectedModelDir()) || null;
}

function updateDeleteConfirmation() {
  const model = selectedModel();
  confirmDeleteButton.disabled = deleteConfirmInput.value !== model?.name;
}

function setStatus(message, type = '') {
  statusOutput.textContent = message;
  statusOutput.className = `status${type ? ` ${type}` : ''}`;
}

function setBusy(isBusy) {
  isGenerating = isBusy;
  generateButton.disabled = isBusy || !selectedModel();
  generateButton.textContent = isBusy ? 'Generating...' : 'Generate';
}

function setModelBusy(isBusy) {
  addModelButton.disabled = isBusy;
  refreshModelsButton.disabled = isBusy;
  deleteModelButton.disabled = isBusy || !selectedModel()?.deletable;
  addModelButton.textContent = isBusy ? 'Adding...' : 'Add Model';
}

function updateModelActions() {
  generateButton.disabled = isGenerating || !selectedModel();
  deleteModelButton.disabled = !selectedModel()?.deletable;
}

async function loadModels(preferredDir = '') {
  const response = await fetch('/api/models');
  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload.error || 'Could not load models.');
  }

  modelSelect.replaceChildren();
  currentModels = payload.models;

  if (payload.models.length === 0) {
    const option = document.createElement('option');
    option.value = '';
    option.textContent = 'No models available';
    modelSelect.append(option);
  }

  for (const model of payload.models) {
    const option = document.createElement('option');
    option.value = model.dir;
    option.textContent = model.deletable ? model.name : `${model.name} (read-only)`;
    option.dataset.deletable = String(Boolean(model.deletable));
    modelSelect.append(option);
  }

  if (preferredDir) {
    modelSelect.value = preferredDir;
  }

  if (!modelSelect.value && payload.models.length > 0) {
    modelSelect.value = payload.models[0].dir;
  }

  updateModelActions();

  if (payload.models.length === 0) {
    dimensionOutput.textContent = '-';
    modelOutput.textContent = '-';
    previewOutput.textContent = 'No local models found. Add a model under ./models before generating embeddings.';
    copyButton.disabled = true;
    currentEmbeddingJson = '';
  }
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

  try {
    if (!selectedModel()) {
      throw new Error('No local model is available. Add a model under ./models first.');
    }

    setBusy(true);
    setStatus('Running');

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

modelSelect.addEventListener('change', () => {
  updateModelActions();
});

deleteModelButton.addEventListener('click', () => {
  const model = selectedModel();

  if (!model?.deletable) {
    return;
  }

  deleteDialogText.textContent = `${model.dir} will be permanently removed from the mapped models volume. Type "${model.name}" exactly to confirm.`;
  deleteConfirmInput.value = '';
  confirmDeleteButton.disabled = true;
  deleteDialog.showModal();
  deleteConfirmInput.focus();
});

deleteConfirmInput.addEventListener('input', () => {
  updateDeleteConfirmation();
});

deleteDialog.addEventListener('close', () => {
  deleteConfirmInput.value = '';
  confirmDeleteButton.disabled = true;
});

confirmDeleteButton.addEventListener('click', async () => {
  const model = selectedModel();

  if (!model?.deletable) {
    deleteDialog.close();
    return;
  }

  if (deleteConfirmInput.value !== model.name) {
    updateDeleteConfirmation();
    return;
  }

  confirmDeleteButton.disabled = true;
  setStatus('Deleting');

  try {
    const response = await fetch('/api/models', {
      method: 'DELETE',
      headers: {
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        modelDir: model.dir,
      }),
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || 'Model delete failed.');
    }

    deleteDialog.close();
    await loadModels();
    dimensionOutput.textContent = '-';
    modelOutput.textContent = '-';
    previewOutput.textContent = `Deleted ${payload.deleted}.`;
    copyButton.disabled = true;
    currentEmbeddingJson = '';
    setStatus('Deleted', 'success');
  } catch (error) {
    previewOutput.textContent = error instanceof Error ? error.message : 'Model delete failed.';
    setStatus('Error', 'error');
  } finally {
    confirmDeleteButton.disabled = false;
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
