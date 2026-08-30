const apiBaseInput = document.getElementById("apiBase");
const fileInput = document.getElementById("fileInput");
const algorithmSelect = document.getElementById("algorithm");
const gaussianGroup = document.getElementById("gaussianGroup");
const kernelSizeInput = document.getElementById("kernelSize");
const sigmaInput = document.getElementById("sigma");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
const originalPreview = document.getElementById("originalPreview");
const resultPreview = document.getElementById("resultPreview");

function setStatus(message, type = "info") {
  statusEl.textContent = message;
  statusEl.className = `status ${type}`;
}

function updateGaussianVisibility() {
  gaussianGroup.hidden = algorithmSelect.value !== "gaussian";
}

function validateGaussianParams() {
  const kernelSize = Number(kernelSizeInput.value);
  const sigma = Number(sigmaInput.value);

  if (!Number.isInteger(kernelSize) || kernelSize <= 1 || kernelSize % 2 === 0) {
    throw new Error("kernel_size must be an odd integer greater than 1");
  }

  if (Number.isNaN(sigma) || sigma < 0) {
    throw new Error("sigma must be a float greater than or equal to 0");
  }

  return { kernelSize, sigma };
}

function buildEndpoint() {
  const base = apiBaseInput.value.replace(/\/$/, "");

  if (algorithmSelect.value === "grayscale") {
    return `${base}/api/v1/image-processing/grayscale`;
  }

  const { kernelSize, sigma } = validateGaussianParams();
  const query = new URLSearchParams({
    kernel_size: String(kernelSize),
    sigma: String(sigma),
  });

  return `${base}/api/v1/image-processing/blur/gaussian?${query.toString()}`;
}

async function runAlgorithm() {
  const file = fileInput.files?.[0];
  if (!file) {
    setStatus("Select an image before running an algorithm.", "error");
    return;
  }

  originalPreview.src = URL.createObjectURL(file);
  resultPreview.removeAttribute("src");

  let endpoint;
  try {
    endpoint = buildEndpoint();
  } catch (error) {
    setStatus(error.message, "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  runBtn.disabled = true;
  setStatus("Processing image...", "info");

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let detail = "Request failed";
      try {
        const payload = await response.json();
        detail = payload.detail || detail;
      } catch {
        // Keep default detail when server does not return JSON
      }
      throw new Error(detail);
    }

    const resultBlob = await response.blob();
    resultPreview.src = URL.createObjectURL(resultBlob);
    setStatus("Done. Compare original and result side by side.", "ok");
  } catch (error) {
    setStatus(error.message || "Unexpected error while processing.", "error");
  } finally {
    runBtn.disabled = false;
  }
}

algorithmSelect.addEventListener("change", updateGaussianVisibility);
runBtn.addEventListener("click", runAlgorithm);

updateGaussianVisibility();
setStatus("Ready. Upload an image and run an algorithm.");
