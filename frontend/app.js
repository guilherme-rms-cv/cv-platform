const apiBaseInput = document.getElementById("apiBase");
const fileInput = document.getElementById("fileInput");
const algorithmSelect = document.getElementById("algorithm");
const gaussianGroup = document.getElementById("gaussianGroup");
const sobelGroup = document.getElementById("sobelGroup");
const cannyGroup = document.getElementById("cannyGroup");
const kernelSizeInput = document.getElementById("kernelSize");
const sigmaInput = document.getElementById("sigma");
const sobelKernelSizeInput = document.getElementById("sobelKernelSize");
const cannyThreshold1Input = document.getElementById("cannyThreshold1");
const cannyThreshold2Input = document.getElementById("cannyThreshold2");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
const originalPreview = document.getElementById("originalPreview");
const resultPreview = document.getElementById("resultPreview");

function setStatus(message, type = "info") {
  statusEl.textContent = message;
  statusEl.className = `status ${type}`;
}

function updateGaussianVisibility() {
  const selected = algorithmSelect.value;
  gaussianGroup.hidden = selected !== "gaussian";
  sobelGroup.hidden = selected !== "sobel";
  cannyGroup.hidden = selected !== "canny";
}

function validateCannyParams() {
  const threshold1 = Number(cannyThreshold1Input.value);
  const threshold2 = Number(cannyThreshold2Input.value);

  if (Number.isNaN(threshold1) || threshold1 < 0 || threshold1 > 255) {
    throw new Error("threshold1 must be between 0 and 255");
  }

  if (Number.isNaN(threshold2) || threshold2 < 0 || threshold2 > 255) {
    throw new Error("threshold2 must be between 0 and 255");
  }

  if (threshold1 >= threshold2) {
    throw new Error("threshold1 must be lower than threshold2");
  }

  return { threshold1, threshold2 };
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

  if (algorithmSelect.value === "sobel") {
    const kernelSize = Number(sobelKernelSizeInput.value);
    if (!Number.isInteger(kernelSize) || ![1, 3, 5, 7].includes(kernelSize)) {
      throw new Error("kernel_size must be one of: 1, 3, 5, 7");
    }

    const query = new URLSearchParams({
      kernel_size: String(kernelSize),
    });

    return `${base}/api/v1/image-processing/edges/sobel?${query.toString()}`;
  }

  if (algorithmSelect.value === "canny") {
    const { threshold1, threshold2 } = validateCannyParams();
    const query = new URLSearchParams({
      threshold1: String(threshold1),
      threshold2: String(threshold2),
    });

    return `${base}/api/v1/image-processing/edges/canny?${query.toString()}`;
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
    const algoLabel =
      algorithmSelect.value === "sobel"
        ? "Sobel"
        : algorithmSelect.value === "canny"
          ? "Canny"
        : algorithmSelect.value === "gaussian"
          ? "Gaussian Blur"
          : "Grayscale";
    setStatus(`Done using ${algoLabel}. Compare original and result side by side.`, "ok");
  } catch (error) {
    setStatus(error.message || "Unexpected error while processing.", "error");
  } finally {
    runBtn.disabled = false;
  }
}

algorithmSelect.addEventListener("change", updateGaussianVisibility);
runBtn.addEventListener("click", runAlgorithm);
window.addEventListener("pageshow", updateGaussianVisibility);

updateGaussianVisibility();
setStatus("Ready. Upload an image and run an algorithm.");
