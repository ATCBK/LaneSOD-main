const form = document.getElementById("predict-form");
const fileInput = document.getElementById("image-input");
const sourcePreview = document.getElementById("source-preview");
const sourcePlaceholder = document.getElementById("source-placeholder");
const resultPreview = document.getElementById("result-preview");
const resultPlaceholder = document.getElementById("result-placeholder");
const statusText = document.getElementById("form-status");

function setStatus(message, type = "") {
  statusText.textContent = message;
  statusText.className = "status-text";
  if (type) {
    statusText.classList.add(`is-${type}`);
  }
}

function clearImagePreview(imageElement, placeholderElement) {
  imageElement.hidden = true;
  imageElement.removeAttribute("src");
  imageElement.onload = null;
  imageElement.onerror = null;
  placeholderElement.hidden = false;
}

function waitForImageLoad(imageElement, resultUrl) {
  clearImagePreview(imageElement, resultPlaceholder);

  return new Promise((resolve, reject) => {
    imageElement.onload = () => {
      imageElement.onload = null;
      imageElement.onerror = null;
      imageElement.hidden = false;
      resultPlaceholder.hidden = true;
      resolve();
    };

    imageElement.onerror = () => {
      imageElement.onload = null;
      imageElement.onerror = null;
      clearImagePreview(imageElement, resultPlaceholder);
      reject(new Error("Failed to load the generated result image."));
    };

    imageElement.src = resultUrl;
  });
}

fileInput?.addEventListener("change", () => {
  const [file] = fileInput.files;
  if (!file) {
    clearImagePreview(sourcePreview, sourcePlaceholder);
    clearImagePreview(resultPreview, resultPlaceholder);
    return;
  }

  const previewUrl = URL.createObjectURL(file);
  sourcePreview.src = previewUrl;
  sourcePreview.hidden = false;
  sourcePlaceholder.hidden = true;

  clearImagePreview(resultPreview, resultPlaceholder);
  setStatus("Image selected. Ready to run inference.");
});

form?.addEventListener("submit", async (event) => {
  event.preventDefault();

  const [file] = fileInput.files;
  if (!file) {
    clearImagePreview(resultPreview, resultPlaceholder);
    setStatus("Please select an image first.", "error");
    return;
  }

  const formData = new FormData();
  formData.append("image", file);

  clearImagePreview(resultPreview, resultPlaceholder);
  setStatus("Running inference. CPU execution may take a moment.");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Inference failed.");
    }

    await waitForImageLoad(resultPreview, payload.result_url);
    setStatus("Inference complete. Result image updated.", "success");
  } catch (error) {
    clearImagePreview(resultPreview, resultPlaceholder);
    setStatus(error.message || "Inference failed.", "error");
  }
});
