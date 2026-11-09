"use strict";

const imageInput = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const previewWrap = document.querySelector("#preview-wrap");
const imagePreview = document.querySelector("#image-preview");
const canvas = document.querySelector("#detection-canvas");
const confidenceInput = document.querySelector("#confidence");
const confidenceValue = document.querySelector("#confidence-value");
const detectButton = document.querySelector("#detect-button");
const buttonLabel = document.querySelector("#button-label");
const spinner = document.querySelector("#spinner");
const message = document.querySelector("#message");
const results = document.querySelector("#results");
const emptyResults = document.querySelector("#empty-results");
const resultStatus = document.querySelector("#result-status");
const detectionList = document.querySelector("#detection-list");
const removeImage = document.querySelector("#remove-image");

let selectedFile = null;
let previewUrl = null;

function setConfidenceTrack() {
  const percent = Number(confidenceInput.value) * 100;
  confidenceInput.style.background = `linear-gradient(to right, var(--accent) ${percent}%, #354155 ${percent}%)`;
  confidenceValue.value = Number(confidenceInput.value).toFixed(2);
}

function clearCanvas() {
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
}

function resetResults() {
  results.hidden = true;
  emptyResults.hidden = false;
  resultStatus.textContent = "WAITING";
  resultStatus.classList.remove("ready");
  detectionList.replaceChildren();
  clearCanvas();
}

function showFile(file) {
  message.textContent = "";
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    message.textContent = "Please choose a valid image file.";
    return;
  }
  if (file.size === 0) {
    message.textContent = "This file is empty. Choose another image.";
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    message.textContent = "Image is larger than 10 MB. Choose a smaller file.";
    return;
  }

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  imagePreview.src = previewUrl;
  imagePreview.onload = () => {
    canvas.width = imagePreview.naturalWidth;
    canvas.height = imagePreview.naturalHeight;
    clearCanvas();
  };
  imagePreview.onerror = () => {
    selectedFile = null;
    message.textContent = "The selected file could not be read as an image.";
    previewWrap.hidden = true;
    dropZone.hidden = false;
  };
  previewWrap.hidden = false;
  dropZone.hidden = true;
  detectButton.disabled = false;
  resetResults();
}

function drawDetections(detections) {
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.lineWidth = Math.max(2, canvas.width / 350);
  context.font = `600 ${Math.max(12, canvas.width / 60)}px sans-serif`;
  for (const [index, item] of detections.entries()) {
    const { x1, y1, x2, y2 } = item.bbox;
    const color = `hsl(${(index * 67 + 78) % 360} 88% 62%)`;
    context.strokeStyle = color;
    context.fillStyle = color;
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `${item.class_name} ${(item.confidence * 100).toFixed(1)}%`;
    const labelHeight = Math.max(20, canvas.width / 35);
    const textWidth = context.measureText(label).width;
    const labelY = Math.max(labelHeight, y1);
    context.fillRect(x1, labelY - labelHeight, textWidth + 12, labelHeight);
    context.fillStyle = "#101709";
    context.fillText(label, x1 + 6, labelY - labelHeight / 2, textWidth);
  }
}

function renderDetections(data) {
  results.hidden = false;
  emptyResults.hidden = true;
  resultStatus.textContent = "COMPLETE";
  resultStatus.classList.add("ready");
  document.querySelector("#object-count").textContent = data.count;
  document.querySelector("#inference-time").textContent = Number(data.inference_time_ms).toFixed(1);
  detectionList.replaceChildren();
  const noDetections = document.querySelector("#no-detections");
  noDetections.hidden = data.detections.length > 0;
  for (const item of data.detections) {
    const row = document.createElement("li");
    row.className = "detection-item";
    const name = document.createElement("span");
    name.className = "detection-name";
    const dot = document.createElement("span");
    dot.className = "detection-dot";
    dot.setAttribute("aria-hidden", "true");
    const className = document.createElement("span");
    className.textContent = item.class_name;
    name.append(dot, className);
    const score = document.createElement("span");
    score.className = "detection-confidence";
    score.textContent = `${(item.confidence * 100).toFixed(1)}%`;
    const coords = document.createElement("span");
    coords.className = "detection-coordinates";
    const { x1, y1, x2, y2 } = item.bbox;
    coords.textContent = `x1 ${x1} · y1 ${y1} · x2 ${x2} · y2 ${y2}`;
    row.append(name, score, coords);
    detectionList.append(row);
  }
  drawDetections(data.detections);
}

dropZone.addEventListener("click", () => imageInput.click());
imageInput.addEventListener("change", (event) => showFile(event.target.files[0]));
confidenceInput.addEventListener("input", setConfidenceTrack);
removeImage.addEventListener("click", () => {
  selectedFile = null;
  imageInput.value = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  imagePreview.removeAttribute("src");
  previewWrap.hidden = true;
  dropZone.hidden = false;
  detectButton.disabled = true;
  message.textContent = "";
  resetResults();
});

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
}
dropZone.addEventListener("drop", (event) => showFile(event.dataTransfer.files[0]));

detectButton.addEventListener("click", async () => {
  if (!selectedFile) return;
  message.textContent = "";
  detectButton.disabled = true;
  buttonLabel.textContent = "Detecting objects";
  spinner.hidden = false;
  resultStatus.textContent = "RUNNING";
  resultStatus.classList.remove("ready");
  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("confidence", confidenceInput.value);
  try {
    const response = await fetch("/detect", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Detection failed. Please try again.");
    renderDetections(data);
  } catch (error) {
    resultStatus.textContent = "ERROR";
    message.textContent = error.message || "Could not connect to the local detection server.";
  } finally {
    detectButton.disabled = false;
    buttonLabel.textContent = "Detect objects";
    spinner.hidden = true;
  }
});

setConfidenceTrack();
