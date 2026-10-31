/**
 * =============================================================================
 * Facial Emotion Recognition - Main UI & Image Upload Controller
 * =============================================================================
 */

// Emotion to Emoji Dictionary
const EMOTION_EMOJIS = {
  "Angry": "😡",
  "Disgust": "🤢",
  "Fear": "😨",
  "Happy": "😄",
  "Neutral": "😐",
  "Sad": "😢",
  "Surprise": "😲"
};

// Emotion to Color Mapping for Canvas Bounding Boxes
const EMOTION_COLORS = {
  "Angry": "#ef4444",
  "Disgust": "#10b981",
  "Fear": "#8b5cf6",
  "Happy": "#22c55e",
  "Neutral": "#64748b",
  "Sad": "#3b82f6",
  "Surprise": "#f59e0b"
};

// State Variables
let currentFacesData = [];
let selectedFaceIndex = 0;

// DOM Elements
const tabImageMode = document.getElementById("tab-image-mode");
const tabWebcamMode = document.getElementById("tab-webcam-mode");
const imageModeSection = document.getElementById("image-mode-section");
const webcamModeSection = document.getElementById("webcam-mode-section");

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const imageStage = document.getElementById("image-stage");
const imagePreview = document.getElementById("image-preview");
const imageCanvas = document.getElementById("image-overlay-canvas");
const btnUploadAnother = document.getElementById("btn-upload-another");
const faceCountTag = document.getElementById("face-count-tag");
const statusMessage = document.getElementById("status-message");

const primaryEmoji = document.getElementById("primary-emoji");
const primaryEmotionName = document.getElementById("primary-emotion-name");
const primaryConfidence = document.getElementById("primary-confidence");
const resultStatusPill = document.getElementById("result-status-pill");
const multiFaceSelector = document.getElementById("multi-face-selector");
const faceSelectDropdown = document.getElementById("face-select-dropdown");

// =============================================================================
// 1. TAB NAVIGATION
// =============================================================================
tabImageMode.addEventListener("click", () => {
  tabImageMode.classList.add("active");
  tabWebcamMode.classList.remove("active");
  imageModeSection.classList.add("active");
  webcamModeSection.classList.remove("active");

  // If webcam is active, stop it cleanly
  if (typeof window.stopWebcamStream === "function") {
    window.stopWebcamStream();
  }
  clearStatusMessage();
});

tabWebcamMode.addEventListener("click", () => {
  tabWebcamMode.classList.add("active");
  tabImageMode.classList.remove("active");
  webcamModeSection.classList.add("active");
  imageModeSection.classList.remove("active");
  clearStatusMessage();
});

// =============================================================================
// 2. IMAGE UPLOAD & DRAG-AND-DROP
// =============================================================================
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    handleImageSelection(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files.length > 0) {
    handleImageSelection(e.target.files[0]);
  }
});

btnUploadAnother.addEventListener("click", () => {
  imageStage.classList.add("hidden");
  dropZone.classList.remove("hidden");
  fileInput.value = "";
  clearCanvas(imageCanvas);
  resetResultsPanel();
  clearStatusMessage();
});

function handleImageSelection(file) {
  // Validate file type
  const validTypes = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
  if (!validTypes.includes(file.type)) {
    showStatusMessage("Please select a valid image file (JPEG, PNG, WEBP).", "error");
    return;
  }

  // Validate size (10 MB max)
  if (file.size > 10 * 1024 * 1024) {
    showStatusMessage("Image file is too large. Maximum size is 10 MB.", "error");
    return;
  }

  clearStatusMessage();
  resultStatusPill.textContent = "Analyzing...";

  const reader = new FileReader();
  reader.onload = (event) => {
    imagePreview.src = event.target.result;
    imagePreview.onload = () => {
      dropZone.classList.add("hidden");
      imageStage.classList.remove("hidden");
      sendImageForPrediction(file);
    };
  };
  reader.readAsDataURL(file);
}

// =============================================================================
// 3. SEND IMAGE TO /predict ENDPOINT
// =============================================================================
async function sendImageForPrediction(file) {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      showStatusMessage(data.error || "Failed to analyze image.", "error");
      resultStatusPill.textContent = "Error";
      resetResultsPanel();
      return;
    }

    handlePredictionResults(data);
  } catch (error) {
    console.error("Prediction fetch error:", error);
    showStatusMessage("Network error or server unavailable. Is 'python app.py' running?", "error");
    resultStatusPill.textContent = "Offline";
    resetResultsPanel();
  }
}

// =============================================================================
// 4. PROCESS RESULTS & DRAW BOUNDING BOXES
// =============================================================================
function handlePredictionResults(data) {
  currentFacesData = data.faces || [];
  const faceCount = data.faces_detected || 0;

  faceCountTag.textContent = `${faceCount} Face${faceCount === 1 ? "" : "s"}`;
  resultStatusPill.textContent = `Completed (${faceCount} found)`;

  if (faceCount === 0) {
    clearCanvas(imageCanvas);
    showStatusMessage(data.message || "No faces detected. Ensure the face is clearly visible and well lit.", "info");
    resetResultsPanel("No Face Found");
    return;
  }

  // Draw bounding boxes on canvas
  drawBoundingBoxesOnImage(imagePreview, imageCanvas, currentFacesData);

  // Setup multi-face selector if more than one face
  if (faceCount > 1) {
    multiFaceSelector.classList.remove("hidden");
    faceSelectDropdown.innerHTML = "";
    currentFacesData.forEach((f, idx) => {
      const opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = `Face #${idx + 1} (${f.emotion} - ${f.confidence}%)`;
      faceSelectDropdown.appendChild(opt);
    });
    faceSelectDropdown.onchange = (e) => {
      selectedFaceIndex = parseInt(e.target.value, 10);
      updateEmotionDisplay(currentFacesData[selectedFaceIndex]);
    };
  } else {
    multiFaceSelector.classList.add("hidden");
  }

  selectedFaceIndex = 0;
  updateEmotionDisplay(currentFacesData[0]);
}

/**
 * Draws bounding boxes, labels, and confidence tags on top of an image.
 */
function drawBoundingBoxesOnImage(imgElement, canvas, faces) {
  const naturalW = imgElement.naturalWidth;
  const naturalH = imgElement.naturalHeight;
  const displayW = imgElement.clientWidth;
  const displayH = imgElement.clientHeight;

  canvas.width = displayW;
  canvas.height = displayH;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const scaleX = displayW / naturalW;
  const scaleY = displayH / naturalH;

  faces.forEach((face, idx) => {
    const { x, y, width, height } = face.box;
    const drawX = x * scaleX;
    const drawY = y * scaleY;
    const drawW = width * scaleX;
    const drawH = height * scaleY;

    const color = EMOTION_COLORS[face.emotion] || "#2563eb";

    // 1. Draw Bounding Box Rectangle
    ctx.lineWidth = 3;
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.roundRect(drawX, drawY, drawW, drawH, 6);
    ctx.stroke();

    // 2. Draw Emotion Tag Pill above box
    const label = `${face.emotion} ${face.confidence}%`;
    ctx.font = "bold 13px Inter, sans-serif";
    const textMetrics = ctx.measureText(label);
    const textW = textMetrics.width;
    const textH = 20;

    const labelX = drawX;
    const labelY = Math.max(drawY - textH - 4, 4);

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(labelX, labelY, textW + 12, textH, 4);
    ctx.fill();

    // 3. Draw Tag Text
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, labelX + 6, labelY + 14);
  });
}

// =============================================================================
// 5. UPDATE PROBABILITY BARS & SPOTLIGHT CARD
// =============================================================================
function updateEmotionDisplay(face) {
  if (!face) return;

  const emotion = face.emotion;
  const confidence = face.confidence;
  const probabilities = face.probabilities || {};

  // Update spotlight card
  primaryEmoji.textContent = EMOTION_EMOJIS[emotion] || "😐";
  primaryEmotionName.textContent = emotion;
  primaryConfidence.textContent = `${confidence.toFixed(1)}%`;

  // Update probability progress bars
  const emotionsList = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"];
  emotionsList.forEach((emo) => {
    const valSpan = document.getElementById(`val-${emo}`);
    const barDiv = document.getElementById(`bar-${emo}`);
    if (valSpan && barDiv) {
      const prob = probabilities[emo] !== undefined ? probabilities[emo] : 0.0;
      valSpan.textContent = `${prob.toFixed(1)}%`;
      barDiv.style.width = `${Math.min(prob, 100)}%`;
    }
  });
}

function resetResultsPanel(customMessage = "Awaiting Input") {
  primaryEmoji.textContent = "😐";
  primaryEmotionName.textContent = "Neutral";
  primaryConfidence.textContent = "0.0%";
  multiFaceSelector.classList.add("hidden");

  const emotionsList = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"];
  emotionsList.forEach((emo) => {
    const valSpan = document.getElementById(`val-${emo}`);
    const barDiv = document.getElementById(`bar-${emo}`);
    if (valSpan && barDiv) {
      valSpan.textContent = "0.0%";
      barDiv.style.width = "0%";
    }
  });
}

function clearCanvas(canvas) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function showStatusMessage(text, type = "info") {
  statusMessage.textContent = text;
  statusMessage.className = `status-banner ${type}`;
  statusMessage.classList.remove("hidden");
}

function clearStatusMessage() {
  statusMessage.textContent = "";
  statusMessage.className = "status-banner hidden";
}

// Window resize handler to reposition canvas bounding boxes if viewport changes
window.addEventListener("resize", () => {
  if (imageModeSection.classList.contains("active") && currentFacesData.length > 0 && !imageStage.classList.contains("hidden")) {
    drawBoundingBoxesOnImage(imagePreview, imageCanvas, currentFacesData);
  }
});

// Export helper for webcam module
window.updateEmotionDisplay = updateEmotionDisplay;
window.resetResultsPanel = resetResultsPanel;
window.showStatusMessage = showStatusMessage;
window.clearStatusMessage = clearStatusMessage;
