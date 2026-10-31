/**
 * =============================================================================
 * Facial Emotion Recognition - Live Webcam Stream & Inference Controller
 * =============================================================================
 */

(function () {
  // DOM Elements
  const video = document.getElementById("webcam-video");
  const overlayCanvas = document.getElementById("webcam-overlay-canvas");
  const btnStartWebcam = document.getElementById("btn-start-webcam");
  const btnStopWebcam = document.getElementById("btn-stop-webcam");
  const liveIndicator = document.getElementById("webcam-live-indicator");
  const placeholder = document.getElementById("webcam-placeholder");
  const fpsPill = document.getElementById("fps-pill");
  const resultStatusPill = document.getElementById("result-status-pill");

  // State Variables
  let mediaStream = null;
  let captureIntervalId = null;
  let isProcessingFrame = false;
  const CAPTURE_INTERVAL_MS = 650; // Modest interval (approx. 1.5 FPS) to prevent lag and network congestion

  // Offscreen capture canvas
  const captureCanvas = document.createElement("canvas");
  const captureCtx = captureCanvas.getContext("2d");

  // ===========================================================================
  // 1. START WEBCAM STREAM
  // ===========================================================================
  btnStartWebcam.addEventListener("click", startWebcam);
  btnStopWebcam.addEventListener("click", stopWebcamStream);

  async function startWebcam() {
    window.clearStatusMessage();
    fpsPill.classList.remove("hidden");
    fpsPill.textContent = "Connecting to camera...";

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      window.showStatusMessage("Camera API (getUserMedia) is not supported in this browser.", "error");
      fpsPill.classList.add("hidden");
      return;
    }

    try {
      // Request video stream with preferred 640x480 resolution
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user"
        },
        audio: false
      });

      video.srcObject = mediaStream;

      video.onloadedmetadata = () => {
        video.play();
        placeholder.classList.add("hidden");
        liveIndicator.classList.remove("hidden");
        btnStartWebcam.classList.add("hidden");
        btnStopWebcam.classList.remove("hidden");
        resultStatusPill.textContent = "Live Stream Active";
        fpsPill.textContent = "Streaming...";

        // Begin periodic capture loop
        captureIntervalId = setInterval(captureAndSendFrame, CAPTURE_INTERVAL_MS);
      };

    } catch (err) {
      console.error("Webcam access error:", err);
      let errorMsg = "Unable to access camera.";
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        errorMsg = "Camera permission denied. Please allow camera access in your browser address bar.";
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        errorMsg = "No webcam device detected on your computer.";
      } else if (err.name === "NotReadableError") {
        errorMsg = "Camera is already in use by another application.";
      }
      window.showStatusMessage(errorMsg, "error");
      fpsPill.classList.add("hidden");
    }
  }

  // ===========================================================================
  // 2. CAPTURE FRAME & SEND TO /webcam
  // ===========================================================================
  async function captureAndSendFrame() {
    // If previous request is still inflight or video not ready, skip this frame
    if (isProcessingFrame || !video.videoWidth || !video.videoHeight) {
      return;
    }

    isProcessingFrame = true;
    const startTime = performance.now();

    try {
      // Set capture canvas to match video stream resolution
      captureCanvas.width = video.videoWidth;
      captureCanvas.height = video.videoHeight;
      captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

      // Extract base64 JPEG
      const base64Data = captureCanvas.toDataURL("image/jpeg", 0.82);

      const response = await fetch("/webcam", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64Data })
      });

      const data = await response.json();
      const latency = Math.round(performance.now() - startTime);
      fpsPill.textContent = `${latency} ms latency`;

      if (!response.ok || !data.success) {
        if (data.error) {
          window.showStatusMessage(data.error, "error");
        }
        clearOverlayCanvas();
        return;
      }

      handleWebcamResults(data);

    } catch (err) {
      console.warn("Webcam frame error:", err);
      fpsPill.textContent = "Reconnecting...";
    } finally {
      isProcessingFrame = false;
    }
  }

  // ===========================================================================
  // 3. DRAW REAL-TIME BOUNDING BOXES ON OVERLAY
  // ===========================================================================
  function handleWebcamResults(data) {
    const faces = data.faces || [];
    const faceCount = data.faces_detected || 0;

    // Match overlay canvas size to displayed video element
    const displayW = video.clientWidth;
    const displayH = video.clientHeight;

    overlayCanvas.width = displayW;
    overlayCanvas.height = displayH;

    const ctx = overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, displayW, displayH);

    if (faceCount === 0) {
      resultStatusPill.textContent = "Live: Looking for face...";
      return;
    }

    resultStatusPill.textContent = `Live: ${faceCount} Face(s)`;

    const scaleX = displayW / video.videoWidth;
    const scaleY = displayH / video.videoHeight;

    const EMOTION_COLORS = {
      "Angry": "#ef4444",
      "Disgust": "#10b981",
      "Fear": "#8b5cf6",
      "Happy": "#22c55e",
      "Neutral": "#64748b",
      "Sad": "#3b82f6",
      "Surprise": "#f59e0b"
    };

    faces.forEach((face) => {
      const { x, y, width, height } = face.box;
      const drawX = x * scaleX;
      const drawY = y * scaleY;
      const drawW = width * scaleX;
      const drawH = height * scaleY;

      const color = EMOTION_COLORS[face.emotion] || "#2563eb";

      // 1. Draw glowing box
      ctx.lineWidth = 3;
      ctx.strokeStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 6;
      ctx.beginPath();
      ctx.roundRect(drawX, drawY, drawW, drawH, 6);
      ctx.stroke();
      ctx.shadowBlur = 0; // reset shadow

      // 2. Draw emotion tag pill
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

      // 3. Label text
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, labelX + 6, labelY + 14);
    });

    // Update dominant emotion in the sidebar for the first face
    if (faces.length > 0 && typeof window.updateEmotionDisplay === "function") {
      window.updateEmotionDisplay(faces[0]);
    }
  }

  // ===========================================================================
  // 4. STOP WEBCAM STREAM CLEANLY
  // ===========================================================================
  function stopWebcamStream() {
    if (captureIntervalId) {
      clearInterval(captureIntervalId);
      captureIntervalId = null;
    }

    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
    }

    video.srcObject = null;
    clearOverlayCanvas();

    placeholder.classList.remove("hidden");
    liveIndicator.classList.add("hidden");
    btnStopWebcam.classList.add("hidden");
    btnStartWebcam.classList.remove("hidden");
    fpsPill.classList.add("hidden");
    resultStatusPill.textContent = "Webcam Stopped";

    if (typeof window.resetResultsPanel === "function") {
      window.resetResultsPanel();
    }
  }

  function clearOverlayCanvas() {
    const ctx = overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  }

  // Expose stop function globally for tab navigation
  window.stopWebcamStream = stopWebcamStream;
})();
