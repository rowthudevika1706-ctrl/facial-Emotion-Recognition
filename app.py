"""
=============================================================================
Facial Emotion Recognition - Flask Backend Web Application
=============================================================================
Routes:
  GET  /        - Serves the frontend user dashboard
  POST /predict - Accepts multipart/form-data image upload, returns JSON predictions
  POST /webcam  - Accepts JSON with base64 encoded webcam frame, returns JSON predictions

Face Detection:
  OpenCV Haar Cascade (haarcascade_frontalface_default.xml)
Model Inference:
  TensorFlow/Keras CNN (models/emotion_model.keras)
"""

import os
import sys
import io
import json
import base64
import logging
import numpy as np
import cv2

# Ensure Windows console handles UTF-8 output safely
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Set TensorFlow log level to reduce console clutter
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf = None
try:
    import tensorflow as tf
except ImportError:
    tf = None

from flask import Flask, render_template, request, jsonify

# =============================================================================
# 1. APPLICATION SETUP & CONFIGURATION
# =============================================================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB maximum upload limit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FER-App")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

# Default fallback emotion classes if class_names.json is missing
DEFAULT_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# =============================================================================
# 2. LOAD MODEL & FACE DETECTOR AT STARTUP (LOAD ONCE)
# =============================================================================
face_model = None
class_names = DEFAULT_CLASSES
model_loaded = False

# Initialize OpenCV Haar Cascade Frontal Face Classifier
haar_xml_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_xml_path)

if face_cascade.empty():
    logger.error("Failed to load OpenCV Haar Cascade from: %s", haar_xml_path)
else:
    logger.info("OpenCV Haar Cascade successfully loaded from: %s", haar_xml_path)

def load_emotion_model():
    """
    Loads the trained Keras CNN model and class names dictionary.
    Called once during startup.
    """
    global face_model, class_names, model_loaded

    if tf is None:
        logger.warning(
            "TensorFlow is not installed in this environment. "
            "Please run 'pip install -r requirements.txt' in a Python 3.10/3.11 virtual environment."
        )
        model_loaded = False
        return

    if os.path.exists(CLASS_NAMES_PATH):
        try:
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                class_names = json.load(f)
            logger.info("Loaded class labels: %s", class_names)
        except Exception as e:
            logger.warning("Could not read class_names.json: %s. Using default labels.", e)

    if os.path.exists(MODEL_PATH):
        try:
            face_model = tf.keras.models.load_model(MODEL_PATH)
            model_loaded = True
            logger.info("Emotion recognition model loaded successfully from: %s", MODEL_PATH)
            # Warm-up inference with dummy input to compile computation graph
            dummy_input = np.zeros((1, 48, 48, 1), dtype=np.float32)
            face_model.predict(dummy_input, verbose=0)
            logger.info("Model warm-up inference completed.")
        except Exception as e:
            logger.error("Failed to load model from %s: %s", MODEL_PATH, e)
            model_loaded = False
    else:
        logger.warning(
            "Model file not found at '%s'. Please run 'python train.py' to train and save the model.",
            MODEL_PATH
        )
        model_loaded = False

# Execute model loading at startup
load_emotion_model()


# =============================================================================
# 3. PREPROCESSING & INFERENCE PIPELINE
# =============================================================================
def allowed_file(filename):
    """Checks whether the file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_image_and_detect_emotions(image_bgr):
    """
    Core pipeline:
    1. Converts input BGR image to Grayscale.
    2. Detects all face bounding boxes via Haar Cascade.
    3. For each face:
       - Crops face Region of Interest (ROI)
       - Resizes ROI to 48x48 pixels
       - Rescales pixels to [0.0, 1.0] (matching train.py exactly)
       - Reshapes to (1, 48, 48, 1)
       - Runs CNN inference
       - Computes confidence % and all 7 emotion probabilities.
    """
    if image_bgr is None:
        return {"success": False, "error": "Invalid or unreadable image data.", "status_code": 400}

    img_h, img_w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return {
            "success": True,
            "faces_detected": 0,
            "faces": [],
            "image_dimensions": {"width": img_w, "height": img_h},
            "message": "No face detected. Please ensure the face is well-lit and facing the camera."
        }

    detected_faces_data = []

    for (x, y, w, h) in faces:
        # Preprocessing matching train.py
        roi_gray = gray[y:y + h, x:x + w]

        # Resize to 48x48
        roi_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Normalize pixel values to 0.0 - 1.0
        roi_normalized = roi_resized.astype("float32") / 255.0

        # Reshape to (1, 48, 48, 1) for Keras CNN input
        roi_tensor = np.expand_dims(roi_normalized, axis=(0, -1))

        # Model prediction
        probabilities = face_model.predict(roi_tensor, verbose=0)[0]

        top_idx = int(np.argmax(probabilities))
        top_emotion = class_names[top_idx]
        confidence = float(probabilities[top_idx] * 100.0)

        # Build dictionary of all 7 emotion probabilities
        prob_dict = {
            class_names[i]: float(round(probabilities[i] * 100.0, 2))
            for i in range(len(class_names))
        }

        detected_faces_data.append({
            "box": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            },
            "emotion": top_emotion,
            "confidence": round(confidence, 2),
            "probabilities": prob_dict
        })

    return {
        "success": True,
        "faces_detected": len(detected_faces_data),
        "faces": detected_faces_data,
        "image_dimensions": {"width": img_w, "height": img_h},
        "message": f"Successfully detected {len(detected_faces_data)} face(s)."
    }


# =============================================================================
# 4. FLASK ROUTES
# =============================================================================
@app.route("/", methods=["GET"])
def index():
    """Serves the main web application UI."""
    return render_template(
        "index.html",
        model_loaded=model_loaded,
        tf_installed=(tf is not None),
        class_names=class_names
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts uploaded image via multipart/form-data.
    Returns JSON containing face coordinates and emotion predictions.
    """
    if not model_loaded:
        return jsonify({
            "success": False,
            "error": "Model not loaded. Please train the model first by running 'python train.py'."
        }), 503

    if "file" not in request.files and "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No file uploaded. Please send an image under the 'file' or 'image' field."
        }), 400

    file = request.files.get("file") or request.files.get("image")

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "No image selected. Please choose a valid image file."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Invalid file format '{file.filename}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}."
        }), 400

    try:
        # Read uploaded image bytes into OpenCV BGR image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            return jsonify({
                "success": False,
                "error": "Corrupted or unreadable image file."
            }), 400

        result = analyze_image_and_detect_emotions(image_bgr)
        return jsonify(result), (200 if result.get("success") else result.get("status_code", 400))

    except Exception as e:
        logger.error("Error processing /predict: %s", e, exc_info=True)
        return jsonify({
            "success": False,
            "error": f"An error occurred while analyzing the image: {str(e)}"
        }), 500


@app.route("/webcam", methods=["POST"])
def webcam():
    """
    Accepts base64 encoded frame from webcam stream via JSON POST.
    Payload: {"image": "data:image/jpeg;base64,..."}
    Returns JSON containing face coordinates and emotion predictions.
    """
    if not model_loaded:
        return jsonify({
            "success": False,
            "error": "Model not loaded. Please train the model first by running 'python train.py'."
        }), 503

    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return jsonify({
            "success": False,
            "error": "Missing 'image' key in JSON payload."
        }), 400

    image_data = payload.get("image", "")

    # Strip data URL header if present (e.g., 'data:image/jpeg;base64,')
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        decoded_bytes = base64.b64decode(image_data)
        file_bytes = np.frombuffer(decoded_bytes, np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            return jsonify({
                "success": False,
                "error": "Could not decode webcam frame."
            }), 400

        result = analyze_image_and_detect_emotions(image_bgr)
        return jsonify(result), (200 if result.get("success") else result.get("status_code", 400))

    except Exception as e:
        logger.error("Error processing /webcam: %s", e, exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Webcam processing error: {str(e)}"
        }), 500


# Error Handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "success": False,
        "error": "Uploaded file is too large. Maximum size allowed is 10 MB."
    }), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found."}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "error": "Internal server error."}), 500


# =============================================================================
# 5. SERVER ENTRYPOINT
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("      FACIAL EMOTION RECOGNITION WEB APPLICATION RUNNING        ")
    print("=" * 65)
    print("Local URL: http://127.0.0.1:5000")
    if not model_loaded:
        print("[!] WARNING: Model not detected at models/emotion_model.keras.")
        print("    Remember to run 'python train.py' to generate the model file.")
    else:
        print("[OK] Model loaded and ready to serve predictions!")
    print("=" * 65 + "\n")

    app.run(host="127.0.0.1", port=5000, debug=True)
