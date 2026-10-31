# Facial Emotion Recognition Web Application
### College-Level Deep Learning Assignment • Python, TensorFlow/Keras, OpenCV & Flask

A complete, beginner-friendly web application that detects human faces in uploaded images or live webcam video and classifies their facial expressions across **7 emotion categories**:
**Angry**, **Disgust**, **Fear**, **Happy**, **Neutral**, **Sad**, and **Surprise**.

The system pairs an **OpenCV Haar Cascade** frontal-face detector with a custom **Convolutional Neural Network (CNN)** trained from scratch on the **FER-2013** dataset.

---

## Table of Contents
1. [Project Overview & Key Features](#1-project-overview--key-features)
2. [Project Directory Layout](#2-project-directory-layout)
3. [Windows Setup & Execution Guide](#3-windows-setup--execution-guide)
4. [Dataset Setup (FER-2013)](#4-dataset-setup-fer-2013)
5. [Training the Model (`train.py`)](#5-training-the-model-trainpy)
6. [Running the Flask Web Application (`app.py`)](#6-running-the-flask-web-application-apppy)
7. [System Architecture & Assignment Viva Guide](#7-system-architecture--assignment-viva-guide)
8. [Troubleshooting Guide](#8-troubleshooting-guide)

---

## 1. Project Overview & Key Features

- **No Pretrained Emotion Models**: Trained 100% from scratch on the 48x48 FER-2013 grayscale dataset.
- **Dual Input Modes**:
  - **Static Image Upload**: Drag-and-drop or browse JPG, PNG, WEBP images with single or multiple faces.
  - **Live Webcam Stream**: Real-time browser capture via `getUserMedia` with interactive bounding boxes and confidence tags.
- **Robust Computer Vision Pipeline**:
  - Face detection via OpenCV Haar Cascades (`haarcascade_frontalface_default.xml`).
  - Strict preprocessing alignment: Crop ROI $\rightarrow$ Grayscale $\rightarrow$ Resize to $48\times48$ $\rightarrow$ Normalize $[0, 1]$ $\rightarrow$ Reshape $(1, 48, 48, 1)$.
- **Class Imbalance Mitigation**: Computes balanced class weights during training to account for underrepresented classes (e.g., *Disgust*).
- **Comprehensive Evaluation**: Generates training curves, confusion matrix heatmap, and precision/recall/F1 metrics in `outputs/`.
- **Fast Dry-Run Mode**: Optional `--sample <N>` flag lets you verify the entire training and web pipeline in under 2 minutes.

---

## 2. Project Directory Layout

```text
d:\Face Emotion Recognition\
├── dataset/
│   ├── train/                  # Training images by class (angry, disgust, etc.)
│   └── test/                   # Evaluation images by class
├── models/
│   ├── emotion_model.keras     # Trained CNN model weights & graph
│   └── class_names.json        # Saved label ordering (for app.py)
├── outputs/
│   ├── accuracy_loss.png       # Training vs. validation curves
│   ├── confusion_matrix.png    # Normalized test confusion matrix
│   └── classification_report.txt # Real test metrics (precision/recall/F1)
├── static/
│   ├── css/
│   │   └── style.css           # Clean, responsive light theme
│   └── js/
│       ├── main.js             # Image upload, canvas overlay & bars
│       └── webcam.js           # Live webcam capture & real-time streaming
├── templates/
│   └── index.html              # Modern single-page dashboard
├── setup_dataset.py            # Dataset verification & dummy sample tool
├── train.py                    # Complete CNN training & evaluation script
├── app.py                      # Production-ready Flask REST API
├── requirements.txt            # Pinned dependencies
└── README.md                   # Full documentation & viva guide
```

---

## 3. Windows Setup & Execution Guide

Follow these exact steps in **Command Prompt (`cmd`)** or **PowerShell**:

### Step 1: Open Terminal in Project Directory
```powershell
cd "d:\Face Emotion Recognition"
```

### Step 2: Create and Activate a Python Virtual Environment
We recommend Python 3.10 or 3.11.

**In PowerShell:**
```powershell
# Create virtual environment
python -m venv venv

# If PowerShell blocks script execution, allow it for this session:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

**In Command Prompt (cmd.exe):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```
*(Your prompt will now show `(venv)` at the beginning).*

### Step 3: Upgrade pip and Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Dataset Setup (FER-2013)

The FER-2013 dataset contains ~35,000 grayscale 48x48 images categorized into 7 classes:
`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

### Option A: Quick Dry-Run (Instant Setup in 5 Seconds)
If you want to test the entire training and web application before downloading the full 300 MB dataset:
```powershell
python setup_dataset.py --create-dummy
```
This generates synthetic sample images for all 7 classes in `dataset/train/` and `dataset/test/` so you can verify the entire pipeline immediately!

### Option B: Automatic Download via Kaggle API
1. Create a free account at [kaggle.com](https://www.kaggle.com).
2. Go to **Account Settings** $\rightarrow$ Click **Create New Token** to download `kaggle.json`.
3. Place `kaggle.json` inside `C:\Users\<YourUsername>\.kaggle\kaggle.json`.
4. Run:
```powershell
python setup_dataset.py --download
```

### Option C: Manual Download and Extraction
1. Visit the Kaggle dataset page: [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)
2. Click **Download (300 MB)** to save `archive.zip`.
3. Extract `archive.zip`. Inside, you will see `train/` and `test/` folders.
4. Move them into `d:\Face Emotion Recognition\dataset\` so the layout looks like:
   ```text
   dataset/
     ├── train/
     │     ├── angry/
     │     ├── disgust/
     │     ├── fear/
     │     ├── happy/
     │     ├── neutral/
     │     ├── sad/
     │     └── surprise/
     └── test/
           └── ... (all 7 classes)
   ```
5. Verify the layout:
```powershell
python setup_dataset.py --check
```

---

## 5. Training the Model (`train.py`)

`train.py` automatically checks your dataset structure, computes class weights, loads and normalizes the images, trains the CNN with Early Stopping and Checkpointing, evaluates real test performance, and writes all graphs to `outputs/`.

### Rapid Testing Mode (Sample Mode):
Train on a subset (e.g. 200 or 500 images per class) to test everything in 1 to 2 minutes:
```powershell
python train.py --sample 200 --epochs 5
```

### Full Training on the Entire Dataset:
```powershell
python train.py --epochs 20 --batch-size 64
```

### What You Should Expect in the Terminal:
1. **Dataset Layout Check**: Prints a table of images found per class for `train` and `test`.
2. **Class Weights**: Prints calculated balancing weights (e.g. ~7.0x weight for `disgust`).
3. **Model Summary**: Displays layer parameters, output shapes, and total trainable parameters.
4. **Epoch Progression**: Shows loss and accuracy progressing over epochs:
   ```text
   Epoch 1/20
   358/358 [==============================] - 28s - loss: 1.6821 - accuracy: 0.3541 - val_loss: 1.5420 - val_accuracy: 0.4120
   Epoch 2/20
   358/358 [==============================] - 27s - loss: 1.3912 - accuracy: 0.4680 - val_loss: 1.3105 - val_accuracy: 0.5015
   ```
5. **Real Test Evaluation**:
   - Computes actual Test Accuracy on unseen images.
   - Saves:
     - `models/emotion_model.keras`
     - `models/class_names.json`
     - `outputs/accuracy_loss.png`
     - `outputs/confusion_matrix.png`
     - `outputs/classification_report.txt`

---

## 6. Running the Flask Web Application (`app.py`)

Once `models/emotion_model.keras` exists:

```powershell
python app.py
```

### Open the Application:
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

- **Image Upload Tab**: Drag and drop any portrait or selfie image. The app highlights all faces with colored bounding boxes and displays the predicted emotion and confidence breakdown.
- **Webcam Tab**: Click **Start Webcam**. Allow browser camera permissions when prompted. The model will run live inferences at ~1.5 FPS with low latency.

---

## 7. System Architecture & Assignment Viva Guide

Use this section to confidently explain your architecture and design decisions during your project viva or college presentation!

### A. End-to-End System Flow
```
[ User Browser ]
       │ (1. Uploads image file or streams base64 frame)
       ▼
[ Flask Server: app.py ]
       │ (2. Decodes image into BGR NumPy array)
       ▼
[ OpenCV Haar Cascade ]
       │ (3. Detects bounding boxes (x, y, w, h))
       ▼
[ Preprocessing Pipeline ]
       │ (4. Crops face ROI ➔ Grayscale ➔ Resizes to 48x48 ➔ Rescales [0, 1] ➔ Reshapes (1, 48, 48, 1))
       ▼
[ Custom CNN: emotion_model.keras ]
       │ (5. Forward pass ➔ Softmax activation over 7 neurons)
       ▼
[ JSON Response ]
       │ (6. Returns bounding box coordinates + emotion labels + % probabilities)
       ▼
[ Modern UI (Vanilla JS & HTML5 Canvas) ]
         (7. Renders bounding boxes on canvas & animates probability progress bars)
```

### B. Why Each CNN Layer Is Used
1. **Input Layer $(48, 48, 1)$**: Accepts 48x48 single-channel (grayscale) images matching the FER-2013 resolution.
2. **Conv2D ($3\times3$ filters)**:
   - *Block 1 (32 and 64 filters)*: Learns low-level spatial features such as directional edges, lines, and simple texture gradients.
   - *Block 2 (128 filters)*: Learns intermediate facial parts: eyelid contours, nostril curves, lip corners.
   - *Block 3 (256 filters)*: Combines parts into high-level holistic emotion cues: raised eyebrows (Surprise), drawn eyebrows (Anger), upturned lips (Happiness).
3. **BatchNormalization**: Normalizes layer inputs across each mini-batch, accelerating convergence and stabilizing weight updates.
4. **ReLU Activation**: Introduces non-linearity ($f(x) = \max(0, x)$) while eliminating the vanishing gradient problem seen in Sigmoid/Tanh.
5. **MaxPooling2D ($2\times2$)**: Downsamples spatial resolution by half (48 $\rightarrow$ 24 $\rightarrow$ 12 $\rightarrow$ 6), providing translational invariance and reducing computational complexity.
6. **Dropout ($0.25$ and $0.50$)**: Randomly zeroes out a fraction of neuron outputs during training to break co-adaptations, acting as strong regularization against overfitting.
7. **Flatten**: Transforms the 3D tensor $(6, 6, 256)$ into a 1D feature vector of length 9,216.
8. **Dense (256)**: Fully connected reasoning layer that integrates all extracted facial features.
9. **Dense (7, Softmax)**: The output classification head. Softmax converts raw logits into a valid probability distribution where all 7 classes sum to $1.0$ ($100\%$).

### C. Key Machine Learning Concepts Explained
- **Loss Function (Categorical Cross-Entropy)**:
  Measures the divergence between the true one-hot vector $\mathbf{y}$ and the predicted Softmax probabilities $\mathbf{\hat{y}}$:
  $$\mathcal{L} = -\sum_{i=1}^{7} y_i \log(\hat{y}_i)$$
- **EarlyStopping**: Monitors validation loss (`val_loss`). If validation loss ceases to improve for 6 consecutive epochs, training terminates to prevent the network from memorizing training noise.
- **Confusion Matrix**: Visualizes where the model confuses emotions (e.g., distinguishing subtle *Fear* vs. *Surprise* or *Sad* vs. *Neutral*).

---

## 8. Troubleshooting Guide

### 1. `ModuleNotFoundError: No module named 'tensorflow'`
- **Cause**: Virtual environment is not activated or dependencies are not installed.
- **Solution**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

### 2. Python 3.12+ or 3.14 Compatibility Warning
- **Cause**: TensorFlow wheels for Windows are best supported on **Python 3.10 and 3.11**.
- **Solution**: If your default system Python is 3.13 or 3.14, install Python 3.11 or use `uv`:
  ```powershell
  uv venv --python 3.11
  .\.venv\Scripts\activate
  pip install -r requirements.txt
  ```

### 3. "Model not loaded. Run train.py first"
- **Cause**: The web server started, but `models/emotion_model.keras` does not exist.
- **Solution**: Run a fast training session:
  ```powershell
  python train.py --sample 200 --epochs 5
  ```

### 4. Webcam Not Starting / Permission Denied
- **Cause**: Browser blocked camera access or another app is using the webcam.
- **Solution**:
  1. Click the camera icon in your browser's address bar and select **"Always allow"**.
  2. Close any apps that might be using the webcam (Zoom, Teams, Skype, Windows Camera app).
  3. Ensure you are accessing `http://127.0.0.1:5000` (modern browsers allow camera on `localhost`).

### 5. `OSError: [WinError 10048] Only one usage of each socket address is normally permitted`
- **Cause**: Another instance of Flask (or another app) is already running on port 5000.
- **Solution**:
  ```powershell
  # Find and kill the process on port 5000
  Get-Process -Id (Get-NetTCPConnection -LocalPort 5000).OwningProcess | Stop-Process -Force
  ```
  Or change the port in `app.py`: `app.run(host="127.0.0.1", port=5001, debug=True)`.

---

## Academic Integrity & Authorship
This project is built from scratch as an educational assignment. It does not utilize any third-party emotion recognition APIs or black-box pre-trained emotion models.
