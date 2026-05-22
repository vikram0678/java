# 🐦 SentiStream – Real-Time Sentiment Analysis Platform

A production-ready MLOps pipeline that analyzes text sentiment (POSITIVE / NEGATIVE) using a fine-tuned DistilBERT model served via a Java Spring Boot REST API, with a Streamlit web UI — all containerized with Docker.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [How to Use the UI](#how-to-use-the-ui)
- [Batch Processing](#batch-processing)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)

---

## 📖 Project Overview

SentiStream takes any text as input and predicts whether it is **POSITIVE** or **NEGATIVE** in sentiment. It is built as a full MLOps pipeline:

```
Text Input → Java REST API → DistilBERT Model → POSITIVE / NEGATIVE
```

**Real-world use cases:**
- Analyzing product reviews on e-commerce platforms
- Monitoring customer support tickets
- Tracking brand sentiment on social media
- Processing feedback forms in bulk via CSV

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| ML Model | DistilBERT (HuggingFace) | Pre-trained sentiment model |
| Model Format | TorchScript (.pt) | Java-compatible model export |
| API Server | Java 17 + Spring Boot 3.2 | REST API for predictions |
| Inference Engine | DJL (Deep Java Library) | Runs PyTorch model inside JVM |
| Frontend | Python + Streamlit | Web UI for user interaction |
| Containerization | Docker + Docker Compose | Portable deployment |

---

## 📁 Project Structure

```
bert-sentiment-mlops/
│
├── src/                                        # Java Spring Boot API
│   └── main/java/com/mlops/sentiment/
│       ├── SentimentApplication.java           # Main entry point
│       ├── controller/
│       │   └── SentimentController.java        # REST endpoints
│       ├── model/
│       │   ├── SentimentRequest.java           # Single input model
│       │   ├── SentimentResponse.java          # Single output model
│       │   ├── BatchRequest.java               # Batch input model
│       │   └── BatchResponse.java              # Batch output model
│       └── service/
│           └── InferenceService.java           # Loads model, runs prediction
│
├── model_output/                               # Trained model files
│   ├── config.json                             # Model config + label mapping
│   ├── model_output.pt                         # TorchScript model (for DJL)
│   ├── tokenizer.json                          # Tokenizer vocabulary
│   └── tokenizer_config.json                  # Tokenizer settings
│
├── src/ui.py                                   # Streamlit frontend
├── preprocess.py                               # Data preprocessing script
├── train.py                                    # Model training script
├── convert_to_pt.py                            # Export model to TorchScript
├── pom.xml                                     # Java Maven dependencies
├── requirements.txt                            # Python UI dependencies
├── Dockerfile.api                              # Docker image for Java API
├── Dockerfile.ui                               # Docker image for Streamlit UI
├── docker-compose.yml                          # Runs both services together
└── .dockerignore                               # Excludes large files from build
```

---

## ✅ Prerequisites

Make sure you have these installed before starting:

| Tool | Version | Download |
|---|---|---|
| Docker Desktop | Latest | https://www.docker.com/products/docker-desktop |
| Python | 3.10+ | https://www.python.org/downloads |
| Git | Latest | https://git-scm.com/downloads |

> **Note:** Java and Maven are NOT needed on your machine — they run inside Docker.

---

## 🚀 Quick Start

### Step 1 — Clone the Repository

```bash
git clone https://github.com/vikram0678/java.git
cd bert-sentiment-mlops
```

### Step 2 — Set Up Python Environment

```bash
python -m venv pyenv
```

**On Windows:**
```bash
pyenv\Scripts\activate
```

**On Mac/Linux:**
```bash
source pyenv/bin/activate
```

### Step 3 — Install Python Dependencies

```bash
pip install torch transformers safetensors
```

### Step 4 — Download and Export the Model

This downloads the pre-trained DistilBERT sentiment model from HuggingFace and converts it to TorchScript format for Java:

```bash
python convert_to_pt.py
```

After this, your `model_output/` folder should contain:
```
model_output/
  ├── config.json
  ├── model.safetensors
  ├── model_output.pt       ← newly generated
  ├── tokenizer.json
  └── tokenizer_config.json
```

### Step 5 — Start the Application

```bash
docker-compose up --build api ui
```

**First run will take 5–10 minutes** because Docker downloads:
- Maven dependencies for Java build
- PyTorch native libraries (~500MB)

Subsequent runs will be much faster due to caching.

### Step 6 — Open the App

Once you see this in the terminal:
```
api-1  | >>> SUCCESS: Model loaded with TextClassificationTranslatorFactory!
api-1  | Tomcat started on port 8000
ui-1   | You can now view your Streamlit app in your browser.
```

Open your browser:
- **Web UI** → http://localhost:8501
- **API Health Check** → http://localhost:8000/api/v1/sentiment/health

---

## 🔌 API Endpoints

### Health Check
```
GET /api/v1/sentiment/health
```
```bash
curl http://localhost:8000/api/v1/sentiment/health
```
```json
{"status": "healthy"}
```

---

### Single Prediction
```
POST /api/v1/sentiment/predict
```
```bash
curl -X POST http://localhost:8000/api/v1/sentiment/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing"}'
```
```json
{
  "text": "This product is absolutely amazing",
  "label": "POSITIVE"
}
```

---

### Batch Prediction (JSON)
```
POST /api/v1/sentiment/batch
```
```bash
curl -X POST http://localhost:8000/api/v1/sentiment/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I love this", "This is terrible", "Not bad at all"]}'
```
```json
{
  "total": 3,
  "results": [
    {"text": "I love this", "label": "POSITIVE"},
    {"text": "This is terrible", "label": "NEGATIVE"},
    {"text": "Not bad at all", "label": "POSITIVE"}
  ]
}
```

---

### Batch Prediction (CSV File)
```
POST /api/v1/sentiment/batch/csv
```

First create a CSV file with a `text` header:
```
text
I love this product
Terrible quality
Great value for money
Would not recommend
```

Then upload it:
```bash
curl -X POST http://localhost:8000/api/v1/sentiment/batch/csv \
  -F "file=@your_file.csv"
```
```json
{
  "total": 4,
  "results": [
    {"text": "I love this product", "label": "POSITIVE"},
    {"text": "Terrible quality", "label": "NEGATIVE"},
    {"text": "Great value for money", "label": "POSITIVE"},
    {"text": "Would not recommend", "label": "NEGATIVE"}
  ]
}
```

---

## 🖥️ How to Use the UI

1. Open http://localhost:8501 in your browser
2. Type any text in the input box
3. Click **Analyze Sentiment**
4. See the result — **POSITIVE** or **NEGATIVE**

**Example texts to try:**

| Text | Expected Result |
|---|---|
| I absolutely love this product | POSITIVE |
| The service was terrible | NEGATIVE |
| Best purchase I have ever made | POSITIVE |
| Complete waste of money | NEGATIVE |
| The design is nice but battery is poor | NEGATIVE |

---

## 📦 Batch Processing

Batch processing lets you analyze **thousands of texts in one request** instead of one by one.

**When to use batch:**
- Analyzing all product reviews at once
- Processing a day's worth of customer support tickets
- Running sentiment on exported social media data

**CSV Format:**
```csv
text
Your first review here
Your second review here
Your third review here
```

The first row must be the header `text`. Each row after is one text to analyze.

---

## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    User / Client                        │
│         (Browser at localhost:8501 or curl)             │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP Request
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Streamlit UI (Port 8501)                   │
│                    ui.py                                │
│         Collects text → calls Java API                  │
└──────────────────────┬──────────────────────────────────┘
                       │ POST /api/v1/sentiment/predict
                       ▼
┌─────────────────────────────────────────────────────────┐
│         Java Spring Boot API (Port 8000)                │
│                                                         │
│  SentimentController  →  InferenceService               │
│  (receives request)       (runs model)                  │
│                                                         │
│  DJL loads model_output.pt into JVM memory              │
│  DistilBERT tokenizes text → runs inference             │
│  Returns: POSITIVE or NEGATIVE                          │
└──────────────────────┬──────────────────────────────────┘
                       │ JSON Response
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Streamlit UI displays result               │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 Troubleshooting

### Docker build is slow
First build always takes 5–10 minutes. Subsequent builds use cache and are much faster. The `djl_cache` Docker volume saves PyTorch native libs so they don't re-download.

### Port already in use
```bash
# Stop all running containers
docker-compose down

# Then start again
docker-compose up api ui
```

### Model not found error
Make sure you ran `python convert_to_pt.py` before starting Docker. The `model_output/` folder must contain `model_output.pt`.

### UI shows "Failed to connect to backend"
The API container is still loading the model. Wait 1–2 minutes and refresh the page.

### Out of memory error
This app is optimized for 8GB RAM. Close other heavy applications before running. The JVM is capped at 512MB and PyTorch threads are limited to reduce memory usage.

---

## 📝 License

MIT License — free to use, modify, and distribute.

---

## 👨‍💻 Author

**Vikram** — Built as a full MLOps pipeline project demonstrating Java-based model serving with DJL, Docker containerization, and REST API design.