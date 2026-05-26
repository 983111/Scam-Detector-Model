# Scam Detection ML System Documentation

## 1. Overview
The Scam Detection ML System is a unified FastAPI-based inference server that analyzes both text messages and URLs to detect potential scams. It combines two primary machine learning models (Model A for URLs and Model B for text messages) to provide a comprehensive risk score and scam classification.

## 2. System Architecture
* **Model A (URL Analyzer):** An ensemble of XGBoost and LightGBM models trained on 23 extracted heuristic features.
* **Model B (Message Classifier):** A fine-tuned DistilBERT (`distilbert-base-uncased`) model utilizing a multi-task architecture with dual classification heads for binary (scam/safe) and multi-class type prediction.
* **API Server:** A FastAPI application that orchestrates model loading, inference, and fallback heuristic modes if models are missing.
* **Scoring Mechanism:** The overall risk probability is a weighted combination of Model A (40% weight) and Model B (60% weight).

## 3. Core Components

### 3.1. API Server (`api/app.py`)
The FastAPI server exposes several endpoints for scam detection. It features a singleton `ModelRegistry` class that loads models at startup and provides pure heuristic fallbacks if the machine learning model files are missing.

#### Endpoints
* `POST /detect`: Full scam analysis combining message and URL signals.
* `POST /detect/url`: URL-only analysis using Model A.
* `POST /detect/message`: Message-only analysis using Model B.
* `GET /health`: Health check indicating whether models are loaded and detailing the compute device.
* `GET /docs`: Auto-generated Swagger UI.

#### Scam Classifications
* `phishing`: Credential theft, fake login pages.
* `urgency`: Pressure tactics, fake deadlines.
* `impersonation`: Brand or authority impersonation.
* `financial`: Lottery, prizes, investment fraud.
* `safe`: Legitimate message.

### 3.2. Message Model (`message_model/train_message_model.py`)
This module handles the fine-tuning of `distilbert-base-uncased`.

* **Custom Architecture:** Modifies the base model by replacing the classifier with a shared backbone (`Linear` -> `GELU` -> `Dropout`) that branches into a binary head and a 5-class type head.
* **Data Pipelines:** Capable of loading datasets from the HuggingFace Hub (e.g., `sms_spam`, Enron phishing datasets) or falling back to a comprehensive synthetic dataset included in the script.
* **Training Loop:** Implements an AdamW optimizer with a linear schedule and warmup. Uses a combined loss function (`binary_loss + 0.5 * type_loss`) to jointly optimize both heads.

### 3.3. URL Feature Extractor (`url_model/url_feature_extractor.py`)
Analyzes URLs strictly via heuristics without making external network API calls. Extracts 23 static features including:

* URL, path, and hostname lengths.
* Shannon entropy for domain and path to detect randomness.
* Presence of suspicious keywords, IP addresses, abnormal ports, or hex encoding.
* Targeted brand impersonation (checking if a brand name is in a subdomain rather than the registrable domain).
* Suspicious Top-Level Domains (TLDs) and URL shortener detection.

### 3.4. URL Model Training (`url_model/train_url_model.py`)
Orchestrates the training for Model A using XGBoost and LightGBM.

* **Datasets:** Automatically attempts to fetch live data from PhishTank (API key supported), OpenPhish, and Alexa/Tranco top domains, falling back to synthetic data if network calls fail.
* **Ensemble:** Trains an `XGBClassifier` and an `LGBMClassifier`, both saving output probabilities that are later weighted.
* **Evaluation:** Automatically finds the optimal classification threshold (maximizing F1 score).

## 4. Setup & Deployment

### 4.1. Local Installation
The project relies on specific Python packages defined in `requirements.txt`:

```bash
# Core ML dependencies
pip install torch==2.2.0 transformers==4.38.0 datasets==2.17.0 xgboost==2.0.3 lightgbm==4.3.0 scikit-learn==1.4.0 joblib==1.3.2

# Data manipulation and API tools
pip install pandas==2.2.0 numpy==1.26.3 fastapi==0.109.2 uvicorn[standard]==0.27.1 pydantic==2.6.0
```
*(See `requirements.txt` for the full dependency list)*

### 4.2. Training the Models
Models must be trained sequentially before starting the API in ML mode:

#### URL Model
```bash
cd url_model
export PHISHTANK_API_KEY="your_key"  # Optional
python train_url_model.py
```

#### Message Model
```bash
cd message_model
python train_message_model.py
```

### 4.3. Docker Integration (`deployment/Dockerfile`)
The repository includes a ready-to-use Docker environment designed specifically for HuggingFace Spaces.

* **Base Image:** `python:3.11-slim`
* **Exposed Port:** 7860
* **Execution Command:** Starts FastAPI using Uvicorn on host `0.0.0.0`

### 4.4. HuggingFace Deployment (`deployment/deploy_to_hf.py`)
An automated deployment script is provided to bundle the necessary folders (`url_model`, `message_model`, `api`, `models`) and the Docker configuration.

```bash
pip install huggingface_hub
huggingface-cli login
python deployment/deploy_to_hf.py --username YOUR_HF_USERNAME --space-name scam-detection-api
```
