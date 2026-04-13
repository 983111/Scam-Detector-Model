# 🛡️ Complete Scam Detection ML System — Full Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Your App (Mobile/Web)                  │
└────────────────────────┬────────────────────────────────┘
                         │ POST /detect
                         ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Inference Server                    │
│          (HuggingFace Spaces — Docker)                   │
│                                                         │
│  ┌─────────────────┐      ┌──────────────────────────┐  │
│  │   Model A        │      │   Model B                │  │
│  │   URL Analyzer   │      │   Message Classifier     │  │
│  │                  │      │                          │  │
│  │  XGBoost +       │      │  DistilBERT fine-tuned   │  │
│  │  LightGBM        │      │  Multi-task:             │  │
│  │  23 features     │      │  - Binary (scam/safe)    │  │
│  │  No API needed   │      │  - Type classification   │  │
│  └──────┬──────────┘      └─────────────┬────────────┘  │
│         │                               │                │
│         └───────────┬───────────────────┘                │
│                     ▼                                    │
│            Combined Risk Score                           │
│         (URL: 40% + Message: 60%)                        │
└─────────────────────────────────────────────────────────┘
```

---

## Step 1: Setup

```bash
# Clone / create project structure
mkdir scam-detector && cd scam-detector

# Install all dependencies
pip install -r requirements.txt

# (Optional) GPU for faster training
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 2: Download Training Data

```bash
# Run the data preparation script
python data/prepare_data.py
```

This downloads (all free):
| Source | URLs/Messages | Key Required? |
|--------|--------------|---------------|
| URLhaus | ~50k malicious URLs | ❌ No |
| OpenPhish | ~5k phishing URLs | ❌ No |
| Tranco top-1M | 100k safe domains | ❌ No |
| PhishTank | ~25k phishing URLs | ✅ Free signup |
| SMS Spam (HF) | 5,574 SMS | ❌ No |
| Phishing Emails | ~10k emails | ❌ No |

**PhishTank API Key** (recommended, 2 min):
1. Go to https://phishtank.org/register.php
2. Register free account
3. Get API key from https://phishtank.org/api_info.php
4. `export PHISHTANK_API_KEY=your_key_here`

---

## Step 3: Train Model A (URL Detection)

```bash
cd url_model
python train_url_model.py
```

**What it does:**
- Extracts 23 features from each URL (no internet needed at inference)
- Trains XGBoost + LightGBM in ensemble
- Auto-calibrates classification threshold
- Saves to `models/url_xgb.joblib`, `models/url_lgb.joblib`

**Expected output:**
```
ROC-AUC:  0.97+
Avg Prec: 0.96+
Optimal threshold: 0.42
```

**Training time:** ~5-10 minutes (CPU) | ~2 min (GPU)

---

## Step 4: Train Model B (Message Detection)

```bash
cd message_model
python train_message_model.py
```

**What it does:**
- Fine-tunes DistilBERT on SMS + email scam datasets
- Dual classification heads: binary + scam type
- Saves best checkpoint to `models/message_model/`

**Expected output:**
```
Epoch 4/4 | Val F1: 0.97+
Scam types: phishing / urgency / impersonation / financial
```

**Training time:** ~20-30 min (GPU) | ~2-3 hours (CPU)

> 💡 Use Google Colab (free T4 GPU) for training if you don't have GPU:
> Upload project to Google Drive, run in Colab notebook.

---

## Step 5: Test Locally

```bash
# Start the API
python api/app.py

# Test in another terminal
curl -X POST http://localhost:7860/detect \
  -H "Content-Type: application/json" \
  -d '{
    "message": "URGENT: Your PayPal account is suspended! Verify: http://paypal-secure.tk/login",
    "url": "http://paypal-secure.tk/login"
  }'
```

**Expected response:**
```json
{
  "is_scam": true,
  "overall_scam_probability": 0.96,
  "risk_level": "critical",
  "scam_type": "phishing",
  "explanation": "Scam detected: URL impersonates a known brand; Message attempts credential theft",
  "url_analysis": {
    "scam_probability": 0.94,
    "is_scam": true,
    "features": {
      "brand_impersonation": 1.0,
      "is_suspicious_tld": 1.0,
      "domain_entropy": 0.62
    }
  },
  "message_analysis": {
    "scam_probability": 0.98,
    "is_scam": true,
    "scam_type": "urgency",
    "type_confidence": 0.91
  },
  "processing_time_ms": 47.3
}
```

---

## Step 6: Deploy to HuggingFace Spaces

### Option A: Automated Script
```bash
pip install huggingface_hub
huggingface-cli login

python deployment/deploy_to_hf.py \
  --username YOUR_HF_USERNAME \
  --space-name scam-detection-api
```

### Option B: Manual Upload
1. Go to https://huggingface.co/new-space
2. Select **Docker** as SDK
3. Create space named `scam-detection-api`
4. Clone the repo locally:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/scam-detection-api
   ```
5. Copy your project files into it
6. Push:
   ```bash
   git add -A
   git commit -m "Deploy scam detection API"
   git push
   ```

HuggingFace will build the Docker container automatically (~5 min).

---

## Step 7: Integrate in Your App

### Python
```python
import requests

SCAM_API = "https://YOUR_USERNAME-scam-detection-api.hf.space"

def check_message(message: str, url: str = None) -> dict:
    payload = {"message": message}
    if url:
        payload["url"] = url
    
    resp = requests.post(f"{SCAM_API}/detect", json=payload, timeout=10)
    return resp.json()

# Usage
result = check_message(
    "Congratulations! You won Rs 50 lakh! Click: http://prize-india.tk/claim",
    url="http://prize-india.tk/claim"
)

if result["is_scam"]:
    print(f"⚠️ SCAM DETECTED: {result['scam_type']}")
    print(f"   Risk: {result['risk_level']}")
    print(f"   {result['explanation']}")
```

### JavaScript / React Native
```javascript
const SCAM_API = "https://YOUR_USERNAME-scam-detection-api.hf.space";

async function checkMessage(message, url = null) {
  const payload = { message };
  if (url) payload.url = url;

  const res = await fetch(`${SCAM_API}/detect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  
  return res.json();
}

// Usage in React Native
const handleIncomingSMS = async (sms) => {
  const urlMatch = sms.text.match(/(https?:\/\/[^\s]+)/);
  const result = await checkMessage(sms.text, urlMatch?.[0]);
  
  if (result.is_scam) {
    Alert.alert(
      "⚠️ Suspicious Message",
      `${result.scam_type.toUpperCase()}: ${result.explanation}`,
      [{ text: "Mark Safe" }, { text: "Block", style: "destructive" }]
    );
  }
};
```

---

## Model Performance (Expected)

| Model | Accuracy | F1 Score | ROC-AUC | Latency |
|-------|----------|----------|---------|---------|
| URL (ensemble) | ~97% | ~0.96 | ~0.98 | <5ms |
| Message (DistilBERT) | ~95% | ~0.95 | ~0.97 | ~50ms |
| Combined | ~97% | ~0.96 | ~0.98 | ~55ms |

---

## Improving Accuracy Further

### 1. More Data (biggest impact)
- **Kaggle**: Search "phishing dataset", "SMS spam" 
- **APWG eCrime datasets**: https://apwg.org/resources/
- **Cybersecurity forums**: collect real scam messages
- **Your own app users**: active learning pipeline

### 2. Better URL Features
```python
# Add to url_feature_extractor.py:
# - WHOIS domain age (new domains = suspicious)
# - DNS MX record check
# - SSL certificate age
# - PageRank-like score
```

### 3. Upgrade Message Model
- Replace DistilBERT with `bert-base-uncased` (more accurate, slower)
- Or use `microsoft/deberta-v3-small` (best small model)
- Or fine-tune a quantized Llama/Gemma for even better accuracy

### 4. Active Learning Loop
- Log all predictions to a database
- Review low-confidence predictions (0.4-0.6 probability)
- Add corrected labels back to training set
- Retrain weekly

---

## Directory Structure
```
scam-detector/
├── url_model/
│   ├── url_feature_extractor.py   ← 23-feature URL analyzer
│   └── train_url_model.py         ← XGBoost + LightGBM training
├── message_model/
│   └── train_message_model.py     ← DistilBERT fine-tuning
├── api/
│   └── app.py                     ← FastAPI inference server
├── data/
│   └── prepare_data.py            ← Data download script
├── deployment/
│   ├── Dockerfile                 ← HuggingFace deployment
│   ├── README.md                  ← HF Space README
│   └── deploy_to_hf.py           ← Automated deploy script
├── models/                        ← Trained model files (git-ignored)
│   ├── url_xgb.joblib
│   ├── url_lgb.joblib
│   ├── url_model_metadata.json
│   └── message_model/
│       ├── best_model.pt
│       ├── tokenizer files...
│       └── metadata.json
└── requirements.txt
```

---

## Cost

| Resource | Cost |
|----------|------|
| HuggingFace Space (CPU) | **FREE** |
| PhishTank API | **FREE** |
| OpenPhish | **FREE** |
| URLhaus | **FREE** |
| SMS Spam dataset | **FREE** |
| Training (Google Colab) | **FREE** |
| **Total** | **$0** |
