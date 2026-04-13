"""
Scam Detection API — FastAPI Inference Server
=============================================
Combines Model A (URL) + Model B (Message) into a single unified API.
Deploy on HuggingFace Spaces (Docker) or any cloud provider.

Endpoints:
  POST /detect         — Full scam analysis (message + URL)
  POST /detect/url     — URL-only analysis
  POST /detect/message — Message-only analysis
  GET  /health         — Health check
  GET  /docs           — Auto-generated Swagger UI
"""

import os
import json
import math
import time
import joblib
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, List
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import local modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from url_model.url_feature_extractor import (
    extract_url_features, features_to_vector, _feature_names
)

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────

MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
URL_MODEL_META = MODEL_DIR / "url_model_metadata.json"
URL_XGB_PATH = MODEL_DIR / "url_xgb.joblib"
URL_LGB_PATH = MODEL_DIR / "url_lgb.joblib"
MSG_MODEL_DIR = MODEL_DIR / "message_model"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TYPE_LABELS = {
    0: "safe",
    1: "phishing",
    2: "urgency",
    3: "impersonation",
    4: "financial",
}


# ─────────────────────────────────────────────────────────────
#  MODEL DEFINITIONS (must match training)
# ─────────────────────────────────────────────────────────────

class ScamDetectorModel(nn.Module):
    def __init__(self, num_types: int = 5, dropout: float = 0.3):
        super().__init__()
        from transformers import DistilBertModel
        self.distilbert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )
        hidden_size = 768
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(512, 2)
        self.type_head = nn.Linear(512, num_types)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, 0, :]
        shared = self.shared(hidden)
        return self.binary_head(shared), self.type_head(shared)


# ─────────────────────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────────────────────

class ModelRegistry:
    """Singleton model loader — loads once at startup."""

    def __init__(self):
        self.url_xgb = None
        self.url_lgb = None
        self.url_meta = {}
        self.msg_model = None
        self.msg_tokenizer = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        print("🔄 Loading models...")
        self._load_url_models()
        self._load_message_model()
        self._loaded = True
        print("✅ All models loaded!")

    def _load_url_models(self):
        try:
            self.url_xgb = joblib.load(URL_XGB_PATH)
            self.url_lgb = joblib.load(URL_LGB_PATH)
            with open(URL_MODEL_META) as f:
                self.url_meta = json.load(f)
            print(f"   ✅ URL models loaded (threshold={self.url_meta['best_threshold']})")
        except Exception as e:
            print(f"   ⚠️  URL models not found: {e}")
            print("   → Run train_url_model.py first, or models will use heuristics only.")

    def _load_message_model(self):
        try:
            from transformers import DistilBertTokenizerFast
            self.msg_tokenizer = DistilBertTokenizerFast.from_pretrained(
                str(MSG_MODEL_DIR)
            )
            self.msg_model = ScamDetectorModel(num_types=5)
            state = torch.load(
                MSG_MODEL_DIR / "best_model.pt",
                map_location=DEVICE
            )
            self.msg_model.load_state_dict(state)
            self.msg_model.eval()
            self.msg_model.to(DEVICE)
            print("   ✅ Message model loaded")
        except Exception as e:
            print(f"   ⚠️  Message model not found: {e}")
            print("   → Run train_message_model.py first.")

    def url_predict(self, url: str) -> Dict:
        """Predict scam probability for a URL."""
        if not self.url_xgb:
            return self._heuristic_url_predict(url)

        features = extract_url_features(url)
        X = np.array([features_to_vector(features)], dtype=np.float32)

        xgb_weight = self.url_meta.get("xgb_weight", 0.5)
        p_xgb = self.url_xgb.predict_proba(X)[0, 1]
        p_lgb = self.url_lgb.predict_proba(X)[0, 1]
        prob = xgb_weight * p_xgb + (1 - xgb_weight) * p_lgb

        threshold = self.url_meta.get("best_threshold", 0.5)
        return {
            "scam_probability": round(float(prob), 4),
            "is_scam": bool(prob >= threshold),
            "features": {k: round(v, 3) for k, v in features.items()},
        }

    def _heuristic_url_predict(self, url: str) -> Dict:
        """Fallback: pure heuristic URL scoring (no ML model needed)."""
        features = extract_url_features(url)

        # Weighted risk score
        score = (
            features["has_ip_address"] * 0.25 +
            features["brand_impersonation"] * 0.25 +
            features["is_suspicious_tld"] * 0.15 +
            features["domain_entropy"] * 0.10 +
            features["has_suspicious_keyword"] * 0.10 +
            features["at_symbol"] * 0.05 +
            features["double_slash_redirect"] * 0.05 +
            features["is_url_shortener"] * 0.05 +
            (1.0 - features["is_https"]) * 0.05 +
            features["has_abnormal_port"] * 0.05
        )
        score = min(score, 1.0)

        return {
            "scam_probability": round(score, 4),
            "is_scam": score >= 0.5,
            "features": {k: round(v, 3) for k, v in features.items()},
            "note": "Heuristic mode (ML model not loaded)",
        }

    def message_predict(self, text: str) -> Dict:
        """Predict scam type and probability for a message."""
        if not self.msg_model:
            return self._heuristic_message_predict(text)

        encoding = self.msg_tokenizer(
            text, truncation=True, padding="max_length",
            max_length=128, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attn_mask = encoding["attention_mask"].to(DEVICE)

        with torch.no_grad():
            binary_logits, type_logits = self.msg_model(input_ids, attn_mask)

        binary_probs = torch.softmax(binary_logits, dim=1)[0].cpu().numpy()
        type_probs = torch.softmax(type_logits, dim=1)[0].cpu().numpy()

        scam_prob = float(binary_probs[1])
        scam_type_id = int(type_probs.argmax())
        scam_type = TYPE_LABELS[scam_type_id]

        return {
            "scam_probability": round(scam_prob, 4),
            "is_scam": scam_prob >= 0.5,
            "scam_type": scam_type,
            "type_confidence": round(float(type_probs.max()), 4),
            "type_probabilities": {
                TYPE_LABELS[i]: round(float(p), 4)
                for i, p in enumerate(type_probs)
            },
        }

    def _heuristic_message_predict(self, text: str) -> Dict:
        """Fallback heuristic message scoring."""
        text_lower = text.lower()

        urgency_kws = ["urgent", "immediately", "24 hours", "act now",
                       "expire", "last chance", "warning", "alert", "suspend"]
        phishing_kws = ["verify", "confirm", "click here", "login", "otp",
                        "pin", "password", "kyc", "update your", "account"]
        financial_kws = ["won", "prize", "lottery", "reward", "cash",
                         "money", "investment", "returns", "gift card", "free"]
        impersonation_kws = ["microsoft", "amazon", "apple", "google",
                             "irs", "fbi", "support team", "customer care",
                             "bank", "sbi", "hdfc"]

        scores = {
            "phishing": sum(1 for kw in phishing_kws if kw in text_lower),
            "urgency": sum(1 for kw in urgency_kws if kw in text_lower),
            "financial": sum(1 for kw in financial_kws if kw in text_lower),
            "impersonation": sum(1 for kw in impersonation_kws if kw in text_lower),
        }

        total_hits = sum(scores.values())
        scam_prob = min(total_hits / 8.0, 1.0)

        if total_hits == 0:
            scam_type = "safe"
        else:
            scam_type = max(scores, key=scores.get)

        return {
            "scam_probability": round(scam_prob, 4),
            "is_scam": scam_prob >= 0.3,
            "scam_type": scam_type,
            "type_confidence": 0.5,
            "note": "Heuristic mode (ML model not loaded)",
        }


# Global registry
registry = ModelRegistry()


# ─────────────────────────────────────────────────────────────
#  PYDANTIC SCHEMAS
# ─────────────────────────────────────────────────────────────

class DetectRequest(BaseModel):
    message: Optional[str] = Field(None, description="Message/SMS/email text")
    url: Optional[str] = Field(None, description="URL found in message")
    sender: Optional[str] = Field(None, description="Sender phone/email (optional)")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "URGENT: Your account will be suspended. Click to verify: http://paypal-secure.tk/verify",
                "url": "http://paypal-secure.tk/verify",
                "sender": "+1-888-SCAM-999"
            }
        }


class URLRequest(BaseModel):
    url: str
    class Config:
        json_schema_extra = {"example": {"url": "http://paypal-secure.tk/verify"}}


class MessageRequest(BaseModel):
    message: str
    sender: Optional[str] = None
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Congratulations! You won $1,000,000. Claim now!",
                "sender": "unknown"
            }
        }


class DetectResponse(BaseModel):
    is_scam: bool
    overall_scam_probability: float
    risk_level: str  # "low" | "medium" | "high" | "critical"
    scam_type: Optional[str]
    url_analysis: Optional[Dict]
    message_analysis: Optional[Dict]
    explanation: str
    processing_time_ms: float


# ─────────────────────────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Scam Detection API",
    description="""
## 🛡️ AI-Powered Scam Detection API

Two-model system for detecting scams in messages and URLs:

- **Model A**: URL analysis (XGBoost + LightGBM, 23 features, no external APIs)
- **Model B**: Message analysis (DistilBERT fine-tuned, detects type + probability)

### Scam Types Detected
- `phishing` — credential theft, fake login pages
- `urgency` — pressure tactics, fake deadlines  
- `impersonation` — brand/authority impersonation
- `financial` — lottery, prizes, investment fraud
- `safe` — legitimate message
""",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    registry.load()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "url_model_loaded": registry.url_xgb is not None,
        "message_model_loaded": registry.msg_model is not None,
        "device": str(DEVICE),
    }


@app.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    """
    Full scam analysis — combines URL + message signals.
    The overall probability is a weighted combination of both models.
    """
    if not req.message and not req.url:
        raise HTTPException(400, "Provide at least one of: message, url")

    t0 = time.time()
    url_result = None
    msg_result = None

    # URL analysis
    if req.url:
        url_result = registry.url_predict(req.url)

    # Message analysis
    if req.message:
        msg_result = registry.message_predict(req.message)

    # Combine probabilities
    probs = []
    if url_result:
        probs.append(url_result["scam_probability"] * 0.4)
    if msg_result:
        probs.append(msg_result["scam_probability"] * 0.6)
    overall_prob = sum(probs) / len(probs) * 2 if probs else 0.0
    overall_prob = min(overall_prob, 1.0)

    # Risk level
    if overall_prob < 0.25:
        risk_level = "low"
    elif overall_prob < 0.5:
        risk_level = "medium"
    elif overall_prob < 0.75:
        risk_level = "high"
    else:
        risk_level = "critical"

    # Scam type (prefer message model's classification)
    scam_type = None
    if msg_result and msg_result.get("is_scam"):
        scam_type = msg_result.get("scam_type")
    elif url_result and url_result.get("is_scam"):
        scam_type = "phishing"

    # Human-readable explanation
    reasons = []
    if url_result:
        f = url_result.get("features", {})
        if f.get("has_ip_address"):
            reasons.append("URL uses raw IP address")
        if f.get("brand_impersonation"):
            reasons.append("URL impersonates a known brand")
        if f.get("is_suspicious_tld"):
            reasons.append("URL uses suspicious domain extension")
        if f.get("domain_entropy", 0) > 0.7:
            reasons.append("Domain name looks randomly generated")
    if msg_result:
        if msg_result.get("scam_type") == "urgency":
            reasons.append("Message creates artificial urgency")
        if msg_result.get("scam_type") == "phishing":
            reasons.append("Message attempts credential theft")
        if msg_result.get("scam_type") == "financial":
            reasons.append("Message contains financial fraud indicators")
        if msg_result.get("scam_type") == "impersonation":
            reasons.append("Message impersonates a trusted entity")

    explanation = (
        "Scam detected: " + "; ".join(reasons) if reasons and overall_prob > 0.5
        else "No significant scam indicators found." if overall_prob < 0.3
        else "Some suspicious signals detected — proceed with caution."
    )

    return DetectResponse(
        is_scam=overall_prob >= 0.5,
        overall_scam_probability=round(overall_prob, 4),
        risk_level=risk_level,
        scam_type=scam_type,
        url_analysis=url_result,
        message_analysis=msg_result,
        explanation=explanation,
        processing_time_ms=round((time.time() - t0) * 1000, 1),
    )


@app.post("/detect/url")
async def detect_url(req: URLRequest):
    """URL-only scam analysis."""
    t0 = time.time()
    result = registry.url_predict(req.url)
    result["processing_time_ms"] = round((time.time() - t0) * 1000, 1)
    return result


@app.post("/detect/message")
async def detect_message(req: MessageRequest):
    """Message-only scam analysis."""
    t0 = time.time()
    result = registry.message_predict(req.message)
    result["processing_time_ms"] = round((time.time() - t0) * 1000, 1)
    return result


@app.get("/")
async def root():
    return {
        "name": "Scam Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/detect", "/detect/url", "/detect/message", "/health"],
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
