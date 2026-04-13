"""
URL Scam Detection Model — Training Script
==========================================
Trains an XGBoost + LightGBM ensemble on phishing URL datasets.
Uses PhishTank, OpenPhish, and Alexa top domains.

Run:
    pip install xgboost lightgbm scikit-learn pandas requests tqdm joblib
    python train_url_model.py
"""

import os
import json
import time
import joblib
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb
import lightgbm as lgb

from url_feature_extractor import extract_url_features, features_to_vector, _feature_names

# ─────────────────────────────────────────────────────────────
#  DATA COLLECTION
# ─────────────────────────────────────────────────────────────

def download_phishtank(output_path: str = "data/phishtank.csv") -> pd.DataFrame:
    """
    Download phishing URLs from PhishTank.
    Sign up free at https://phishtank.org/api_info.php
    Set env var: PHISHTANK_API_KEY
    """
    os.makedirs("data", exist_ok=True)
    api_key = os.getenv("PHISHTANK_API_KEY", "")

    print("📥 Downloading PhishTank data...")
    url = f"http://data.phishtank.com/data/{api_key}/online-valid.csv"

    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        df = pd.read_csv(output_path, usecols=["url"])
        df["label"] = 1
        print(f"   ✅ PhishTank: {len(df)} phishing URLs")
        return df
    except Exception as e:
        print(f"   ⚠️  PhishTank failed ({e}). Using synthetic data.")
        return _synthetic_phishing_urls()


def download_openphish(output_path: str = "data/openphish.txt") -> pd.DataFrame:
    """Download from OpenPhish community feed (free, no key needed)."""
    os.makedirs("data", exist_ok=True)
    print("📥 Downloading OpenPhish data...")
    try:
        resp = requests.get("https://openphish.com/feed.txt", timeout=30)
        resp.raise_for_status()
        urls = [u.strip() for u in resp.text.splitlines() if u.strip()]
        df = pd.DataFrame({"url": urls, "label": 1})
        print(f"   ✅ OpenPhish: {len(df)} phishing URLs")
        return df
    except Exception as e:
        print(f"   ⚠️  OpenPhish failed ({e}). Using synthetic data.")
        return _synthetic_phishing_urls(n=500)


def load_alexa_safe_urls(output_path: str = "data/alexa_top.csv",
                         n: int = 50000) -> pd.DataFrame:
    """
    Load Alexa/Tranco top domains as safe URLs.
    Uses Tranco list (https://tranco-list.eu/) — free, no key needed.
    """
    os.makedirs("data", exist_ok=True)
    print("📥 Downloading Tranco top domains (safe URLs)...")
    try:
        resp = requests.get(
            "https://tranco-list.eu/top-1m.csv.zip",
            timeout=60, stream=True
        )
        resp.raise_for_status()
        import zipfile, io
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None, names=["rank", "domain"])
        df = df.head(n)
        df["url"] = "https://" + df["domain"]
        df["label"] = 0
        df = df[["url", "label"]]
        print(f"   ✅ Tranco: {len(df)} safe URLs")
        return df
    except Exception as e:
        print(f"   ⚠️  Tranco failed ({e}). Using synthetic safe URLs.")
        return _synthetic_safe_urls(n=5000)


def _synthetic_phishing_urls(n: int = 2000) -> pd.DataFrame:
    """Generate synthetic phishing URLs for fallback/testing."""
    import random, string
    templates = [
        "http://paypal-secure-{rand}.tk/login/verify",
        "http://amazon-account-update.xyz/signin?token={rand}",
        "http://192.168.{a}.{b}/banking/login",
        "http://secure-{brand}-verify.top/account/suspend",
        "http://{rand}.click/free-prize-claim",
        "http://login.{brand}-support.ml/auth",
    ]
    brands = ["amazon", "paypal", "apple", "google", "netflix"]
    urls = []
    for _ in range(n):
        tmpl = random.choice(templates)
        rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        brand = random.choice(brands)
        a, b = random.randint(1, 254), random.randint(1, 254)
        urls.append(tmpl.format(rand=rand, brand=brand, a=a, b=b))
    return pd.DataFrame({"url": urls, "label": 1})


def _synthetic_safe_urls(n: int = 2000) -> pd.DataFrame:
    """Generate synthetic safe URLs for fallback/testing."""
    domains = [
        "google.com", "youtube.com", "facebook.com", "amazon.com",
        "wikipedia.org", "twitter.com", "reddit.com", "linkedin.com",
        "github.com", "stackoverflow.com", "medium.com", "bbc.com",
        "nytimes.com", "cnn.com", "apple.com", "microsoft.com",
        "netflix.com", "spotify.com", "airbnb.com", "uber.com",
    ]
    paths = ["/", "/about", "/search", "/home", "/articles", "/blog", "/news"]
    import random
    urls = [
        f"https://{random.choice(domains)}{random.choice(paths)}"
        for _ in range(n)
    ]
    return pd.DataFrame({"url": urls, "label": 0})


# ─────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features from all URLs into X matrix."""
    X, y = [], []
    print(f"\n🔧 Extracting features from {len(df)} URLs...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            feats = extract_url_features(row["url"])
            X.append(features_to_vector(feats))
            y.append(row["label"])
        except Exception:
            pass  # skip malformed URLs
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ─────────────────────────────────────────────────────────────
#  MODEL TRAINING
# ─────────────────────────────────────────────────────────────

def train_ensemble(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray):
    """Train XGBoost + LightGBM ensemble."""

    # ── XGBoost ───────────────────────────────────────────────
    print("\n🚀 Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # ── LightGBM ──────────────────────────────────────────────
    print("\n🚀 Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
    )

    return xgb_model, lgb_model


def ensemble_predict_proba(xgb_model, lgb_model,
                            X: np.ndarray,
                            xgb_weight: float = 0.5) -> np.ndarray:
    """Weighted ensemble prediction."""
    lgb_weight = 1.0 - xgb_weight
    p_xgb = xgb_model.predict_proba(X)[:, 1]
    p_lgb = lgb_model.predict_proba(X)[:, 1]
    return xgb_weight * p_xgb + lgb_weight * p_lgb


def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=["Safe", "Scam"]))
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    print(f"   ROC-AUC:  {auc:.4f}")
    print(f"   Avg Prec: {ap:.4f}")
    return auc, ap


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    os.makedirs("models", exist_ok=True)

    # 1. Load data
    dfs = []
    dfs.append(download_phishtank())
    dfs.append(download_openphish())
    dfs.append(load_alexa_safe_urls(n=50000))
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["url"]).drop_duplicates(subset=["url"])

    print(f"\n📦 Total dataset: {len(df)} URLs")
    print(f"   Phishing: {df['label'].sum()}")
    print(f"   Safe:     {(df['label'] == 0).sum()}")

    # 2. Extract features
    X, y = build_feature_matrix(df)

    # 3. Train/val/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.11, stratify=y_temp, random_state=42
    )
    print(f"\n✂️  Split → train:{len(X_train)}  val:{len(X_val)}  test:{len(X_test)}")

    # 4. Train
    xgb_model, lgb_model = train_ensemble(X_train, y_train, X_val, y_val)

    # 5. Evaluate ensemble on test
    y_prob = ensemble_predict_proba(xgb_model, lgb_model, X_test)
    auc, ap = evaluate(y_test, y_prob)

    # 6. Find optimal threshold (maximize F1)
    from sklearn.metrics import f1_score
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1s = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]
    best_threshold = float(thresholds[np.argmax(f1s)])
    print(f"\n🎯 Optimal threshold: {best_threshold:.2f}  (F1={max(f1s):.4f})")

    # 7. Feature importance
    feature_names = _feature_names()
    imp = pd.Series(xgb_model.feature_importances_, index=feature_names)
    print("\n📈 Top 10 feature importances (XGBoost):")
    print(imp.sort_values(ascending=False).head(10).to_string())

    # 8. Save models + metadata
    joblib.dump(xgb_model, "models/url_xgb.joblib")
    joblib.dump(lgb_model, "models/url_lgb.joblib")

    metadata = {
        "feature_names": feature_names,
        "best_threshold": best_threshold,
        "roc_auc": round(auc, 4),
        "avg_precision": round(ap, 4),
        "xgb_weight": 0.5,
        "model_version": "1.0.0",
    }
    with open("models/url_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n✅ Models saved to models/")
    print("   url_xgb.joblib, url_lgb.joblib, url_model_metadata.json")


if __name__ == "__main__":
    main()
