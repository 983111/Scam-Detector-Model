"""
Message Scam Detection — DistilBERT Fine-tuning
================================================
Model B: Classifies SMS/email text as scam/not-scam.
Also outputs scam TYPE: phishing / urgency / impersonation / financial / safe

Run:
    pip install transformers datasets torch scikit-learn pandas tqdm accelerate
    python train_message_model.py
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertConfig,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
#  LABEL DEFINITIONS
# ─────────────────────────────────────────────────────────────

# Binary labels
BINARY_LABELS = {0: "safe", 1: "scam"}

# Type labels (multi-class)
TYPE_LABELS = {
    0: "safe",
    1: "phishing",        # credential/link theft
    2: "urgency",         # "act now or lose account"
    3: "impersonation",   # pretending to be a brand/person
    4: "financial",       # lottery, prizes, investment fraud
}

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 4
LR = 2e-5


# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_sms_spam_dataset() -> pd.DataFrame:
    """
    Load UCI SMS Spam Collection dataset.
    Download from: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
    Or use HuggingFace datasets (automatic):
      datasets.load_dataset("sms_spam")
    """
    try:
        from datasets import load_dataset
        print("📥 Loading SMS Spam dataset from HuggingFace...")
        ds = load_dataset("sms_spam", split="train")
        df = pd.DataFrame({"text": ds["sms"], "label": ds["label"]})
        # label: 0=ham (safe), 1=spam (scam)
        print(f"   ✅ SMS Spam: {len(df)} messages")
        return df
    except Exception as e:
        print(f"   ⚠️  HuggingFace SMS Spam failed ({e}). Using synthetic.")
        return _synthetic_messages()


def load_enron_phishing() -> pd.DataFrame:
    """
    Enron-based phishing email dataset.
    Kaggle: https://www.kaggle.com/datasets/rtatman/fraudulent-email-corpus
    Also try: datasets.load_dataset("zefang-liu/phishing-email-dataset")
    """
    try:
        from datasets import load_dataset
        print("📥 Loading phishing email dataset...")
        ds = load_dataset("zefang-liu/phishing-email-dataset", split="train")
        df = pd.DataFrame({
            "text": ds["email_text"],
            "label": ds["label"]
        })
        print(f"   ✅ Phishing emails: {len(df)}")
        return df
    except Exception as e:
        print(f"   ⚠️  Phishing email dataset failed ({e}). Using synthetic.")
        return pd.DataFrame()


def _synthetic_messages() -> pd.DataFrame:
    """
    Comprehensive synthetic dataset for training/testing.
    In production: supplement with 50k+ real samples.
    """
    scam_messages = [
        # Phishing
        ("Your PayPal account has been limited. Verify now: http://paypal-secure.tk/verify", 1, "phishing"),
        ("Click here to confirm your Amazon order or it will be cancelled: bit.ly/amz123", 1, "phishing"),
        ("Your Apple ID was used in a new login. Confirm identity: apple-id-verify.xyz", 1, "phishing"),
        ("HDFC Bank: Your account will be suspended. Update KYC: hdfc-kyc.top/update", 1, "phishing"),
        ("Dear customer, your Netflix subscription failed. Update payment: netflix-billing.click", 1, "phishing"),

        # Urgency
        ("URGENT: Your account will be deleted in 24 hours unless you act NOW!", 1, "urgency"),
        ("WARNING: Suspicious activity detected. Call us immediately: 1-800-SCAM-NOW", 1, "urgency"),
        ("Last chance! Your prize expires in 1 hour. Claim before midnight!", 1, "urgency"),
        ("ALERT: Your debit card has been blocked. Call customer care IMMEDIATELY.", 1, "urgency"),
        ("Time-sensitive: Verify your identity in the next 30 minutes or lose access.", 1, "urgency"),

        # Impersonation
        ("Hi, this is John from Microsoft. Your computer has a virus, call 1-888-123-4567", 1, "impersonation"),
        ("Message from SBI Bank: Your FD is due. Contact our agent at 9876543210.", 1, "impersonation"),
        ("Google Security Team: We detected unusual sign-in. Reply YES to secure account.", 1, "impersonation"),
        ("This is IRS. You owe $3,200 in back taxes. Call now to avoid arrest.", 1, "impersonation"),
        ("Amazon Customer Service: Your refund of $450 is ready. Click to claim.", 1, "impersonation"),

        # Financial fraud
        ("Congratulations! You've won £1,000,000 in the UK lottery. Send your bank details.", 1, "financial"),
        ("Investment opportunity: 300% returns guaranteed. WhatsApp us now!", 1, "financial"),
        ("You are selected for a government scheme. Get Rs 50,000. Apply: gov-scheme.top", 1, "financial"),
        ("Dear winner, claim your $500 Walmart gift card. Survey complete: survey-reward.xyz", 1, "financial"),
        ("Make $5000/week from home! No experience needed. Click to join our team.", 1, "financial"),

        # More varied scam messages
        ("Your OTP is 483920. Never share this. [But actually provide to agent]", 1, "phishing"),
        ("Bank transfer of $10,000 requires your PIN confirmation. Reply with PIN.", 1, "phishing"),
        ("Free iPhone 15 for survey participants! Limited offer: iphone-free.tk/claim", 1, "financial"),
        ("Your parcel is held at customs. Pay Rs 200 fee: customs-india.click/pay", 1, "phishing"),
        ("We're sending you a check for $4500. Just pay $50 processing fee first.", 1, "financial"),
    ]

    safe_messages = [
        ("Your order #12345 has been shipped. Track at amazon.com/orders", 0, "safe"),
        ("Hi! Are you free for coffee tomorrow at 3pm?", 0, "safe"),
        ("Meeting rescheduled to Monday 10am. Please confirm attendance.", 0, "safe"),
        ("Your OTP for login is 847291. Valid for 10 minutes. Do not share.", 0, "safe"),
        ("Reminder: Your dentist appointment is on Thursday at 2:30 PM.", 0, "safe"),
        ("Thanks for your purchase! Your receipt is attached.", 0, "safe"),
        ("The package you ordered will arrive by Friday.", 0, "safe"),
        ("Happy birthday! Hope you have a wonderful day.", 0, "safe"),
        ("Your subscription renews on May 1. Manage at account.netflix.com", 0, "safe"),
        ("Team lunch is at 1pm today. See you there!", 0, "safe"),
        ("Please review the attached Q3 report before tomorrow's meeting.", 0, "safe"),
        ("Your flight AA1234 departs at 6:45 AM. Check in at terminal B.", 0, "safe"),
        ("New message from your doctor. Login to patient portal to view.", 0, "safe"),
        ("Sale starts Friday! Up to 40% off. Visit our store.", 0, "safe"),
        ("Your electricity bill of Rs 1,240 is due on 25th. Pay at MSEDCL website.", 0, "safe"),
        ("Budget approved. Please proceed with the vendor contract.", 0, "safe"),
        ("Groceries delivered! Rate your experience on the app.", 0, "safe"),
        ("Your gym membership renews on the 1st. Rs 999 will be debited.", 0, "safe"),
        ("Class canceled tomorrow due to weather. Stay safe!", 0, "safe"),
        ("Your video call is starting in 5 minutes. Join at meet.google.com/xyz", 0, "safe"),
    ]

    rows = []
    for (text, label, type_) in scam_messages + safe_messages:
        rows.append({"text": text, "label": label, "scam_type": type_})

    # Expand with augmented variations
    df = pd.DataFrame(rows)
    return df


def assign_scam_types(df: pd.DataFrame) -> pd.DataFrame:
    """Heuristically assign scam types to messages without type labels."""
    urgency_keywords = ["urgent", "immediately", "act now", "24 hours",
                        "expire", "last chance", "warning", "alert"]
    phishing_keywords = ["verify", "confirm", "update", "login", "click here",
                         "otp", "pin", "password", "kyc", "account suspended"]
    financial_keywords = ["won", "winner", "prize", "lottery", "reward",
                          "cash", "money", "investment", "returns", "earn",
                          "gift card", "check", "payment"]
    impersonation_keywords = ["microsoft", "amazon", "google", "apple", "sbi",
                              "hdfc", "irs", "fbi", "bank", "support team"]

    def _classify(row):
        if row["label"] == 0:
            return "safe"
        text = row["text"].lower()
        if any(kw in text for kw in phishing_keywords):
            return "phishing"
        if any(kw in text for kw in urgency_keywords):
            return "urgency"
        if any(kw in text for kw in impersonation_keywords):
            return "impersonation"
        if any(kw in text for kw in financial_keywords):
            return "financial"
        return "phishing"  # default scam type

    if "scam_type" not in df.columns:
        df["scam_type"] = df.apply(_classify, axis=1)
    return df


# ─────────────────────────────────────────────────────────────
#  DATASET CLASS
# ─────────────────────────────────────────────────────────────

TYPE_TO_ID = {v: k for k, v in TYPE_LABELS.items()}


class ScamDataset(Dataset):
    def __init__(self, texts: List[str], binary_labels: List[int],
                 type_labels: List[int], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt"
        )
        self.binary_labels = torch.tensor(binary_labels, dtype=torch.long)
        self.type_labels = torch.tensor(type_labels, dtype=torch.long)

    def __len__(self):
        return len(self.binary_labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "binary_label": self.binary_labels[idx],
            "type_label": self.type_labels[idx],
        }


# ─────────────────────────────────────────────────────────────
#  MULTI-TASK MODEL
# ─────────────────────────────────────────────────────────────

class ScamDetectorModel(nn.Module):
    """
    DistilBERT with dual classification heads:
    - Head 1: Binary (scam / not-scam)
    - Head 2: Type (phishing / urgency / impersonation / financial / safe)
    """

    def __init__(self, num_types: int = 5, dropout: float = 0.3):
        super().__init__()
        self.distilbert = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        )
        hidden_size = self.distilbert.config.dim  # 768

        # Replace single head with shared backbone + two heads
        self.distilbert.classifier = nn.Identity()
        self.distilbert.pre_classifier = nn.Identity()

        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(512, 2)
        self.type_head = nn.Linear(512, num_types)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        shared = self.shared(hidden)
        binary_logits = self.binary_head(shared)
        type_logits = self.type_head(shared)
        return binary_logits, type_logits


# ─────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    binary_criterion = nn.CrossEntropyLoss()
    type_criterion = nn.CrossEntropyLoss()

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        binary_labels = batch["binary_label"].to(device)
        type_labels = batch["type_label"].to(device)

        optimizer.zero_grad()
        binary_logits, type_logits = model(input_ids, attn_mask)

        loss_binary = binary_criterion(binary_logits, binary_labels)
        loss_type = type_criterion(type_logits, type_labels)
        loss = loss_binary + 0.5 * loss_type  # weight type loss lower

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_model(model, loader, device):
    model.eval()
    all_binary_preds, all_binary_labels = [], []
    all_type_preds, all_type_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            binary_labels = batch["binary_label"].cpu().numpy()
            type_labels = batch["type_label"].cpu().numpy()

            binary_logits, type_logits = model(input_ids, attn_mask)
            binary_preds = binary_logits.argmax(dim=1).cpu().numpy()
            type_preds = type_logits.argmax(dim=1).cpu().numpy()

            all_binary_preds.extend(binary_preds)
            all_binary_labels.extend(binary_labels)
            all_type_preds.extend(type_preds)
            all_type_labels.extend(type_labels)

    binary_f1 = f1_score(all_binary_labels, all_binary_preds, average="binary")
    print("\n📊 Binary Classification:")
    print(classification_report(all_binary_labels, all_binary_preds,
                                target_names=["Safe", "Scam"]))
    print("📊 Scam Type Classification:")
    print(classification_report(all_type_labels, all_type_preds,
                                target_names=list(TYPE_LABELS.values())))
    return binary_f1


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    os.makedirs("models/message_model", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # 1. Load data
    dfs = [load_sms_spam_dataset()]
    email_df = load_enron_phishing()
    if len(email_df):
        dfs.append(email_df)
    dfs.append(_synthetic_messages())

    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["text"])
    df = assign_scam_types(df)

    print(f"\n📦 Total messages: {len(df)}")
    print(df["scam_type"].value_counts())

    # 2. Encode labels
    df["type_id"] = df["scam_type"].map(TYPE_TO_ID).fillna(0).astype(int)

    # 3. Train/val split
    train_df, val_df = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=42
    )
    print(f"\n✂️  Train: {len(train_df)}  Val: {len(val_df)}")

    # 4. Tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # 5. Datasets & loaders
    train_dataset = ScamDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        train_df["type_id"].tolist(),
        tokenizer, MAX_LENGTH,
    )
    val_dataset = ScamDataset(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        val_df["type_id"].tolist(),
        tokenizer, MAX_LENGTH,
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 6. Model + optimizer
    model = ScamDetectorModel(num_types=len(TYPE_LABELS)).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    # 7. Training loop
    best_f1 = 0
    for epoch in range(EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}")
        val_f1 = evaluate_model(model, val_loader, device)
        print(f"Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained = lambda path: torch.save(model.state_dict(), path)
            torch.save(model.state_dict(), "models/message_model/best_model.pt")
            tokenizer.save_pretrained("models/message_model/")
            print(f"   💾 Saved best model (F1={val_f1:.4f})")

    # 8. Save metadata
    metadata = {
        "base_model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "type_labels": TYPE_LABELS,
        "best_val_f1": round(best_f1, 4),
        "model_version": "1.0.0",
    }
    with open("models/message_model/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Training complete! Best F1: {best_f1:.4f}")
    print("   Saved to models/message_model/")


if __name__ == "__main__":
    main()
