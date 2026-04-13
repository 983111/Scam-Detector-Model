"""
URL Feature Extractor for Scam Detection
=========================================
Extracts 20+ heuristic features from URLs without any API calls.
Used by Model A (URL Scam Detection).
"""

import re
import math
import urllib.parse
from collections import Counter
from typing import Dict, List


# Suspicious keywords commonly found in phishing URLs
SUSPICIOUS_KEYWORDS = [
    "login", "signin", "sign-in", "secure", "verify", "verification",
    "account", "update", "confirm", "banking", "paypal", "amazon",
    "apple", "google", "microsoft", "netflix", "ebay", "wallet",
    "password", "credential", "auth", "authenticate", "recover",
    "unlock", "suspend", "limited", "urgent", "alert", "warning",
    "free", "prize", "winner", "claim", "reward", "gift",
]

# Trusted TLDs (less suspicious)
TRUSTED_TLDS = {".com", ".org", ".net", ".edu", ".gov", ".io", ".co"}

# Suspicious TLDs often used in phishing
SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".click", ".tk", ".ml", ".ga", ".cf",
    ".gq", ".work", ".loan", ".online", ".site", ".website",
    ".club", ".win", ".download", ".stream", ".review",
}

# Well-known brands to detect impersonation
BRAND_NAMES = [
    "paypal", "amazon", "apple", "google", "microsoft", "netflix",
    "facebook", "instagram", "twitter", "whatsapp", "telegram",
    "bank", "chase", "wellsfargo", "citibank", "hdfc", "sbi",
    "ebay", "shopify", "dropbox", "linkedin", "uber", "ola",
]


def shannon_entropy(string: str) -> float:
    """Calculate Shannon entropy of a string (measures randomness)."""
    if not string:
        return 0.0
    freq = Counter(string)
    length = len(string)
    entropy = -sum((count / length) * math.log2(count / length)
                   for count in freq.values())
    return round(entropy, 4)


def extract_url_features(url: str) -> Dict[str, float]:
    """
    Extract all features from a URL.
    Returns a dict of feature_name -> float value.
    """
    features = {}

    # ── Normalize ──────────────────────────────────────────────────────────
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        # Return high-risk defaults on parse failure
        return {k: 1.0 for k in _feature_names()}

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    full_url = url.lower()
    domain_lower = hostname.lower()

    # ── Length features ─────────────────────────────────────────────────────
    features["url_length"] = min(len(url) / 200.0, 1.0)          # normalized 0-1
    features["hostname_length"] = min(len(hostname) / 50.0, 1.0)
    features["path_length"] = min(len(path) / 100.0, 1.0)
    features["query_length"] = min(len(query) / 100.0, 1.0)

    # ── Subdomain / dot count ───────────────────────────────────────────────
    dot_count = domain_lower.count(".")
    features["dot_count"] = min(dot_count / 5.0, 1.0)
    features["subdomain_count"] = min(max(dot_count - 1, 0) / 4.0, 1.0)

    # ── IP address presence ─────────────────────────────────────────────────
    ip_pattern = re.compile(
        r"^(\d{1,3}\.){3}\d{1,3}$|"
        r"^\[?[0-9a-fA-F:]+\]?$"
    )
    features["has_ip_address"] = 1.0 if ip_pattern.match(hostname) else 0.0

    # ── HTTPS ───────────────────────────────────────────────────────────────
    features["is_https"] = 1.0 if parsed.scheme == "https" else 0.0

    # ── Suspicious keywords ─────────────────────────────────────────────────
    keyword_hits = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in full_url)
    features["suspicious_keyword_count"] = min(keyword_hits / 5.0, 1.0)
    features["has_suspicious_keyword"] = 1.0 if keyword_hits > 0 else 0.0

    # ── TLD features ────────────────────────────────────────────────────────
    tld = "." + domain_lower.split(".")[-1] if "." in domain_lower else ""
    features["is_suspicious_tld"] = 1.0 if tld in SUSPICIOUS_TLDS else 0.0
    features["is_trusted_tld"] = 1.0 if tld in TRUSTED_TLDS else 0.0

    # ── Entropy (randomness) ────────────────────────────────────────────────
    # Extract registrable domain (e.g., "xk92mq3p" from "xk92mq3p.xyz")
    parts = domain_lower.split(".")
    registrable = parts[-2] if len(parts) >= 2 else domain_lower
    features["domain_entropy"] = min(shannon_entropy(registrable) / 4.5, 1.0)
    features["path_entropy"] = min(shannon_entropy(path) / 5.0, 1.0)

    # ── Special character counts ─────────────────────────────────────────────
    features["hyphen_count"] = min(domain_lower.count("-") / 5.0, 1.0)
    features["at_symbol"] = 1.0 if "@" in url else 0.0
    features["double_slash_redirect"] = 1.0 if url.count("//") > 1 else 0.0
    features["hex_encoding"] = 1.0 if "%" in url else 0.0

    # ── Brand impersonation ─────────────────────────────────────────────────
    brand_hits = sum(1 for brand in BRAND_NAMES if brand in domain_lower)
    # If brand is in subdomain but not the registrable domain -> impersonation
    is_impersonation = 0.0
    for brand in BRAND_NAMES:
        if brand in domain_lower:
            # Impersonation = brand in subdomain, not as actual domain
            if brand not in registrable:
                is_impersonation = 1.0
                break
    features["brand_impersonation"] = is_impersonation
    features["brand_keyword_in_url"] = min(brand_hits / 3.0, 1.0)

    # ── Digit ratio in domain ───────────────────────────────────────────────
    digit_ratio = sum(c.isdigit() for c in registrable) / max(len(registrable), 1)
    features["digit_ratio_in_domain"] = round(digit_ratio, 4)

    # ── URL shortener detection ─────────────────────────────────────────────
    shorteners = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
                  "short.link", "cutt.ly", "rb.gy", "is.gd", "clck.ru"}
    features["is_url_shortener"] = 1.0 if domain_lower in shorteners else 0.0

    # ── Abnormal port ───────────────────────────────────────────────────────
    port = parsed.port
    features["has_abnormal_port"] = 0.0
    if port and port not in (80, 443, 8080, 8443):
        features["has_abnormal_port"] = 1.0

    return features


def _feature_names() -> List[str]:
    """Return all feature names (for schema validation)."""
    return [
        "url_length", "hostname_length", "path_length", "query_length",
        "dot_count", "subdomain_count", "has_ip_address", "is_https",
        "suspicious_keyword_count", "has_suspicious_keyword",
        "is_suspicious_tld", "is_trusted_tld", "domain_entropy",
        "path_entropy", "hyphen_count", "at_symbol", "double_slash_redirect",
        "hex_encoding", "brand_impersonation", "brand_keyword_in_url",
        "digit_ratio_in_domain", "is_url_shortener", "has_abnormal_port",
    ]


def features_to_vector(features: Dict[str, float]) -> List[float]:
    """Convert feature dict to ordered list for ML model input."""
    return [features.get(name, 0.0) for name in _feature_names()]


if __name__ == "__main__":
    # Quick test
    test_urls = [
        "https://google.com",
        "http://192.168.1.1/login/verify-account",
        "https://paypal-secure-verify.xyz/login?token=abc123",
        "http://amaz0n-account-update.tk/signin",
        "https://github.com/user/repo",
    ]
    for url in test_urls:
        f = extract_url_features(url)
        print(f"\nURL: {url}")
        print(f"  Entropy: {f['domain_entropy']:.2f}  |  "
              f"IP: {f['has_ip_address']}  |  "
              f"Impersonation: {f['brand_impersonation']}  |  "
              f"Susp. TLD: {f['is_suspicious_tld']}  |  "
              f"Keywords: {f['has_suspicious_keyword']}")
