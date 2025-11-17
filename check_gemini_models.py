"""
Utility: Check which Gemini models are available/usable for your API key.

Quick start (edit in code):
  - Open this file and set API_KEY below to your key string.

Other ways:
  - Set environment variable GEMINI_API_KEY, or pass --key YOUR_KEY
  - Optional: --no-validate skips per-model content generation checks

Examples:
  python check_gemini_models.py
  python check_gemini_models.py --key YOUR_KEY
  python check_gemini_models.py --no-validate
"""

import argparse
import os
import sys
import time
from typing import Dict, List

import requests
import getpass


MODELS_LIST_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEN_CONTENT_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Set your API key here to hardcode it (preferred for your request)
# Example: API_KEY = "AIza...your_key_here"
API_KEY = "Your API key"


def list_models(api_key: str, timeout: int = 15) -> List[Dict]:
    params = {"key": api_key}
    try:
        resp = requests.get(MODELS_LIST_URL, params=params, timeout=timeout)
        if resp.status_code != 200:
            print(f"Failed to list models: HTTP {resp.status_code}: {resp.text[:200]}")
            return []
        data = resp.json()
        return data.get("models", [])
    except Exception as e:
        print(f"Error listing models: {e}")
        return []


def validate_model(api_key: str, model: str, timeout: int = 15) -> bool:
    url = GEN_CONTENT_URL.format(model=model)
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Say OK"}]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 8
        }
    }
    try:
        resp = requests.post(f"{url}?key={api_key}", headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            # Basic check for candidates presence
            body = resp.json()
            return bool(body.get("candidates"))
        # Some models may exist but be unavailable to the key (403/404)
        return False
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Gemini model availability for an API key")
    parser.add_argument("--key", dest="api_key", help="Gemini API key (or set GEMINI_API_KEY)")
    parser.add_argument("--timeout", type=int, default=15, help="HTTP timeout seconds (default: 15)")
    parser.add_argument("--no-validate", action="store_true", help="Only list models; skip per-model generation test")
    args = parser.parse_args()

    # Priority: hardcoded API_KEY > --key flag > GEMINI_API_KEY env > interactive prompt
    api_key = API_KEY or args.api_key or os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        # Allow manual entry if nothing provided
        try:
            entered = getpass.getpass("Enter Gemini API key: ").strip()
        except Exception:
            entered = input("Enter Gemini API key: ").strip()
        api_key = entered
    if not api_key:
        print("Error: No API key provided.")
        return 2

    print("Listing models (v1beta)...")
    models = list_models(api_key, timeout=args.timeout)
    if not models:
        print("No models returned or request failed.")
        return 1

    # Stable/common text models to prioritize viewing first
    priority_names = {
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash-002",
        "gemini-1.5-flash",
        "gemini-1.5-pro-002",
        "gemini-1.5-pro",
        "gemini-pro",
    }

    # Sort: priority first, then alphabetically
    def sort_key(m: Dict) -> tuple:
        name = m.get("name", "")
        pri = 0 if name in priority_names else 1
        return (pri, name)

    models_sorted = sorted(models, key=sort_key)

    if args.no_validate:
        print("\nAvailable models (not validated):")
        for m in models_sorted:
            print(f"- {m.get('name','')}  (displayName={m.get('displayName','')})")
        return 0

    print("\nValidating model access with a tiny generateContent call...")
    usable: List[str] = []
    unusable: List[str] = []

    for idx, m in enumerate(models_sorted, 1):
        name = m.get("name", "")
        if not name:
            continue
        ok = validate_model(api_key, name, timeout=args.timeout)
        status = "OK" if ok else "NO ACCESS"
        print(f"[{idx:02d}] {name:30s} -> {status}")
        (usable if ok else unusable).append(name)
        # Be nice to the API
        time.sleep(0.1)

    print("\nSummary:")
    if usable:
        print("Usable models:")
        for n in usable:
            print(f"  - {n}")
    else:
        print("  (none)")

    if unusable:
        print("\nModels listed but not usable with this key:")
        for n in unusable:
            print(f"  - {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


