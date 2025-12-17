#!/usr/bin/env python3
"""
Quick script to check if environment variables are set correctly
Run this before starting the server to diagnose issues
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print("=" * 60)
print("Environment Variables Check")
print("=" * 60)

# Required variables
required_vars = {
    "GROQ_API_KEY": "Required for LLM queries",
}

# Optional variables
optional_vars = {
    "HUGGINGFACE_API_KEY": "Optional - For private Hugging Face models (public models work without it)",
    "EMBEDDING_MODEL": "Optional - Default: BAAI/bge-small-en-v1.5",
    "QDRANT_URL": "Optional - Qdrant Cloud URL (falls back to in-memory)",
    "QDRANT_API_KEY": "Optional - Qdrant Cloud API key",
    "GROQ_MODEL": "Optional - Default: llama-3.3-70b-versatile",
    "PORT": "Optional - Default: 8000",
    "HOST": "Optional - Default: 0.0.0.0",
    "LOG_LEVEL": "Optional - Default: INFO",
    "ENVIRONMENT": "Optional - Default: development",
    "ALLOWED_ORIGINS": "Optional - Default: *",
}

print("\nRequired Variables:")
print("-" * 60)
all_required_set = True
for var, description in required_vars.items():
    value = os.getenv(var)
    if value:
        # Mask the key (show first 4 and last 4 chars)
        masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
        print(f"[OK] {var}: {masked}")
    else:
        print(f"[MISSING] {var}: MISSING - {description}")
        all_required_set = False

print("\nOptional Variables:")
print("-" * 60)
for var, description in optional_vars.items():
    value = os.getenv(var)
    if value:
        print(f"[SET] {var}: {value}")
    else:
        print(f"[NOT SET] {var}: Not set - {description}")

print("\n" + "=" * 60)
if all_required_set:
    print("[SUCCESS] All required variables are set!")
    print("[INFO] You can now start the server with: uvicorn main:app --reload")
else:
    print("[ERROR] Some required variables are missing!")
    print("[INFO] Please set them in your .env file or environment")
    print("[INFO] See TROUBLESHOOTING.md for help")

print("=" * 60)
