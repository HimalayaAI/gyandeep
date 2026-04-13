"""Dashboard configuration for Gyandeep."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    load_dotenv(dotenv_path=env_path)

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = str(FRONTEND_DIR / "static")
ASSETS_DIR = str(FRONTEND_DIR / "assets")
TEMPLATES_DIR = str(FRONTEND_DIR)
UPLOAD_DIR = str(BASE_DIR / "uploads")
DATA_DIR = str(BASE_DIR / "data")

GLOBAL_CONTEXT_FILE = str(Path(DATA_DIR) / "context.txt")
ENV_CONTEXT_FILE = str(Path(DATA_DIR) / "surrounding_context.txt")

SARVAMAI_KEY = os.getenv("SARVAMAI_KEY", "")
API_KEY_PLACEHOLDER = "YOUR_SARVAM_API_KEY"
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "sarvam-m")
SARVAM_MAX_TOKENS = int(os.getenv("SARVAM_MAX_TOKENS", "1200"))
_reasoning_effort = os.getenv("SARVAM_REASONING_EFFORT", "medium").strip()
SARVAM_REASONING_EFFORT = _reasoning_effort if _reasoning_effort else None
_default_temp = "0.5" if SARVAM_REASONING_EFFORT else "0.2"
SARVAM_TEMPERATURE = float(os.getenv("SARVAM_TEMPERATURE", _default_temp))
MODEL_CONTEXT_WINDOW = int(os.getenv("MODEL_CONTEXT_WINDOW", "7192"))
CONTEXT_SAFETY_TOKENS = int(os.getenv("CONTEXT_SAFETY_TOKENS", "200"))
CONTEXT_TOKEN_CHAR_RATIO = float(os.getenv("CONTEXT_TOKEN_CHAR_RATIO", "3.0"))
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "800"))

OCR_MIN_TEXT_LENGTH = int(os.getenv("OCR_MIN_TEXT_LENGTH", "40"))
OCR_DPI = int(os.getenv("OCR_DPI", "200"))
OCR_FALLBACK_MESSAGE = os.getenv("OCR_FALLBACK_MESSAGE", "OCR failed on this page.")
OCR_SEMAPHORE_LIMIT = int(os.getenv("OCR_SEMAPHORE_LIMIT", "4"))

CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "5"))
ANALYSIS_CHUNK_SIZE = int(os.getenv("ANALYSIS_CHUNK_SIZE", "10"))

PRECOMPUTE_OCR_ON_UPLOAD = os.getenv("PRECOMPUTE_OCR_ON_UPLOAD", "true").lower() in {"1", "true", "yes"}
PRECOMPUTE_EMBEDDINGS_ON_UPLOAD = os.getenv("PRECOMPUTE_EMBEDDINGS_ON_UPLOAD", "true").lower() in {"1", "true", "yes"}
EMBEDDING_SOURCE_PREFIX = os.getenv("EMBEDDING_SOURCE_PREFIX", "upload")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
EMBEDDING_WARMUP = os.getenv("EMBEDDING_WARMUP", "false").lower() in {"1", "true", "yes"}


SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
SSE_MEDIA_TYPE = "text/event-stream"

TESSERACT_PATH = os.getenv("TESSERACT_PATH", "")
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", str(50 * 1024 * 1024)))  # 50 MB

DEFAULT_ANALYSIS_MESSAGE = "No analysis has been generated yet."
API_EMPTY_RESPONSE_MESSAGE = "No response content returned by the API."

ERR_SARVAM_NOT_CONFIGURED = "Sarvam API not configured. Please set SARVAMAI_KEY."
ERR_NO_PDF_UPLOADED = "No PDF uploaded yet."
ERR_NO_CONTEXT = "No analysis context found. Generate it first."


def validate_config() -> None:
    if not Path(TEMPLATES_DIR).exists():
        raise RuntimeError(f"Templates directory not found: {TEMPLATES_DIR}")
    if not Path(STATIC_DIR).exists():
        raise RuntimeError(f"Static directory not found: {STATIC_DIR}")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
