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
PLUGIN_ARTIFACTS_DIR = str(Path(DATA_DIR) / "plugin_artifacts")

GLOBAL_CONTEXT_FILE = str(Path(DATA_DIR) / "context.txt")
ENV_CONTEXT_FILE = str(Path(DATA_DIR) / "surrounding_context.txt")

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

ANIMATION_CONTEXT_MAX_CHARS = int(os.getenv("ANIMATION_CONTEXT_MAX_CHARS", "9000"))
ANIMATION_RENDER_TIMEOUT_SECONDS = int(os.getenv("ANIMATION_RENDER_TIMEOUT_SECONDS", "180"))


SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
SSE_MEDIA_TYPE = "text/event-stream"

TESSERACT_PATH = os.getenv("TESSERACT_PATH", "")

DEFAULT_ANALYSIS_MESSAGE = "No analysis has been generated yet."
API_EMPTY_RESPONSE_MESSAGE = "No response content returned by the API."

ERR_SARVAM_NOT_CONFIGURED = "Sarvam API not configured. Please set SARVAMAI_KEY."

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # one config for all
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")       # only used for Ollama/custom
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS") or "1200")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE") or "0.7")
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT") or None


ERR_LLM_NOT_CONFIGURED = "LLM provider not configured. Set LLM_API_KEY or SARVAMAI_KEY."
ERR_NO_PDF_UPLOADED = "No PDF uploaded yet."
ERR_NO_CONTEXT = "No analysis context found. Generate it first."


def validate_config() -> None:
    if not Path(TEMPLATES_DIR).exists():
        raise RuntimeError(f"Templates directory not found: {TEMPLATES_DIR}")
    if not Path(STATIC_DIR).exists():
        raise RuntimeError(f"Static directory not found: {STATIC_DIR}")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PLUGIN_ARTIFACTS_DIR, exist_ok=True)
