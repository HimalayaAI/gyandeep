import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
SARVAMAI_KEY = os.getenv("SARVAMAI_KEY")
API_KEY_PLACEHOLDER = "your_key_here"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
UPLOAD_DIR = "uploads"
STATIC_DIR = "static"
TEMPLATES_DIR = "templates"
TESSERACT_PATH = os.getenv(
    "TESSERACT_PATH",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)

# ---------------------------------------------------------------------------
# Sarvam AI
# ---------------------------------------------------------------------------
SARVAM_MODEL = "sarvam-30b"
SARVAM_MAX_TOKENS = int(os.getenv("SARVAM_MAX_TOKENS", "16384"))

# ---------------------------------------------------------------------------
# OCR settings
# ---------------------------------------------------------------------------
OCR_MIN_TEXT_LENGTH = 20
OCR_DPI = 150
OCR_FALLBACK_MESSAGE = "[Image could not be parsed by Tesseract]"

# ---------------------------------------------------------------------------
# Context / analysis
# ---------------------------------------------------------------------------
CONTEXT_WINDOW = 5
GLOBAL_CONTEXT_FILE = "context.txt"
ENV_CONTEXT_FILE = "surrounding_context.txt"
DEFAULT_ANALYSIS_MESSAGE = (
    "No global analysis has been generated yet for this document."
)
API_EMPTY_RESPONSE_MESSAGE = (
    "[Error: API returned empty or rejected response for this context]"
)

# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------
OCR_SEMAPHORE_LIMIT = 20
ANALYSIS_CHUNK_SIZE = 20

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
SSE_MEDIA_TYPE = "text/event-stream"

# ---------------------------------------------------------------------------
# Error messages (repeated across endpoints)
# ---------------------------------------------------------------------------
ERR_SARVAM_NOT_CONFIGURED = (
    "Sarvam AI client not configured. Check your .env API Key"
)
ERR_NO_PDF_UPLOADED = "No PDF uploaded"
ERR_NO_CONTEXT = (
    "No context data found! Please securely run the equivalent "
    "Analysis button first to extract and generate the data."
)

# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------
_REQUIRED_ENV_VARS = ["SARVAMAI_KEY"]


def validate_config() -> None:
    """Check all required environment variables are present at app startup.

    Raises ``RuntimeError`` with a descriptive message listing every
    missing or placeholder variable.
    """
    missing = []
    for var in _REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if not value or value == API_KEY_PLACEHOLDER:
            missing.append(var)
    if missing:
        raise RuntimeError(
            f"Missing or placeholder environment variable(s): "
            f"{', '.join(missing)}. "
            f"Set them in your .env file or shell environment before "
            f"starting the app."
        )
