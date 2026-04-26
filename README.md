# GyanDeep

GyanDeep is an AI learning platform for Nepali high‑school students. It lets students upload a textbook PDF, view pages on the left, and ask questions on the right. OCR and embeddings are precomputed so the assistant can answer using either the current page window or whole‑book retrieval.

**Core features**
- PDF viewer with page controls and upload flow
- OCR (Tesseract) for scanned pages
- Whole‑book embeddings stored in Postgres + pgvector
- Chat UI with streamed responses and a separate “Thinking” panel
- Multi‑book catalog with persisted metadata
- Animation generation via Manim plugin jobs (script + MP4 artifacts)

## Requirements
- Python 3.11+ (3.12 works)
- Docker + Docker Compose
- Tesseract OCR installed and available in `PATH`
- (macOS) `brew install tesseract`
- Optional for animation rendering: Manim CE + LaTeX + ffmpeg
  - macOS: `brew install tesseract`
  - Arch Linux: `sudo pacman -S tesseract tesseract-data-eng`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr tesseract-data-eng`
- Rust Toolchain *(to compile the high-performance Rust extension during install)*

## Quick Start
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Build the lightning-fast Rust extension in release mode:
   ```bash
   pip install maturin
   maturin develop --release -m rust/Cargo.toml
   ```
4. Create `.env` from the example and choose an LLM provider:
   ```bash
   cp .env.example .env
   ```
5. Edit `.env` using one of the LLM setup examples below.
6. Start services (DB + app):
   ```bash
   ./start_services.sh
   ```
7. Open the app at [http://localhost:8000](http://localhost:8000).

## LLM Setup
GyanDeep reads one unified set of `LLM_*` environment variables. Use one of these setups in your `.env`.

### Ollama
Use Ollama when you want to run a local model.

Install and start Ollama, then pull the model:
```bash
ollama serve
ollama pull llama3
```

Set `.env`:
```env
LLM_PROVIDER=ollama
LLM_API_KEY=dummy
LLM_MODEL=llama3
LLM_MAX_TOKENS=1200
LLM_TEMPERATURE=0.7
LLM_BASE_URL=http://localhost:11434
LLM_REASONING_EFFORT=
```

For Ollama, `LLM_API_KEY` can be any dummy value. `LLM_MODEL` must match a model available in your local Ollama installation.

### OpenAI
Use OpenAI when you want hosted OpenAI models.

Set `.env`:
```env
LLM_PROVIDER=openai
LLM_API_KEY=your_openai_api_key
LLM_MODEL=gpt-4o-mini
LLM_MAX_TOKENS=1200
LLM_TEMPERATURE=0.7
LLM_BASE_URL=
LLM_REASONING_EFFORT=
```

For OpenAI, leave `LLM_BASE_URL` empty.

### Sarvam
Use Sarvam when you want Sarvam-hosted models.

Set `.env`:
```env
LLM_PROVIDER=sarvam
LLM_API_KEY=your_sarvam_api_key
LLM_MODEL=sarvam-m
LLM_MAX_TOKENS=1200
LLM_TEMPERATURE=0.5
LLM_BASE_URL=
LLM_REASONING_EFFORT=medium
```

`LLM_REASONING_EFFORT` is only used by Sarvam. Common values are `low`, `medium`, and `high`.

## How It Works
- **Upload** a PDF from the dashboard.
- **OCR** runs automatically for pages without native text.
- **Embeddings** are generated in batches and upserted into pgvector.
- **Ask** a question:
  - *Current page mode* builds a structured (paraphrased) summary of ±5 pages for cleaner context.
  - *Whole book mode* retrieves top‑K chunks from pgvector.
- **Animate** from the chat area:
  - Click **Animate** to create an async plugin job.
  - Job progress is logged in chat.
  - On success, the app returns links to the generated `script.py` and rendered `lesson.mp4`.

## Ingestion Scripts
The one‑off ingestion utilities live under `core/services/ingestion`:
- `pdf_ocr.py` – extract OCR text from a PDF to a text file
- `embedding_pipeline.py` – chunk a text file, generate embeddings, and upsert

Example usage:
```bash
python -m core.services.ingestion.pdf_ocr path/to/book.pdf --out totalBook.txt
python -m core.services.ingestion.embedding_pipeline totalBook.txt --source "grade-5-science"
```

## Configuration
See `.env.example` for all settings. Common ones:
- `LLM_PROVIDER` – `ollama`, `openai`, `sarvam`, `anthropic`, `gemini`, or `openrouter`
- `LLM_API_KEY` – provider API key; use `dummy` for local Ollama
- `LLM_MODEL` – model name to use for chat responses
- `LLM_BASE_URL` – only needed for Ollama or custom provider endpoints
- `LLM_MAX_TOKENS`, `LLM_TEMPERATURE`, `LLM_REASONING_EFFORT`
- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`
- `PRECOMPUTE_OCR_ON_UPLOAD` (default: true)
- `PRECOMPUTE_EMBEDDINGS_ON_UPLOAD` (default: true)
- `EMBEDDING_PROVIDER` (`sentence_transformers` or `openai`)
- `EMBEDDING_MODEL_NAME` (default: `all-MiniLM-L6-v2`)
- `ANIMATION_CONTEXT_MAX_CHARS` (default: `9000`)
- `ANIMATION_RENDER_TIMEOUT_SECONDS` (default: `180`)

## Troubleshooting
- **DB auth errors**: ensure `.env` has `DB_PASSWORD` and the docker service matches it. If you changed it, remove the old volume and restart:
  ```bash
  docker compose -f core/services/storage/docker/docker-compose.yaml down -v
  ./start_services.sh
  ```
- **OCR missing**: verify Tesseract is installed and available on your shell `PATH`.

## Project Structure (high level)
- `dashboard/backend` – FastAPI backend + OCR + embeddings
- `dashboard/frontend` – HTML/CSS/JS app
- `core/models` – Pydantic models aligned to the DB schema
- `core/agents` – prompt + context managers (agent-ready)
- `core/services/ingestion` – OCR + embedding utilities
- `core/services/inference` – AI inference wrapper (model calls + parsing)
- `core/services/plugins` – plugin runtime + Manim animation plugin
- `core/services/storage` – schema + pgvector helpers
- `tests/` – unit tests

---
Made for Nepal’s classrooms, with students in mind.
