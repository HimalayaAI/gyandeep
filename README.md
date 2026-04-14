# GyanDeep

GyanDeep is an AI learning platform for Nepali high‑school students. It lets students upload a textbook PDF, view pages on the left, and ask questions on the right. OCR and embeddings are precomputed so the assistant can answer using either the current page window or whole‑book retrieval.

**Core features**
- PDF viewer with page controls and upload flow
- OCR (Tesseract) for scanned pages
- Whole‑book embeddings stored in Postgres + pgvector
- Chat UI with streamed responses and a separate “Thinking” panel
- Multi‑book catalog with persisted metadata
- ⚡ **Rust-Accelerated Batching**: Document chunking and pgvector serialization are processed in Rust (via PyO3/Rayon) for massive parallel performance.
- 📄 **Parallel OCR**: Tesseract operates concurrently to quickly parse entirely textless textbook pages.
- ✨ **Demo Mode**: Automatically falls back to raw text extraction if `SARVAMAI_KEY` is not provided.

## Requirements
- Python 3.11+ (3.12 works)
- Docker + Docker Compose
- Tesseract OCR installed and available in `PATH`
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
3. Build the lightning-fast Rust extension (`gyandeep_rs`) in release mode:
   ```bash
   pip install maturin
   maturin develop --release -m gyandeep_rs/Cargo.toml
   ```
4. Create `.env` from the example and add your API key:
   ```bash
   cp .env.example .env
   # Edit .env and set SARVAMAI_KEY=...
   # Note: If no key is set, the app will run in Demo Mode using extracted text fallback.
   ```
5. Start services (DB + app):
   ```bash
   ./start_services.sh
   ```
6. Open the app at [http://localhost:8000](http://localhost:8000).

## How It Works
- **Upload** a PDF from the dashboard.
- **OCR** runs automatically for pages without native text.
- **Embeddings** are generated in batches and upserted into pgvector.
- **Ask** a question:
  - *Current page mode* builds a structured (paraphrased) summary of ±5 pages for cleaner context.
  - *Whole book mode* retrieves top‑K chunks from pgvector.

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
- `SARVAMAI_KEY` – required for intelligent chat responses (bypassed if `ALLOW_NO_KEY_DEMO=true`)
- `ALLOW_NO_KEY_DEMO` (default: true) – fallbacks to raw text extraction context if no key is given.
- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`
- `PRECOMPUTE_OCR_ON_UPLOAD` (default: true)
- `PRECOMPUTE_EMBEDDINGS_ON_UPLOAD` (default: true)
- `EMBEDDING_PROVIDER` (`sentence_transformers` or `openai`)
- `EMBEDDING_MODEL_NAME` (default: `all-MiniLM-L6-v2`)

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
- `core/services/storage` – schema + pgvector helpers
- `tests/` – unit tests

---
Made for Nepal’s classrooms, with students in mind.
