import os
import fitz  # PyMuPDF
import json
import asyncio
import hashlib
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import (
    SARVAMAI_KEY,
    API_KEY_PLACEHOLDER,
    UPLOAD_DIR,
    STATIC_DIR,
    ASSETS_DIR,
    TEMPLATES_DIR,
    TESSERACT_PATH,
    SARVAM_MODEL,
    SARVAM_MAX_TOKENS,
    SARVAM_REASONING_EFFORT,
    SARVAM_TEMPERATURE,
    MODEL_CONTEXT_WINDOW,
    CONTEXT_SAFETY_TOKENS,
    CONTEXT_TOKEN_CHAR_RATIO,
    SUMMARY_MAX_TOKENS,
    OCR_MIN_TEXT_LENGTH,
    OCR_DPI,
    OCR_FALLBACK_MESSAGE,
    CONTEXT_WINDOW,
    GLOBAL_CONTEXT_FILE,
    ENV_CONTEXT_FILE,
    DEFAULT_ANALYSIS_MESSAGE,
    API_EMPTY_RESPONSE_MESSAGE,
    OCR_SEMAPHORE_LIMIT,
    ANALYSIS_CHUNK_SIZE,
    SERVER_HOST,
    SERVER_PORT,
    SSE_MEDIA_TYPE,
    ERR_SARVAM_NOT_CONFIGURED,
    ERR_NO_PDF_UPLOADED,
    ERR_NO_CONTEXT,
    PRECOMPUTE_OCR_ON_UPLOAD,
    PRECOMPUTE_EMBEDDINGS_ON_UPLOAD,
    EMBEDDING_SOURCE_PREFIX,
    RETRIEVAL_TOP_K,
    EMBEDDING_WARMUP,
    validate_config,
)
from .logger import get_logger
from core.services.storage.embedding_service import EmbeddingService, index_embeddings
from core.services.ingestion.embedding_pipeline import chunk_text
from core.services.inference import InferenceService
from core.agents.context_manager import ContextManager
from core.agents.prompt_manager import PromptManager

# Setup PyTesseract
import pytesseract
from PIL import Image
import io

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

inference_service = InferenceService(
    api_key=SARVAMAI_KEY,
    api_key_placeholder=API_KEY_PLACEHOLDER,
    model=SARVAM_MODEL,
    max_tokens=SARVAM_MAX_TOKENS,
    temperature=SARVAM_TEMPERATURE,
    reasoning_effort=SARVAM_REASONING_EFFORT,
)
context_manager = ContextManager(
    inference_service,
    model_context_window=MODEL_CONTEXT_WINDOW,
    safety_tokens=CONTEXT_SAFETY_TOKENS,
    token_char_ratio=CONTEXT_TOKEN_CHAR_RATIO,
    summary_max_tokens=SUMMARY_MAX_TOKENS,
)

app = FastAPI()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

os.makedirs(UPLOAD_DIR, exist_ok=True)

global_pdf_data = {
    "filename": None,
    "filepath": None,
    "toc": [],
    "pages": {},
    "total_pages": 0,
    "analysis": DEFAULT_ANALYSIS_MESSAGE,
    "book_id": None,
    "precompute": {
        "running": False,
        "current_page": 0,
        "total_pages": 0,
        "embeddings_done": False,
    },
}

logger = get_logger(__name__)


def _extract_response_payload(response) -> tuple[str, str]:
    """Extract answer content and any reasoning content without exposing it."""
    return inference_service.extract_response_payload(response)


def _stream_text_chunks(text: str, size: int = 80):
    for i in range(0, len(text), size):
        yield text[i : i + size]


def _sarvam_params(messages: list[dict]) -> dict:
    return inference_service.build_params(messages)


def _db_connect():
    try:
        import psycopg2
    except ImportError as exc:
        raise ImportError("psycopg2-binary is required for database persistence.") from exc

    db_host = os.getenv("DB_HOST", "localhost")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD")
    if not db_password:
        db_password = "postgres"
    db_name = os.getenv("DB_NAME", "gyandeep")

    return psycopg2.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        dbname=db_name,
    )


def _upsert_book(filename: str, file_hash: str, total_pages: int) -> Optional[str]:
    try:
        conn = _db_connect()
    except Exception as exc:
        logger.warning(f"DB connect failed for book persistence: {exc}")
        return None

    query = """
    INSERT INTO books (filename, file_hash, total_pages)
    VALUES (%s, %s, %s)
    ON CONFLICT (file_hash) DO UPDATE
    SET filename = EXCLUDED.filename,
        total_pages = EXCLUDED.total_pages,
        updated_at = NOW()
    RETURNING id
    """

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(query, (filename, file_hash, total_pages))
                row = cur.fetchone()
                return str(row[0]) if row else None
    finally:
        conn.close()


def _load_ocr_page(book_id: str, page_index: int) -> Optional[str]:
    try:
        conn = _db_connect()
    except Exception:
        return None

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT content FROM ocr_pages
                    WHERE book_id = %s AND page_index = %s
                    """,
                    (book_id, page_index),
                )
                row = cur.fetchone()
                return row[0] if row else None
    finally:
        conn.close()


def _save_ocr_page(book_id: str, page_index: int, content: str) -> None:
    try:
        conn = _db_connect()
    except Exception as exc:
        logger.warning(f"DB connect failed for OCR persistence: {exc}")
        return

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ocr_pages (book_id, page_index, content)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (book_id, page_index)
                    DO UPDATE SET content = EXCLUDED.content
                    """,
                    (book_id, page_index, content),
                )
    finally:
        conn.close()


def _list_books() -> list[dict]:
    try:
        conn = _db_connect()
    except Exception as exc:
        logger.warning(f"DB connect failed for listing books: {exc}")
        return []

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, filename, total_pages, updated_at
                    FROM books
                    ORDER BY updated_at DESC
                    """
                )
                rows = cur.fetchall()
                return [
                    {
                        "id": str(row[0]),
                        "filename": row[1],
                        "total_pages": row[2],
                        "updated_at": row[3].isoformat() if row[3] else None,
                    }
                    for row in rows
                ]
    finally:
        conn.close()


def _get_book_by_id(book_id: str) -> Optional[dict]:
    try:
        conn = _db_connect()
    except Exception as exc:
        logger.warning(f"DB connect failed for fetching book: {exc}")
        return None

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, filename, total_pages
                    FROM books
                    WHERE id = %s
                    """,
                    (book_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return {"id": str(row[0]), "filename": row[1], "total_pages": row[2]}
    finally:
        conn.close()


@app.on_event("startup")
async def startup_event():
    validate_config()
    if EMBEDDING_WARMUP:
        async def _warmup():
            try:
                embedder = EmbeddingService()
                await embedder.get_embeddings(["warmup"])
                logger.info("Embedding model warmup complete.")
            except Exception as exc:
                logger.warning(f"Embedding warmup failed: {exc}")
        asyncio.create_task(_warmup())


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return HTMLResponse(status_code=204)


@app.get("/api/books")
async def list_books():
    return JSONResponse(content={"books": _list_books()})


@app.post("/api/books/select")
async def select_book(request: Request):
    data = await request.json()
    book_id = data.get("book_id")
    if not book_id:
        raise HTTPException(status_code=400, detail="book_id required")

    book = _get_book_by_id(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    filepath = os.path.join(UPLOAD_DIR, book["filename"])
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Book file not found on disk")

    global global_pdf_data
    global_pdf_data["filename"] = book["filename"]
    global_pdf_data["filepath"] = filepath
    global_pdf_data["total_pages"] = book["total_pages"]
    global_pdf_data["book_id"] = book_id
    global_pdf_data["pages"] = {}

    return JSONResponse(content={"status": "ok", "book": book})

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global global_pdf_data
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    
    file_bytes = await file.read()
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    file_hash = hashlib.sha256(file_bytes).hexdigest()
        
    try:
        doc = fitz.open(file_path)
        toc = doc.get_toc() # [level, title, page_number]
        total_pages = len(doc)

        book_id = _upsert_book(file.filename, file_hash, total_pages)
        
        global_pdf_data = {
            "filename": file.filename,
            "filepath": file_path,
            "toc": toc,
            "pages": {},
            "total_pages": total_pages,
            "analysis": DEFAULT_ANALYSIS_MESSAGE,
            "book_id": book_id,
            "precompute": {
                "running": False,
                "current_page": 0,
                "total_pages": total_pages,
                "embeddings_done": False,
            },
        }
        
        doc.close()

        if PRECOMPUTE_OCR_ON_UPLOAD:
            logger.info("Starting OCR precompute for %s (%s pages)", file.filename, total_pages)
            asyncio.create_task(_precompute_ocr_and_embeddings())
        
        return {
            "status": "success",
            "filename": file.filename,
            "total_pages": total_pages,
            "toc_entries": len(toc),
            "book_id": book_id,
        }
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Expose uploads folder for the frontend PDF reader
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

def extract_page_sync(page_index: int) -> str:
    """Synchronous function to perform extraction and Tesseract OCR on a single page"""
    global global_pdf_data
    
    if page_index in global_pdf_data["pages"] and global_pdf_data["pages"][page_index]:
        return global_pdf_data["pages"][page_index]

    if global_pdf_data.get("book_id"):
        cached = _load_ocr_page(global_pdf_data["book_id"], page_index)
        if cached:
            global_pdf_data["pages"][page_index] = cached
            return cached
        
    if not global_pdf_data["filepath"]:
        return ""
        
    doc = fitz.open(global_pdf_data["filepath"])
    
    if page_index < 0 or page_index >= len(doc):
        doc.close()
        return ""
        
    page = doc[page_index]
    text = page.get_text("text").strip()
    
    if len(text) < OCR_MIN_TEXT_LENGTH:
        logger.info(f"Page {page_index+1} lacks native text. Running Tesseract OCR...")
        try:
            pix = page.get_pixmap(dpi=OCR_DPI)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(img)
        except Exception as e:
            logger.error(f"Tesseract Error: {e}")
            text = OCR_FALLBACK_MESSAGE
            
    global_pdf_data["pages"][page_index] = text
    doc.close()
    if global_pdf_data.get("book_id"):
        _save_ocr_page(global_pdf_data["book_id"], page_index, text)
    return text

async def extract_page_async(page_index: int) -> str:
    """Run the synchronous extraction in a thread pool to avoid blocking FastAPI"""
    return await asyncio.to_thread(extract_page_sync, page_index)

async def build_context(center_page: int, window: int = CONTEXT_WINDOW) -> str:
    global global_pdf_data
    total = global_pdf_data["total_pages"]
    if total == 0:
        return ""
    
    start_page = max(0, center_page - window)
    end_page = min(total - 1, center_page + window)
    
    tasks = [extract_page_async(p) for p in range(start_page, end_page + 1)]
    results = await asyncio.gather(*tasks)
    
    context_parts = []
    for i, text in enumerate(results):
        p = start_page + i
        if text.strip():
            context_parts.append(f"--- Page {p + 1} ---\n{text.strip()}")
            
    return "\n\n".join(context_parts)


async def _generate_structured_context(raw_text: str, label: str) -> str:
    """Convert raw OCR text into structured synthetic context."""
    if not raw_text.strip():
        return ""
    extracted = await context_manager.build_structured_context(raw_text)
    return extracted if extracted else API_EMPTY_RESPONSE_MESSAGE


async def _build_env_context(current_page: int) -> tuple[str, str]:
    """Build raw +/- window context and paraphrased structured context."""
    extracted_text = await build_context(current_page, window=CONTEXT_WINDOW)
    if not extracted_text.strip():
        return "", ""
    structured = await _generate_structured_context(extracted_text, label="current_page_window")
    return structured, extracted_text



async def _precompute_ocr_and_embeddings():
    global global_pdf_data
    total = global_pdf_data["total_pages"]
    if total == 0:
        return

    global_pdf_data["precompute"]["running"] = True
    global_pdf_data["precompute"]["total_pages"] = total

    logger.info("OCR precompute running for %s pages", total)
    sem = asyncio.Semaphore(OCR_SEMAPHORE_LIMIT)

    async def extract_with_semaphore(p):
        async with sem:
            return await extract_page_async(p)

    batch_size = max(1, OCR_SEMAPHORE_LIMIT)
    for start in range(0, total, batch_size):
        tasks = [extract_with_semaphore(p) for p in range(start, min(start + batch_size, total))]
        await asyncio.gather(*tasks)
        global_pdf_data["precompute"]["current_page"] = min(start + batch_size, total)
        if global_pdf_data["precompute"]["current_page"] % 10 == 0 or global_pdf_data["precompute"]["current_page"] == total:
            logger.info(
                "OCR precompute progress: %s/%s pages",
                global_pdf_data["precompute"]["current_page"],
                total,
            )
<<<<<<< Updated upstream
=======
            chunks = chunk_text(all_text)
            embedder = EmbeddingService()
            try:
                logger.info("Embedding precompute started (%s chunks)", len(chunks))
                embeddings = await embedder.get_embeddings(chunks)
                source_value = global_pdf_data.get("book_id") or global_pdf_data["filename"]
                source = f"{EMBEDDING_SOURCE_PREFIX}:{source_value}"
                index_embeddings(chunks, embeddings, source=source)
                global_pdf_data["precompute"]["embeddings_done"] = True
                logger.info("Embedding precompute finished")
            except Exception as exc:
                logger.warning("Embedding precompute failed: %s", exc)
>>>>>>> Stashed changes

    if PRECOMPUTE_EMBEDDINGS_ON_UPLOAD:
        all_text = "\n\n".join(
            [global_pdf_data["pages"].get(i, "") for i in range(total)]
        )
        chunks = _chunk_text(all_text)
        embedder = EmbeddingService()
        try:
            logger.info("Embedding precompute started (%s chunks)", len(chunks))
            embeddings = await embedder.get_embeddings(chunks)
            source_value = global_pdf_data.get("book_id") or global_pdf_data["filename"]
            source = f"{EMBEDDING_SOURCE_PREFIX}:{source_value}"
            index_embeddings(chunks, embeddings, source=source)
            global_pdf_data["precompute"]["embeddings_done"] = True
            logger.info("Embedding precompute finished")
        except Exception as exc:
            logger.warning(f"Embedding precompute failed: {exc}")

    global_pdf_data["precompute"]["running"] = False
    logger.info("OCR precompute complete")


async def _retrieve_relevant_chunks(query: str, top_k: int) -> list[str]:
    embedder = EmbeddingService()
    source = None
    if global_pdf_data.get("book_id") or global_pdf_data.get("filename"):
        source_value = global_pdf_data.get("book_id") or global_pdf_data.get("filename")
        source = f"{EMBEDDING_SOURCE_PREFIX}:{source_value}"
    return await embedder.get_relevant_chunks(query, top_k=top_k, source=source)

@app.post("/api/analyze_env")
async def analyze_env(request: Request):
    """
    Runs an analysis over the local environment (+/- 5 pages),
    and asks Sarvam to create a high-level summary of this section.
    """
    if not inference_service.is_configured():
        raise HTTPException(status_code=503, detail=ERR_SARVAM_NOT_CONFIGURED)
        
    data = await request.json()
    current_page = data.get("current_page", 1) - 1 # 0-indexed
    
    global global_pdf_data
    total = global_pdf_data["total_pages"]
    if total == 0:
        raise HTTPException(status_code=400, detail=ERR_NO_PDF_UPLOADED)
        
    try:
        analysis_result, raw_text = await _build_env_context(current_page)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="No text found in this context window.")

        with open(ENV_CONTEXT_FILE, "w", encoding="utf-8") as f:
            f.write(analysis_result)
            
        return JSONResponse(content={"analysis": analysis_result + f" (Saved to {ENV_CONTEXT_FILE})"})
    except Exception as e:
        logger.error(f"Analysis Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze_global")
async def analyze_global():
    """
    Runs an analysis over the ENTIRE PDF using concurrent OCR,
    and then asks Sarvam AI for synthetic data conversion into context.txt.
    """
    if not inference_service.is_configured():
        raise HTTPException(status_code=503, detail=ERR_SARVAM_NOT_CONFIGURED)

    global global_pdf_data
    total = global_pdf_data["total_pages"]
    if total == 0:
        raise HTTPException(status_code=400, detail=ERR_NO_PDF_UPLOADED)
        
    sem = asyncio.Semaphore(OCR_SEMAPHORE_LIMIT)
    async def extract_with_semaphore(p):
        async with sem:
            return await extract_page_async(p)
            
    with open(GLOBAL_CONTEXT_FILE, "w", encoding="utf-8") as f:
        f.write("")
        
    overall_analysis = ""
    
    try:
        for i in range(0, total, ANALYSIS_CHUNK_SIZE):
            chunk_end = min(i + ANALYSIS_CHUNK_SIZE, total)
            logger.info(f"Processing global analysis for pages {i+1} to {chunk_end}...")
            
            tasks = [extract_with_semaphore(p) for p in range(i, chunk_end)]
            results = await asyncio.gather(*tasks)
            
            extracted_text = "\n\n".join([r for r in results if r])
            if not extracted_text.strip():
                continue
                
            analysis_result = await context_manager.build_global_chunk_summary(
                extracted_text,
                page_start=i + 1,
                page_end=chunk_end,
            )
            if not analysis_result:
                analysis_result = API_EMPTY_RESPONSE_MESSAGE
            
            with open(GLOBAL_CONTEXT_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n\n--- Analysis for Pages {i+1} to {chunk_end} ---\n\n")
                f.write(analysis_result)
                
            overall_analysis += f"\n\n--- Analysis for Pages {i+1} to {chunk_end} ---\n\n" + analysis_result
            
        global_pdf_data["analysis"] = overall_analysis
        return JSONResponse(content={
            "analysis": f"Synthetic data successfully generated in chunks and appended to {GLOBAL_CONTEXT_FILE}!"
        })
    except Exception as e:
        logger.error(f"Analysis Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask")
async def ask_question(request: Request):
    data = await request.json()
    query = data.get("query")
    mode = data.get("mode", "analyze") # default to global analyze mode
    current_page = data.get("current_page", 1) - 1
    book_id = data.get("book_id")
    
    global global_pdf_data
    if global_pdf_data["total_pages"] == 0:
        if book_id:
            book = _get_book_by_id(book_id)
            if book:
                filepath = os.path.join(UPLOAD_DIR, book["filename"])
                if os.path.exists(filepath):
                    global_pdf_data["filename"] = book["filename"]
                    global_pdf_data["filepath"] = filepath
                    global_pdf_data["total_pages"] = book["total_pages"]
                    global_pdf_data["book_id"] = book_id
                    global_pdf_data["pages"] = {}
        if global_pdf_data["total_pages"] == 0:
            raise HTTPException(status_code=400, detail=ERR_NO_PDF_UPLOADED)

    if not inference_service.is_configured():
        async def stream_error():
            yield f"data: {json.dumps({'error': ERR_SARVAM_NOT_CONFIGURED})}\n\n"
            yield "event: end\ndata: {}\n\n"
        return StreamingResponse(stream_error(), media_type=SSE_MEDIA_TYPE)

    file_context = ERR_NO_CONTEXT
    retrieved_chunks = []

    if mode == "analyze":
        try:
            retrieved_chunks = await _retrieve_relevant_chunks(query, RETRIEVAL_TOP_K)
        except Exception as e:
            logger.error(f"Retrieval error: {e}")

        if not retrieved_chunks:
            target_file = GLOBAL_CONTEXT_FILE
            if os.path.exists(target_file):
                with open(target_file, "r", encoding="utf-8") as f:
                    file_context = f.read()
            else:
                file_context = (
                    "No embeddings found for whole-book retrieval. "
                    "Enable PRECOMPUTE_EMBEDDINGS_ON_UPLOAD=true and re-upload the book."
                )
        else:
            file_context = "\n\n".join(
                [f"--- Retrieved Chunk {i+1} ---\n{chunk}" for i, chunk in enumerate(retrieved_chunks)]
            )
    else:
        target_file = "current-page-window"
        structured_context, raw_context = await _build_env_context(current_page)
        if structured_context.strip():
            file_context = structured_context
            with open(ENV_CONTEXT_FILE, "w", encoding="utf-8") as f:
                f.write(structured_context)
        else:
            file_context = raw_context.strip() if raw_context.strip() else "No text found for the current page window."
    
    if mode == "analyze":
        system_prompt = PromptManager.whole_book_prompt(file_context)
    else:
        system_prompt = PromptManager.current_page_prompt(file_context)

    user_prompt = f"Question: {query}"

    async def single_chunk_response():
        try:
            response = await asyncio.to_thread(
                inference_service.chat_completions,
                [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
            )
            content, reasoning = _extract_response_payload(response)

            if reasoning:
                think_text = reasoning.strip()
                if len(think_text) > 1200:
                    think_text = think_text[:1200] + "..."
                for chunk in _stream_text_chunks(think_text, size=120):
                    yield f"data: {json.dumps({'type': 'thinking', 'content': chunk, 'append': True})}\n\n"
                    await asyncio.sleep(0.01)

            if not content:
                yield f"data: {json.dumps({'content': API_EMPTY_RESPONSE_MESSAGE})}\n\n"
                yield "event: end\ndata: {}\n\n"
                return

            for chunk in _stream_text_chunks(content, size=80):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
                await asyncio.sleep(0.02)
            yield "event: end\ndata: {}\n\n"
        except Exception as e:
            logger.error(f"Chat error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "event: end\ndata: {}\n\n"

    return StreamingResponse(single_chunk_response(), media_type=SSE_MEDIA_TYPE)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.backend.app:app", host=SERVER_HOST, port=SERVER_PORT, reload=True)
