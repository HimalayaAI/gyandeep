import os
import fitz  # PyMuPDF
import json
import asyncio
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import (
    UPLOAD_DIR,
    STATIC_DIR,
    ASSETS_DIR,
    TEMPLATES_DIR,
    PLUGIN_ARTIFACTS_DIR,
    TESSERACT_PATH,
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
    ERR_LLM_NOT_CONFIGURED,
    ERR_NO_PDF_UPLOADED,
    ERR_NO_CONTEXT,
    PRECOMPUTE_OCR_ON_UPLOAD,
    PRECOMPUTE_EMBEDDINGS_ON_UPLOAD,
    EMBEDDING_SOURCE_PREFIX,
    RETRIEVAL_TOP_K,
    EMBEDDING_WARMUP,
    ANIMATION_CONTEXT_MAX_CHARS,
    ANIMATION_RENDER_TIMEOUT_SECONDS,
    LLM_PROVIDER,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_BASE_URL,
    LLM_REASONING_EFFORT,
    validate_config,
)
from .logger import get_logger
from core.services.storage.embedding_service import EmbeddingService, index_embeddings
from core.services.inference import InferenceService
from core.services.plugins import ManimVideoPlugin, PluginJobRequest, PluginRuntime
from core.agents.context_manager import ContextManager
from core.agents.prompt_manager import PromptManager

# Setup PyTesseract
import pytesseract
from PIL import Image
import io

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

inference_service = InferenceService(
    provider=LLM_PROVIDER,
    api_key=LLM_API_KEY,
    model=LLM_MODEL,
    max_tokens=LLM_MAX_TOKENS,
    temperature=LLM_TEMPERATURE,
    base_url=LLM_BASE_URL,
    reasoning_effort=LLM_REASONING_EFFORT,
)
context_manager = ContextManager(
    inference_service,
    model_context_window=MODEL_CONTEXT_WINDOW,
    safety_tokens=CONTEXT_SAFETY_TOKENS,
    token_char_ratio=CONTEXT_TOKEN_CHAR_RATIO,
    summary_max_tokens=SUMMARY_MAX_TOKENS,
)
plugin_runtime = PluginRuntime(artifact_root=PLUGIN_ARTIFACTS_DIR)
plugin_runtime.register(
    ManimVideoPlugin(
        inference_service=inference_service,
        skill_root="manim-video",
        render_timeout_seconds=ANIMATION_RENDER_TIMEOUT_SECONDS,
    )
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


def _create_plugin_job(
    plugin_id: str,
    query: str,
    mode: str,
    current_page: int,
    book_id: Optional[str],
    context_text: str,
) -> Optional[str]:
    try:
        conn = _db_connect()
    except Exception as exc:
        logger.warning(f"DB connect failed for plugin job creation: {exc}")
        return None

    query_sql = """
    INSERT INTO plugin_jobs (plugin_id, status, query, mode, current_page, book_id, context_text)
    VALUES (%s, 'queued', %s, %s, %s, %s, %s)
    RETURNING id
    """
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    query_sql,
                    (plugin_id, query, mode, current_page, book_id, context_text),
                )
                row = cur.fetchone()
                return str(row[0]) if row else None
    finally:
        conn.close()


def _append_plugin_job_event(job_id: str, phase: str, message: str) -> None:
    try:
        conn = _db_connect()
    except Exception as exc:
        logger.warning(f"DB connect failed for plugin event logging: {exc}")
        return

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO plugin_job_events (job_id, phase, message)
                    VALUES (%s, %s, %s)
                    """,
                    (job_id, phase, message),
                )
    finally:
        conn.close()


def _update_plugin_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return

    allowed_fields = {
        "status",
        "plan_text",
        "script_path",
        "video_path",
        "error_text",
        "started_at",
        "finished_at",
        "metadata",
    }
    updates = {k: v for k, v in fields.items() if k in allowed_fields}
    if not updates:
        return

    try:
        conn = _db_connect()
    except Exception as exc:
        logger.warning(f"DB connect failed for plugin job update: {exc}")
        return

    set_parts: list[str] = []
    values: list[Any] = []
    for key, value in updates.items():
        if key in {"started_at", "finished_at"} and value == "now":
            set_parts.append(f"{key} = NOW()")
        else:
            set_parts.append(f"{key} = %s")
            values.append(value)
    set_parts.append("updated_at = NOW()")
    values.append(job_id)

    sql = f"UPDATE plugin_jobs SET {', '.join(set_parts)} WHERE id = %s"

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(values))
    finally:
        conn.close()


def _fetch_plugin_job(job_id: str) -> Optional[dict]:
    try:
        conn = _db_connect()
    except Exception as exc:
        logger.warning(f"DB connect failed for plugin job lookup: {exc}")
        return None

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, plugin_id, status, query, mode, current_page, book_id, context_text,
                           plan_text, script_path, video_path, error_text,
                           created_at, updated_at, started_at, finished_at
                    FROM plugin_jobs
                    WHERE id = %s
                    """,
                    (job_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "id": str(row[0]),
                    "plugin_id": row[1],
                    "status": row[2],
                    "query": row[3],
                    "mode": row[4],
                    "current_page": row[5],
                    "book_id": str(row[6]) if row[6] else None,
                    "context_text": row[7] or "",
                    "plan_text": row[8] or "",
                    "script_path": row[9],
                    "video_path": row[10],
                    "error_text": row[11],
                    "created_at": row[12].isoformat() if row[12] else None,
                    "updated_at": row[13].isoformat() if row[13] else None,
                    "started_at": row[14].isoformat() if row[14] else None,
                    "finished_at": row[15].isoformat() if row[15] else None,
                }
    finally:
        conn.close()


def _fetch_plugin_job_events(job_id: str) -> list[dict]:
    try:
        conn = _db_connect()
    except Exception as exc:
        logger.warning(f"DB connect failed for plugin events lookup: {exc}")
        return []

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT phase, message, created_at
                    FROM plugin_job_events
                    WHERE job_id = %s
                    ORDER BY created_at ASC
                    """,
                    (job_id,),
                )
                rows = cur.fetchall()
                return [
                    {
                        "phase": row[0],
                        "message": row[1],
                        "created_at": row[2].isoformat() if row[2] else None,
                    }
                    for row in rows
                ]
    finally:
        conn.close()


def _mark_incomplete_plugin_jobs_interrupted() -> None:
    try:
        conn = _db_connect()
    except Exception as exc:
        logger.warning(f"DB connect failed for plugin job recovery: {exc}")
        return

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE plugin_jobs
                    SET status = 'interrupted',
                        error_text = COALESCE(error_text, 'Server restarted before completion.'),
                        finished_at = NOW(),
                        updated_at = NOW()
                    WHERE status IN ('queued', 'planning', 'scripting', 'rendering')
                    """
                )
                interrupted_count = cur.rowcount
        if interrupted_count:
            logger.info("Recovered %s interrupted plugin jobs.", interrupted_count)
    finally:
        conn.close()


def _truncate_animation_context(text: str) -> str:
    text = text.strip()
    if len(text) <= ANIMATION_CONTEXT_MAX_CHARS:
        return text
    return text[:ANIMATION_CONTEXT_MAX_CHARS] + "\n\n[truncated]"


async def _build_animation_context(query: str, mode: str, current_page: int) -> str:
    local_raw = await build_context(current_page, window=CONTEXT_WINDOW)
    local_structured = ""
    if mode != "analyze" and inference_service.is_configured():
        try:
            local_structured, _raw = await _build_env_context(current_page)
        except Exception as exc:
            logger.warning(f"Animation structured local context failed: {exc}")

    if mode == "analyze":
        retrieved_chunks: list[str] = []
        try:
            retrieved_chunks = await _retrieve_relevant_chunks(query, RETRIEVAL_TOP_K)
        except Exception as exc:
            logger.warning(f"Animation retrieval failed: {exc}")

        sections = []
        if local_raw.strip():
            sections.append(f"=== Current Page Window (+/- {CONTEXT_WINDOW}) ===\n{local_raw.strip()}")
        if retrieved_chunks:
            chunk_text = "\n\n".join(
                [f"--- Retrieved Chunk {i+1} ---\n{chunk}" for i, chunk in enumerate(retrieved_chunks)]
            )
            sections.append(f"=== Whole Book Retrieval ===\n{chunk_text}")
        merged = "\n\n".join(sections).strip()
        return _truncate_animation_context(merged or local_raw)

    preferred = local_structured.strip() if local_structured.strip() else local_raw
    return _truncate_animation_context(preferred)


async def _run_plugin_job(job_id: str) -> None:
    job = _fetch_plugin_job(job_id)
    if not job:
        return

    async def emit(phase: str, message: str) -> None:
        _append_plugin_job_event(job_id, phase, message)
        if phase in {"planning", "scripting", "rendering"}:
            _update_plugin_job(job_id, status=phase)

    _update_plugin_job(job_id, status="planning", started_at="now")
    _append_plugin_job_event(job_id, "queued", "Animation job accepted.")

    request = PluginJobRequest(
        job_id=job_id,
        plugin_id=job["plugin_id"],
        query=job["query"],
        context_text=job["context_text"],
        mode=job["mode"],
        current_page=job["current_page"],
        book_id=job["book_id"],
        output_dir=plugin_runtime.create_job_dir(job_id),
    )

    try:
        result = await plugin_runtime.run_job(request, emit)
        _update_plugin_job(
            job_id,
            status="succeeded",
            plan_text=result.plan_text,
            script_path=result.script_path,
            video_path=result.video_path,
            finished_at="now",
        )
        _append_plugin_job_event(job_id, "done", "Animation generated successfully.")
    except Exception as exc:
        _update_plugin_job(
            job_id,
            status="failed",
            error_text=str(exc),
            finished_at="now",
        )
        _append_plugin_job_event(job_id, "failed", str(exc))


@app.on_event("startup")
async def startup_event():
    validate_config()
    _mark_incomplete_plugin_jobs_interrupted()
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


@app.post("/api/plugins/jobs")
async def create_plugin_job(request: Request):
    data = await request.json()
    plugin_id = str(data.get("plugin_id", "manim_video")).strip()
    query = str(data.get("query", "")).strip()
    mode = str(data.get("mode", "environment")).strip().lower()
    current_page = int(data.get("current_page", 1)) - 1
    book_id = data.get("book_id")

    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    if mode not in {"environment", "analyze"}:
        raise HTTPException(status_code=400, detail="mode must be environment or analyze")
    if current_page < 0:
        current_page = 0
    if not plugin_runtime.has_handler(plugin_id):
        raise HTTPException(status_code=400, detail=f"Unknown plugin: {plugin_id}")

    global global_pdf_data
    if global_pdf_data["total_pages"] == 0 and book_id:
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

    try:
        context_text = await _build_animation_context(query, mode, current_page)
    except Exception as exc:
        logger.error(f"Animation context build failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    resolved_book_id = book_id or global_pdf_data.get("book_id")
    job_id = _create_plugin_job(
        plugin_id=plugin_id,
        query=query,
        mode=mode,
        current_page=current_page + 1,
        book_id=resolved_book_id,
        context_text=context_text,
    )
    if not job_id:
        raise HTTPException(status_code=500, detail="Failed to persist plugin job")

    asyncio.create_task(_run_plugin_job(job_id))
    return JSONResponse(
        content={
            "status": "queued",
            "job_id": job_id,
            "plugin_id": plugin_id,
        }
    )


@app.get("/api/plugins/jobs/{job_id}")
async def get_plugin_job(job_id: str):
    job = _fetch_plugin_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Plugin job not found")
    events = _fetch_plugin_job_events(job_id)
    artifacts = {}
    if job.get("script_path"):
        artifacts["script"] = f"/api/plugins/jobs/{job_id}/artifacts/script"
    if job.get("video_path"):
        artifacts["video"] = f"/api/plugins/jobs/{job_id}/artifacts/video"

    return JSONResponse(content={"job": job, "events": events, "artifacts": artifacts})


@app.get("/api/plugins/jobs/{job_id}/artifacts/{artifact_type}")
async def get_plugin_job_artifact(job_id: str, artifact_type: str):
    job = _fetch_plugin_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Plugin job not found")

    if artifact_type == "script":
        path_value = job.get("script_path")
        media_type = "text/x-python"
    elif artifact_type == "video":
        path_value = job.get("video_path")
        media_type = "video/mp4"
    else:
        raise HTTPException(status_code=404, detail="Unknown artifact type")

    if not path_value:
        raise HTTPException(status_code=404, detail="Artifact not ready")

    path = Path(path_value).resolve()
    artifact_root = Path(PLUGIN_ARTIFACTS_DIR).resolve()
    if artifact_root not in path.parents:
        raise HTTPException(status_code=403, detail="Invalid artifact path")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing")

    return FileResponse(path=str(path), media_type=media_type, filename=path.name)


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


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


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
        raise HTTPException(status_code=503, detail=ERR_LLM_NOT_CONFIGURED)
        
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
        raise HTTPException(status_code=503, detail=ERR_LLM_NOT_CONFIGURED)

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
            yield f"data: {json.dumps({'error': ERR_LLM_NOT_CONFIGURED})}\n\n"
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
