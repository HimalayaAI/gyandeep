import os
import fitz  # PyMuPDF
import json
import asyncio
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import (
    SARVAMAI_KEY,
    API_KEY_PLACEHOLDER,
    UPLOAD_DIR,
    STATIC_DIR,
    TEMPLATES_DIR,
    TESSERACT_PATH,
    SARVAM_MODEL,
    SARVAM_MAX_TOKENS,
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
    validate_config,
)
from logger import get_logger

# Setup PyTesseract
import pytesseract
from PIL import Image
import io

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Setup Sarvam AI
try:
    from sarvamai import SarvamAI
    if SARVAMAI_KEY and SARVAMAI_KEY != API_KEY_PLACEHOLDER:
        sarvam_client = SarvamAI(api_subscription_key=SARVAMAI_KEY)
    else:
        sarvam_client = None
except ImportError:
    sarvam_client = None

app = FastAPI()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

os.makedirs(UPLOAD_DIR, exist_ok=True)

global_pdf_data = {
    "filename": None,
    "filepath": None,
    "toc": [],
    "pages": {},
    "total_pages": 0,
    "analysis": DEFAULT_ANALYSIS_MESSAGE
}

logger = get_logger(__name__)


def _extract_response_content(response) -> str:
    """Pull usable text from a Sarvam API response.

    Reasoning models sometimes spend all completion tokens on
    ``reasoning_content`` and leave ``content`` empty.  This helper
    checks ``content`` first, then falls back to ``reasoning_content``.
    """
    msg = response.choices[0].message
    if msg.content and msg.content.strip():
        return msg.content.strip()
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning and reasoning.strip():
        logger.warning("API returned empty content; falling back to reasoning_content")
        return reasoning.strip()
    return ""


@app.on_event("startup")
async def startup_event():
    validate_config()


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global global_pdf_data
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    try:
        doc = fitz.open(file_path)
        toc = doc.get_toc() # [level, title, page_number]
        total_pages = len(doc)
        
        global_pdf_data = {
            "filename": file.filename,
            "filepath": file_path,
            "toc": toc,
            "pages": {},
            "total_pages": total_pages,
            "analysis": DEFAULT_ANALYSIS_MESSAGE
        }
        
        doc.close()
        
        return {
            "status": "success",
            "filename": file.filename,
            "total_pages": total_pages,
            "toc_entries": len(toc)
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

@app.post("/api/analyze_env")
async def analyze_env(request: Request):
    """
    Runs an analysis over the local environment (+/- 5 pages),
    and asks Sarvam to create a high-level summary of this section.
    """
    if not sarvam_client:
        raise HTTPException(status_code=500, detail=ERR_SARVAM_NOT_CONFIGURED)
        
    data = await request.json()
    current_page = data.get("current_page", 1) - 1 # 0-indexed
    
    global global_pdf_data
    total = global_pdf_data["total_pages"]
    if total == 0:
        raise HTTPException(status_code=400, detail=ERR_NO_PDF_UPLOADED)
        
    extracted_text = await build_context(current_page, window=CONTEXT_WINDOW)
    
    prompt = f"""You are a document analyzer. Convert this raw OCR text into highly structured, clean synthetic data context.
Provide a clear, high-level summary of what this specific section is about and outline its main topics.

Raw OCR Document Text:
{extracted_text}
"""
    
    try:
        response = await asyncio.to_thread(
            sarvam_client.chat.completions,
            model=SARVAM_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=SARVAM_MAX_TOKENS,
        )
        extracted = _extract_response_content(response)
        analysis_result = extracted if extracted else API_EMPTY_RESPONSE_MESSAGE
        
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
    if not sarvam_client:
        raise HTTPException(status_code=500, detail=ERR_SARVAM_NOT_CONFIGURED)

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
                
            prompt = f"""Convert all of the following messy OCR text into clean, structured synthetic data context. 
Organize the overarching concepts strictly and clearly so it can be used for RAG applications.

Raw Document Text (Pages {i+1} to {chunk_end}):
{extracted_text}
"""
            
            response = await asyncio.to_thread(
                sarvam_client.chat.completions,
                model=SARVAM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=SARVAM_MAX_TOKENS,
            )
            extracted = _extract_response_content(response)
            analysis_result = extracted if extracted else API_EMPTY_RESPONSE_MESSAGE
            
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
    
    global global_pdf_data
    if global_pdf_data["total_pages"] == 0:
        raise HTTPException(status_code=400, detail=ERR_NO_PDF_UPLOADED)

    if not sarvam_client:
        async def stream_error():
            yield f"data: {json.dumps({'error': ERR_SARVAM_NOT_CONFIGURED})}\n\n"
            yield "event: end\ndata: {}\n\n"
        return StreamingResponse(stream_error(), media_type=SSE_MEDIA_TYPE)

    target_file = GLOBAL_CONTEXT_FILE if mode == "analyze" else ENV_CONTEXT_FILE
    file_context = ERR_NO_CONTEXT
    
    if os.path.exists(target_file):
        with open(target_file, "r", encoding="utf-8") as f:
            file_context = f.read()
    
    system_prompt = f"""You are an advanced, helpful document assistant responding to queries.
You must strictly rely on the generated Synthetic Data context provided from {target_file}.
If the context does not contain the answer, explicitly state that you cannot find it in the current mode's context.

Extracted Synthetic Context ({target_file}):
{file_context}
"""

    user_prompt = f"Question: {query}"

    async def single_chunk_response():
        try:
            response = await asyncio.to_thread(
                sarvam_client.chat.completions,
                model=SARVAM_MODEL,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                max_tokens=SARVAM_MAX_TOKENS,
            )
            content = _extract_response_content(response)
            yield f"data: {json.dumps({'content': content})}\n\n"
            yield "event: end\ndata: {}\n\n"
        except Exception as e:
            logger.error(f"Chat error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "event: end\ndata: {}\n\n"

    return StreamingResponse(single_chunk_response(), media_type=SSE_MEDIA_TYPE)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=SERVER_HOST, port=SERVER_PORT, reload=True)
