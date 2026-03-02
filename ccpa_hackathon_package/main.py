"""
main.py — FastAPI application for CCPA Compliance Analysis.

Endpoints:
  GET  /health  → {"status": "ok"}
  POST /analyze → {"harmful": bool, "articles": [...]}

Startup:
  1. Parse ccpa_statute.pdf
  2. Build sentence-transformer embeddings
  3. Load FAISS index
  All done ONCE at startup, reused for every request.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from embeddings import statute_index
from hf_inference import build_prompt, call_hf_inference
from pdf_parser import parse_and_chunk
from sanitizer import SAFE_DEFAULT, sanitize_response

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PDF_PATH = os.environ.get("CCPA_PDF_PATH", "ccpa_statute.pdf")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    prompt: str = ""


class AnalyzeResponse(BaseModel):
    harmful: bool
    articles: list[str]


# ---------------------------------------------------------------------------
# Startup lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the FAISS index at startup."""
    logger.info("=" * 60)
    logger.info("CCPA Compliance API — Starting up")
    logger.info("=" * 60)

    # Step 1: Parse PDF
    logger.info(f"Parsing PDF: {PDF_PATH}")
    try:
        chunks = parse_and_chunk(PDF_PATH)
        logger.info(f"Parsed {len(chunks)} chunks from statute PDF")
    except FileNotFoundError:
        logger.error(f"CRITICAL: PDF not found at {PDF_PATH}")
        logger.error("The server will start but /analyze will return defaults.")
        chunks = []

    # Step 2: Build embeddings and FAISS index
    if chunks:
        logger.info("Building FAISS index...")
        statute_index.build(chunks)
        logger.info("FAISS index ready!")
    else:
        logger.warning("No chunks to index. Retrieval will be unavailable.")

    logger.info("=" * 60)
    logger.info("CCPA Compliance API — Ready to serve requests")
    logger.info("=" * 60)

    yield  # Server is running

    logger.info("CCPA Compliance API — Shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CCPA Compliance API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse(content={"status": "ok"}, status_code=200)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Analyze a business practice for CCPA compliance violations.

    Returns strict JSON: {"harmful": bool, "articles": [...]}
    """
    prompt = request.prompt.strip()

    # Edge case: empty prompt
    if not prompt:
        logger.info("Empty prompt received, returning safe default")
        return JSONResponse(content=SAFE_DEFAULT.copy())

    # Edge case: FAISS index not available
    if not statute_index.is_ready:
        logger.warning("FAISS index not ready, returning safe default")
        return JSONResponse(content=SAFE_DEFAULT.copy())

    try:
        # Step 1: Retrieve relevant statute chunks
        logger.info(f"Analyzing prompt: {prompt[:80]}...")
        relevant_chunks = statute_index.retrieve(prompt, top_k=5)

        if not relevant_chunks:
            logger.warning("No relevant chunks retrieved")
            return JSONResponse(content=SAFE_DEFAULT.copy())

        # Step 2: Build LLM prompt with context
        full_prompt = build_prompt(prompt, relevant_chunks)

        # Step 3: Call HF Inference API
        raw_output = call_hf_inference(full_prompt)

        if not raw_output:
            logger.warning("Empty response from HF API")
            return JSONResponse(content=SAFE_DEFAULT.copy())

        # Step 4: Sanitize and validate response
        result = sanitize_response(raw_output)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        return JSONResponse(content=SAFE_DEFAULT.copy())


# ---------------------------------------------------------------------------
# Direct run support (for development)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
