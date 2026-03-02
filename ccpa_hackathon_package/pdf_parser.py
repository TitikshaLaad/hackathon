"""
pdf_parser.py — Extract and chunk text from the CCPA statute PDF.

Uses PyMuPDF (fitz) for fast, reliable PDF text extraction.
Chunks are sized to 500–1000 tokens (approximated by whitespace splitting)
with overlap to preserve context across chunk boundaries.
"""

import logging
import os

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Chunking parameters
TARGET_CHUNK_SIZE = 700      # target tokens per chunk
MIN_CHUNK_SIZE = 400         # minimum tokens (discard smaller trailing chunks)
OVERLAP_SIZE = 100           # token overlap between consecutive chunks


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    full_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            full_text.append(text)

    doc.close()
    combined = "\n".join(full_text)
    logger.info(f"Extracted {len(combined)} characters from {pdf_path}")
    return combined


def chunk_text(text: str) -> list[str]:
    """
    Split text into chunks of TARGET_CHUNK_SIZE tokens with OVERLAP_SIZE overlap.
    'Tokens' are approximated by whitespace-separated words.
    """
    words = text.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + TARGET_CHUNK_SIZE
        chunk_words = words[start:end]

        if len(chunk_words) < MIN_CHUNK_SIZE and chunks:
            # Append small trailing fragment to the last chunk
            chunks[-1] += " " + " ".join(chunk_words)
            break

        chunks.append(" ".join(chunk_words))
        start = end - OVERLAP_SIZE  # overlap for context continuity

    logger.info(f"Created {len(chunks)} chunks from {len(words)} words")
    return chunks


def parse_and_chunk(pdf_path: str) -> list[str]:
    """Main entry point: extract text from PDF and return chunks."""
    text = extract_text_from_pdf(pdf_path)
    return chunk_text(text)
