# CCPA Compliance API

A production-grade RAG-based API for analyzing business practices against the California Consumer Privacy Act (CCPA). Uses sentence-transformers for embedding, FAISS for vector search, and Mistral-7B-Instruct via the Hugging Face Inference API for analysis.

## Architecture

```
ccpa_statute.pdf
    ↓ (PyMuPDF extraction + chunking)
FAISS Vector Index (built at startup)
    ↓ (top-5 retrieval per query)
Mistral-7B-Instruct (HF Inference API)
    ↓ (JSON parsing + sanitization)
Strict JSON Response
```

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Hugging Face API token ([get one here](https://huggingface.co/settings/tokens))

### 1. Set your HF token

**Linux / macOS:**
```bash
export HF_TOKEN=hf_your_token_here
```

**Windows (PowerShell):**
```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

Or create a `.env` file in the project directory:
```
HF_TOKEN=hf_your_token_here
```

### 2. Build and Run with Docker Compose

```bash
docker compose up -d --build
```

The server will:
1. Parse `ccpa_statute.pdf`
2. Build embeddings and FAISS index
3. Start serving on port 8000

Startup takes ~60-90 seconds (embedding generation).

### 3. Verify it's running

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok"}
```

## API Endpoints

### `GET /health`

Returns `200 OK` with `{"status": "ok"}` when the server is ready.

### `POST /analyze`

Analyze a business practice for CCPA violations.

**Request:**
```json
{"prompt": "We are selling our customers' personal information to third-party data brokers without informing them."}
```

**Response (violation detected):**
```json
{"harmful": true, "articles": ["Section 1798.120"]}
```

**Response (no violation):**
```json
{"harmful": false, "articles": []}
```

## Curl Examples

### Harmful prompt
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "We sell customer data to brokers without telling them or giving opt-out options."}'
```

### Safe prompt
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Our company provides a clear privacy policy and allows customers to opt out at any time."}'
```

### Empty prompt (edge case)
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": ""}'
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes | — | Hugging Face API token |
| `CCPA_PDF_PATH` | No | `ccpa_statute.pdf` | Path to the CCPA statute PDF |

## Local Development (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Set token
export HF_TOKEN=hf_your_token_here

# Run
python main.py
```

Server starts at `http://localhost:8000`.

## Project Structure

```
├── main.py              # FastAPI application
├── pdf_parser.py        # PDF text extraction and chunking
├── embeddings.py        # Sentence-transformer embeddings + FAISS
├── hf_inference.py      # HF Inference API client
├── sanitizer.py         # JSON validation and sanitization
├── ccpa_sections.py     # Valid CCPA section whitelist
├── ccpa_statute.pdf     # CCPA statute document
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker image definition
├── docker-compose.yml   # Docker Compose configuration
└── validate_format.py   # Organizer test script
```

## Stopping

```bash
docker compose down
```
