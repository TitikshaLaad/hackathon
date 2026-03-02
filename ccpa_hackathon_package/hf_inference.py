"""
hf_inference.py — Hugging Face Inference API client.

Calls a hosted LLM (Mistral-7B-Instruct) via the HF Inference API.
No local model weights are downloaded.
"""

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

# Configuration
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def get_hf_token() -> str:
    """Read HF_TOKEN from environment variable."""
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        logger.warning("HF_TOKEN not set! API calls will fail.")
    return token


def build_prompt(user_prompt: str, context_chunks: list[str]) -> str:
    """
    Construct the full prompt for the LLM with retrieved context.
    Uses a strict system instruction to enforce JSON-only output.
    """
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""<s>[INST]
You are a CCPA (California Consumer Privacy Act) compliance analyzer. Your ONLY job is to determine if a given business practice violates the CCPA.

RULES:
1. Return ONLY raw JSON. No explanation, no markdown, no extra text.
2. The JSON must have EXACTLY two keys: "harmful" (boolean) and "articles" (list of strings).
3. If the practice violates the CCPA, set "harmful" to true and list the violated sections.
4. If the practice does NOT violate the CCPA, set "harmful" to false and "articles" to [].
5. Only cite sections that are explicitly referenced in the provided CCPA statute text below.
6. Valid section format: "Section 1798.XXX" (e.g., "Section 1798.100", "Section 1798.120").
7. If the prompt is unrelated to data privacy or CCPA, return {{"harmful": false, "articles": []}}.
8. If unclear whether a violation exists, return {{"harmful": false, "articles": []}}.

CCPA STATUTE CONTEXT:
{context}

BUSINESS PRACTICE TO ANALYZE:
{user_prompt}

Respond with ONLY the JSON object. No other text.
[/INST]"""

    return prompt


def call_hf_inference(prompt: str) -> str:
    """
    Call the Hugging Face Inference API and return the raw text response.
    Handles retries for model loading and transient errors.
    """
    token = get_hf_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.01,
            "max_new_tokens": 300,
            "return_full_text": False,
            "do_sample": False,
        },
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"HF API call attempt {attempt}/{MAX_RETRIES}")
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )

            if response.status_code == 503:
                # Model is loading
                wait_time = RETRY_DELAY * attempt
                logger.warning(
                    f"Model loading (503). Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
                continue

            if response.status_code == 429:
                # Rate limited
                wait_time = RETRY_DELAY * attempt
                logger.warning(
                    f"Rate limited (429). Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
                continue

            response.raise_for_status()

            result = response.json()

            # HF API returns a list of generated texts
            if isinstance(result, list) and len(result) > 0:
                generated = result[0].get("generated_text", "")
                logger.info(f"HF API response length: {len(generated)} chars")
                return generated

            logger.warning(f"Unexpected HF API response format: {result}")
            return str(result)

        except requests.exceptions.Timeout:
            logger.error(f"HF API timeout on attempt {attempt}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
        except requests.exceptions.RequestException as e:
            logger.error(f"HF API error on attempt {attempt}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    logger.error("All HF API retry attempts failed")
    return ""
