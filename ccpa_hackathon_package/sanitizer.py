"""
sanitizer.py — JSON validation and sanitization for LLM output.

Ensures the API response strictly conforms to the expected format:
  {"harmful": bool, "articles": ["Section 1798.XXX", ...]}

Handles:
  - Invalid JSON repair
  - Type coercion (string "true"/"false" → bool)
  - Section number validation against whitelist
  - Deduplication
  - Fallback to safe default on unrecoverable errors
"""

import json
import logging
import re

from ccpa_sections import filter_valid_sections

logger = logging.getLogger(__name__)

# Safe default response
SAFE_DEFAULT = {"harmful": False, "articles": []}


def extract_json_from_text(text: str) -> str | None:
    """
    Extract JSON object from LLM output that may contain extra text.
    Looks for the first {...} block in the response.
    """
    # Try to find JSON object boundaries
    # Handle potential nested braces by finding matching pairs
    depth = 0
    start = -1

    for i, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]

    return None


def repair_json(raw: str) -> dict | None:
    """
    Attempt to repair malformed JSON from LLM output.
    Tries multiple strategies in order of likelihood.
    """
    if not raw or not raw.strip():
        return None

    # Strategy 1: Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON object from surrounding text
    json_str = extract_json_from_text(raw)
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Strategy 3: Fix common issues
    cleaned = raw.strip()
    # Remove markdown code fences
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```\s*", "", cleaned)
    # Remove trailing commas before closing brackets/braces
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    # Fix single quotes to double quotes
    cleaned = cleaned.replace("'", '"')

    json_str = extract_json_from_text(cleaned)
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Strategy 4: Regex-based extraction
    try:
        harmful_match = re.search(
            r'"harmful"\s*:\s*(true|false|"true"|"false")', cleaned, re.IGNORECASE
        )
        articles_match = re.search(
            r'"articles"\s*:\s*\[(.*?)\]', cleaned, re.DOTALL
        )

        if harmful_match:
            harmful_str = harmful_match.group(1).strip('"').lower()
            harmful = harmful_str == "true"

            articles = []
            if articles_match:
                articles_raw = articles_match.group(1)
                articles = re.findall(r'"([^"]+)"', articles_raw)

            return {"harmful": harmful, "articles": articles}
    except Exception as e:
        logger.warning(f"Regex extraction failed: {e}")

    return None


def sanitize_response(raw_output: str) -> dict:
    """
    Parse, validate, and sanitize the LLM output into a strict response format.

    Returns:
        dict with exactly {"harmful": bool, "articles": list[str]}
    """
    if not raw_output or not raw_output.strip():
        logger.warning("Empty LLM output, returning safe default")
        return SAFE_DEFAULT.copy()

    logger.debug(f"Raw LLM output: {raw_output[:500]}")

    # Step 1: Parse JSON
    parsed = repair_json(raw_output)
    if parsed is None:
        logger.warning("Failed to parse LLM output as JSON")
        return SAFE_DEFAULT.copy()

    # Step 2: Extract and validate 'harmful' field
    harmful = parsed.get("harmful")
    if isinstance(harmful, str):
        harmful = harmful.lower() == "true"
    elif not isinstance(harmful, bool):
        harmful = False

    # Step 3: Extract and validate 'articles' field
    articles_raw = parsed.get("articles", [])
    if not isinstance(articles_raw, list):
        articles_raw = []

    # Ensure all items are strings
    articles_raw = [str(a) for a in articles_raw if a]

    # Step 4: Filter through whitelist — only valid CCPA sections survive
    articles = filter_valid_sections(articles_raw)

    # Step 5: Enforce consistency rules
    if not harmful:
        # If not harmful, articles MUST be empty
        articles = []
    elif harmful and not articles:
        # If harmful but no valid articles survived filtering, default to safe
        logger.warning(
            "harmful=true but no valid articles found after filtering. "
            "Falling back to harmful=false."
        )
        harmful = False
        articles = []

    result = {"harmful": harmful, "articles": articles}
    logger.info(f"Sanitized response: {result}")
    return result
