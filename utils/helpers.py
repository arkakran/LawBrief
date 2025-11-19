import json
import re
from typing import Any, Optional


def parse_json_response(text: str) -> Any:

    if not text or not isinstance(text, str):
        raise ValueError("LLM returned empty or non-string response")

    cleaned = text.strip()

    # Remove triple-backtick code blocks
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*", "", cleaned)
        cleaned = cleaned.replace("```", "").strip()

    # Try direct JSON load
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Extract JSON object
    obj_match = re.search(r"\{[\s\S]*\}", cleaned)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except Exception:
            pass

    # Extract JSON array
    arr_match = re.search(r"\[[\s\S]*\]", cleaned)
    if arr_match:
        try:
            return json.loads(arr_match.group(0))
        except Exception:
            pass

    raise ValueError("Could not parse JSON from LLM response")


def format_page_citation(
    page_start: Optional[int],
    page_end: Optional[int],
    line_start: Optional[int] = None,
    line_end: Optional[int] = None
) -> str:
    """
    Return clean citation string:
        Page 4, Lines 20–22
        Pages 4–6, Line 12
    """
    if not page_start:
        return "No citation"

    # Page range
    if page_start == page_end:
        page_str = f"Page {page_start}"
    else:
        page_str = f"Pages {page_start}-{page_end}"

    # Line range
    if line_start and line_end:
        if line_start == line_end:
            return f"{page_str}, Line {line_start}"
        return f"{page_str}, Lines {line_start}-{line_end}"

    return page_str


def sanitize_filename(filename: str) -> str:
    """Remove unsafe characters to avoid file traversal or shell injection."""
    filename = re.sub(r"[^\w.\- ]", "", filename)
    return filename[:200]  # safety length limit
