import json
import re
from typing import Any


def extract_json_from_string(text: str) -> dict[str, Any] | None:
    """Extract the first valid JSON object from an LLM response."""
    if not text:
        return None

    text = text.strip()

    # Remove common markdown fences.
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False

        for i in range(start, len(text)):
            ch = text[i]

            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        value = json.loads(candidate)
                        if isinstance(value, dict):
                            return value
                    except json.JSONDecodeError:
                        break

        start = text.find("{", start + 1)

    return None
