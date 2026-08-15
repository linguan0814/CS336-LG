from __future__ import annotations

import re
from typing import Any


# Extracts an A/B/C/D option from an MMLU model response when possible.
def parse_mmlu_response(mmlu_example: dict[str, Any], model_output: str) -> str | None:
    del mmlu_example
    match = re.search(r"(?:answer\s+is|answer:|option)\s*([ABCD])\b", model_output, re.I)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([ABCD])\b", model_output)
    if match:
        return match.group(1)
    return None


# Extracts the last numeric value from a GSM8K model response when possible.
def parse_gsm8k_response(model_output: str) -> str | None:
    matches = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", model_output)
    if not matches:
        return None
    return matches[-1].replace(",", "")
