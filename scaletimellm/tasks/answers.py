"""Answer extraction and tolerant comparison helpers."""

from __future__ import annotations

import math
import re
from typing import Any

_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")
_FRACTION_RE = re.compile(r"[-+]?\d+\s*/\s*\d+")
_MARKER_PATTERNS = [
    re.compile(r"####\s*([^\n]+)"),
    re.compile(r"(?:final\s+answer|answer)\s*(?:is|=|:)\s*([^\n]+)", re.IGNORECASE),
]


def canonical_number(value: float | int | str) -> str | None:
    """Canonicalise a numeric value to a short deterministic string."""
    try:
        number = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.10g}"


def parse_number(text: str | None) -> float | None:
    """Parse the first number (or simple fraction) in ``text``."""
    if text is None:
        return None
    cleaned = str(text).replace(",", "").replace("$", "").strip()
    fraction = _FRACTION_RE.search(cleaned)
    if fraction:
        numerator, denominator = fraction.group(0).split("/")
        try:
            return float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError):
            return None
    match = _NUMBER_RE.search(cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def numbers_match(
    gold: str | float | int,
    prediction: str | float | int | None,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-4,
) -> bool:
    """Compare numbers with absolute and relative tolerance."""
    left = parse_number(str(gold))
    right = parse_number(None if prediction is None else str(prediction))
    if left is None or right is None:
        return False
    return math.isclose(left, right, rel_tol=rel_tol, abs_tol=abs_tol)


def extract_final_number(completion: str | None) -> str | None:
    """Extract the final numeric answer after ``####`` or an answer phrase.

    The last marker wins if a model repeats itself.  If no marker is present,
    the last number in the completion is used, which matches common GSM8K
    evaluation practice while remaining deterministic.
    """
    if not completion:
        return None
    text = str(completion).strip()
    for pattern in _MARKER_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            candidate = matches[-1].group(1)
            number = parse_number(candidate)
            if number is not None:
                return canonical_number(number)
    matches = list(_NUMBER_RE.finditer(text))
    if matches:
        return canonical_number(matches[-1].group(0))
    return None


def extract_last_line(completion: str | None) -> str | None:
    """Return the last non-empty line, stripping common prefixes."""
    if not completion:
        return None
    lines = [line.strip() for line in str(completion).splitlines() if line.strip()]
    if not lines:
        return None
    text = lines[-1]
    text = re.sub(r"^(?:final\s+answer|answer)\s*(?:is|=|:)\s*", "", text, flags=re.I)
    text = re.sub(r"^####\s*", "", text)
    text = text.strip().strip("`\"' ")
    return text or None


def extract_fenced_code(completion: str | None) -> str | None:
    """Extract Python code from a fenced block or from bare ``def`` source."""
    if not completion:
        return None
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", str(completion), flags=re.S | re.I)
    if fences:
        return fences[-1].strip()
    text = str(completion).strip()
    if re.search(r"^\s*(?:from\s+\w+\s+import|import\s+\w+|def\s+\w+)", text, re.M):
        start = re.search(r"^\s*(?:from\s+\w+\s+import|import\s+\w+|def\s+\w+)", text, re.M)
        return text[start.start() :].strip() if start else text
    return None


def relaxed_string_match(gold: Any, prediction: Any, numeric_tol: float = 0.05) -> bool:
    """Case-insensitive string match with a relative numeric tolerance."""
    if prediction is None:
        return False
    left = str(gold).strip().lower()
    right = str(prediction).strip().lower()
    right = re.sub(r"[.!?]+$", "", right)
    if left == right:
        return True
    left_number = parse_number(left)
    right_number = parse_number(right)
    if left_number is not None and right_number is not None:
        if math.isclose(left_number, right_number, rel_tol=numeric_tol, abs_tol=1e-9):
            return True
        scale = max(abs(left_number), 1e-9)
        return abs(left_number - right_number) / scale <= numeric_tol
    return False
