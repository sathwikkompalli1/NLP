"""
IQAS Text Cleaning Utilities
=============================
Normalize and clean raw text extracted from documents.
"""

import re
import unicodedata
from typing import Optional

from utils.logger import get_logger

log = get_logger("cleaner")


def clean_text(text: str, preserve_latex: bool = False) -> str:
    """
    Clean and normalize raw document text.

    Steps:
        1. Fix hyphenated line breaks ("computa-\\ntion" -> "computation")
        2. Normalize unicode (NFKD)
        3. Remove non-printable characters
        4. Remove PDF artifacts (form feeds, excessive whitespace)
        5. Normalize whitespace while preserving sentence boundaries
        6. Optionally strip LaTeX equations

    Args:
        text: Raw text extracted from a document.
        preserve_latex: If True, keep LaTeX equations; otherwise strip them.

    Returns:
        Cleaned text string.
    """
    if not text or not text.strip():
        return ""

    # Step 1: Fix hyphenated line breaks
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

    # Step 2: Normalize unicode
    text = unicodedata.normalize("NFKD", text)

    # Step 3: Remove non-printable characters (keep newlines and tabs)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t", "\r")
    )

    # Step 4: Remove PDF artifacts
    text = text.replace("\f", "")  # Form feeds
    text = re.sub(r'\x00', '', text)  # Null bytes

    # Step 5: Handle LaTeX
    if not preserve_latex:
        # Remove inline LaTeX: $...$
        text = re.sub(r'\$[^$]+\$', ' [EQUATION] ', text)
        # Remove block LaTeX: \[...\] or \begin{equation}...\end{equation}
        text = re.sub(r'\\\[.*?\\\]', ' [EQUATION] ', text, flags=re.DOTALL)
        text = re.sub(
            r'\\begin\{equation\}.*?\\end\{equation\}',
            ' [EQUATION] ',
            text,
            flags=re.DOTALL,
        )

    # Step 6: Remove common PDF headers/footers patterns
    text = _remove_headers_footers(text)

    # Step 7: Normalize whitespace
    # Replace multiple spaces with single space (preserve newlines for paragraph detection)
    text = re.sub(r'[^\S\n]+', ' ', text)
    # Collapse more than 2 consecutive newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    # Final strip
    text = text.strip()

    return text


def _remove_headers_footers(text: str) -> str:
    """
    Remove common PDF header/footer patterns.

    Patterns removed:
        - Page numbers (standalone lines like "Page 5", "- 3 -", "5")
        - Repeated short lines that appear on every page
    """
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip standalone page numbers
        if re.match(r'^(Page\s+)?\d+$', stripped, re.IGNORECASE):
            continue
        if re.match(r'^-\s*\d+\s*-$', stripped):
            continue

        # Skip very short lines that look like headers (all caps, < 5 words)
        if (
            len(stripped) > 0
            and len(stripped.split()) <= 3
            and stripped.isupper()
            and len(stripped) < 50
        ):
            # Keep it if it looks like a real heading (followed by content)
            cleaned_lines.append(line)
            continue

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace (including newlines) into single spaces."""
    return re.sub(r'\s+', ' ', text).strip()


def remove_special_chars(text: str, keep_punctuation: bool = True) -> str:
    """
    Remove special characters from text.

    Args:
        text: Input string.
        keep_punctuation: If True, keep standard punctuation (.!?,;:'-).

    Returns:
        Cleaned string.
    """
    if keep_punctuation:
        pattern = r'[^a-zA-Z0-9\s.,!?;:\'\"-]'
    else:
        pattern = r'[^a-zA-Z0-9\s]'
    return re.sub(pattern, ' ', text)
