"""
extractor.py — PDF Extractor for Loomis

Takes a textbook PDF as input and returns a dictionary where
each key is a chapter title and each value is the cleaned text of that chapter.

WHY PyMuPDF (fitz)?
    - It's one of the fastest PDF libraries in Python
    - It gives us access to font metadata (size, bold, etc.) which is
      crucial for detecting chapter headings
    - It handles complex multi-column layouts better than most alternatives

How it works:
    1. Opens the PDF and iterates through every page
    2. For each page, extract text blocks with font size information
    3. Identifies chapter boundaries by looking for heading patterns
       (large font, "Chapter X" text patterns, etc.)
    4. Groups all text between headings into chapter content
    5. Cleans the text (remove page numbers, excess whitespace, etc.)
    6. Returns a dictionary: {"Chapter 1: Title": "cleaned text...", ...}
"""

import fitz  # PyMuPDF — imported as 'fitz' for historical reasons
import re
import logging

# Set up logging so we can see what the extractor is doing
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [EXTRACTOR] %(message)s")


def is_chapter_heading(text: str, font_size: float = 0, avg_font_size: float = 12.0) -> bool:
    """
    Checks if a text block looks like a chapter heading.

    Two strategies to detect headings:
    1. Text Pattern: Does the text match patterns like "Chapter 1", "CHAPTER ONE",
       "Chapter 1: Introduction", etc.?
    2. Font Size: Is the font significantly larger than the average body text?
       (Textbooks typically use 16-24pt fonts for chapter titles vs 10-12pt for body)

    Args:
        text: The text content of the block
        font_size: The font size of this block (from PyMuPDF metadata)
        avg_font_size: The average font size across the document (for comparison)

    Returns:
        True if this looks like a chapter heading, False otherwise
    """
    # Clean up the text for pattern matching
    stripped = text.strip()

    # Skip empty lines or very long lines (headings are usually short)
    if not stripped or len(stripped) > 200:
        return False

    # Strategy 1: Regex pattern matching for common chapter heading formats
    # This catches: "Chapter 1", "CHAPTER ONE", "Chapter 1: Introduction",
    #               "Chapter 1 — The Beginning", "CHAPTER I", etc.
    chapter_pattern = re.compile(
        r"^chapter\s+"                      # Must start with 'chapter' (case-insensitive)
        r"(\d+|[IVXLC]+|"                   # Followed by a number (1, 2, 3...) or Roman numeral
        r"one|two|three|four|five|six|"      # ...or a spelled-out number
        r"seven|eight|nine|ten|eleven|"
        r"twelve|thirteen|fourteen|fifteen|"
        r"sixteen|seventeen|eighteen|"
        r"nineteen|twenty)"
        r"(\s*[:\-—.]\s*.*)?$",             # Optionally followed by a colon/dash and title
        re.IGNORECASE
    )

    if chapter_pattern.match(stripped):
        return True

    # Strategy 2: Font size heuristic
    # If the font is 1.5x or more larger than average, it's likely a heading
    if font_size > 0 and avg_font_size > 0:
        if font_size >= avg_font_size * 1.5:
            # But only if the text is short enough to be a title (not a pull quote)
            if len(stripped) < 100:
                return True

    return False


def clean_text(text: str) -> str:
    """
    Cleans up raw extracted text to make it suitable for LLM processing.

    This function handles cases such as:
    - Page numbers scattered throughout
    - Figure/table captions we don't want in the podcast
    - Headers/footers repeated on every page
    - Excessive whitespace and line breaks
    - Math equations rendered as garbage characters

    Args:
        text: Raw text extracted from a PDF chapter

    Returns:
        Cleaned text ready to be sent to the LLM
    """
    # Remove standalone page numbers (lines that are just a number)
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)

    # Remove common figure/table captions
    text = re.sub(r"(?i)^(figure|fig\.|table)\s+\d+[\.\-:].*$", "", text, flags=re.MULTILINE)

    # Remove lines that are mostly non-ASCII characters (likely math equations)
    lines = text.split("\n")
    cleaned_lines: list[str] = []
    for line in lines:
        if not line.strip():
            cleaned_lines.append("")
            continue
        # Count how many characters are "normal" (letters, digits, basic punctuation)
        ascii_chars = sum(1 for c in line if c.isascii() and (c.isalnum() or c in " .,;:!?'-\"()"))
        total_chars = len(line.strip())
        # Keep the line if at least 50% of characters are normal ASCII
        if total_chars == 0 or (ascii_chars / total_chars) >= 0.5:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Collapse multiple blank lines into a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove excessive spaces (but keep single spaces between words)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text