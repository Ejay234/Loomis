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


def is_heading_candidate(text: str) -> bool:
    """
    Checks if a line could be a heading
    The actual length-based detection is done in extract_chapters() using statistical analysis
    This function only checks content rules.

    Args:
        text: The text content of the block

    Returns:
        True if this looks like a chapter heading, False otherwise
    """
    # Clean up the text for pattern matching
    stripped = text.strip()

    # Skip empty lines
    if not stripped:
        return False

    # Must start with an uppercase letter
    if not stripped[0].isupper():
        return False

    # Must have at least 3 words
    if len(stripped.split()) < 3:
        return False

    # Must not end with punctuation
    if stripped.endswith((',',';')):
        return False

    # Must have balanced parentheses
    if stripped.count('(') != stripped.count(')'):
        return False

    # Must not end with an abbreviation
    last_word = stripped.split()[-1]
    if re.match(r'[A-Z][A-Za-z]?\d*$', last_word) and len(last_word) <= 4:
        return False

    # Classic Chapter X pattern
    chapter_pattern = re.compile(
        r"^chapter\s+(\d+|[IVXLC]+|one|two|three|four|five|"
        r"six|seven|eight|nine|ten|eleven|twelve|thirteen|"
        r"fourteen|fifteen|sixteen|seventeen|eighteen|"
        r"nineteen|twenty)(\s*[:\-—.]\s*.*)?$",
        re.IGNORECASE
    )
    if chapter_pattern.match(stripped):
        return True # Always true regardless of length
    
    return True 

def compute_heading_threshold(doc: fitz.Document) -> float:
    """
    Scans the entire document and computes a line-length threshold for heading detection.

    Logic: Body text lines cluster around a consistent length.
    Headings are statistical outliers, hence 40% of the median line length is used as the cutoff

    Args:
        doc: An open PyMuPDF document

    Returns:
        The line-length threshold (float)
    """
    line_lengths = []
    
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                text = ""
                for span in line["spans"]:
                    text += span["text"]
                text = text.strip()
                if text and len(text) > 5:
                    line_lengths.append(len(text))
    
    if not line_lengths:
        return 60 # Fallback
    
    # Sort and find the median
    line_lengths.sort()
    median = line_lengths[len(line_lengths) // 2]

    threshold = median * 0.4

    logging.info(f"Median line length: {median} chars")
    logging.info(f"Heading threshold: {threshold:.0f} chars (40% of median)")

    return threshold

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


def get_average_font_size(doc: fitz.Document, sample_pages: int = 10) -> float:
    """
    Calculates the average font size across the document by sampling pages.

    Used as a baseline to detect headings
    Any text significantly larger than the average is likely a heading
    Sampled instead of scanning every page because textbooks can be 500+ pages

    Args:
        doc: An open PyMuPDF document
        sample_pages: How many pages to sample (default 10)

    Returns:
        The average font size in points (e.g., 11.5)
    """
    font_sizes = []
    # Sample pages evenly spaced throughout the document
    total_pages = len(doc)
    step = max(1, total_pages // sample_pages)

    for page_num in range(0, total_pages, step):
        page = doc[page_num]
        # get_text("dict") returns detailed block info including font sizes
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 0:  # type 0 = text block (not image)
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["text"].strip():  # Skip empty spans
                            font_sizes.append(span["size"])

    if not font_sizes:
        return 12.0 # default font size

    return sum(font_sizes) / len(font_sizes)


def extract_chapters(pdf_path: str) -> dict[str, str]:
    """
    Opens a PDF and splits it into chapters.

    Main fucntion to call, full process:
    1. Opens the PDF
    2. Calculates average font size for heading detection
    3. Walks through every page, looking for chapter boundaries
    4. Groups text between boundaries into chapters
    5. Cleans each chapter's text

    Args:
        pdf_path: Path to the PDF file (e.g., "textbooks/biology101.pdf")

    Returns:
        A dictionary where:
        - Keys are chapter titles (e.g., "Chapter 1: Introduction to Biology")
        - Values are the cleaned text content of each chapter
    """
    logging.info(f"Opening PDF: {pdf_path}")

    # Open the PDF document
    doc = fitz.open(pdf_path)
    logging.info(f"PDF has {len(doc)} pages")

    # Calculate the average font size so we can detect headings
    avg_font_size = get_average_font_size(doc)
    logging.info(f"Average font size: {avg_font_size:.1f}pt")

    # Compute the heading length threshold
    heading_threshold = compute_heading_threshold(doc)
    logging.info(f"Heading threshold: {heading_threshold:.0f} chars")

    # These will hold our results as we scan through the PDF
    chapters = {}
    current_chapter = None
    current_text = []

    # Walk through every page in the document
    for page_num in range(len(doc)):
        page = doc[page_num]

        # Extract text with full detail
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            # Skip image blocks (we only care about text)
            if block["type"] != 0:
                continue

            # Each block can contain multiple lines, each with multiple "spans"

            # Extract each line separatedly
            lines_in_block = []
            for line in block["lines"]:
                line_text = ""
                line_font_size = 0
                for span in line["spans"]:
                    line_text += span["text"]
                    line_font_size = max(line_font_size, span["size"])
                lines_in_block.append((line_text.strip(), line_font_size))

                for line_text, line_font_size in lines_in_block:
                    if not line_text:
                        continue

                    # Check if this block is a chapter heading
                    if is_heading_candidate(line_text) and len(line_text) < heading_threshold:
                        # Save the previous chapter, before starting a new one
                        if current_chapter and current_text:
                            raw_text = "\n".join(current_text)
                            chapters[current_chapter] = clean_text(raw_text)
                            logging.info(
                                f"Extracted: {current_chapter} "
                                f"({len(chapters[current_chapter])} chars)"
                            )

                        # Start a new chapter
                        current_chapter = line_text
                        current_text = []
                        logging.info(f"Found chapter heading on page {page_num + 1}: {line_text}")
                    else:
                        # Add to current chapter
                        if current_chapter and line_text:
                            current_text.append(line_text)

    # Last chapter case
    if current_chapter and current_text:
        raw_text = "\n".join(current_text)
        chapters[current_chapter] = clean_text(raw_text)
        logging.info(
            f"Extracted: {current_chapter} "
            f"({len(chapters[current_chapter])} chars)"
        )

    doc.close()

    # No chapters case
    if not chapters:
        logging.warning(
            "No chapter headings detected! "
            "Treating the entire PDF as one chapter."
        )
        # Re-open and extract all text as a single
        doc = fitz.open(pdf_path)
        all_text = ""
        for page in doc:
            all_text += page.get_text() + "\n"
        doc.close()
        chapters["Full Document"] = clean_text(all_text)

    logging.info(f"Extraction complete: {len(chapters)} chapter(s) found")
    return chapters


# Usage: python extractor.py
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extractor.py <path_to_pdf>")
        print("Example: python extractor.py textbooks/biology101.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    chapters = extract_chapters(pdf_path)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    for title, text in chapters.items():
        # Show the first 200 characters as a preview
        preview = text[:200].replace("\n", " ")
        print(f"\n📖 {title}")
        print(f"   Length: {len(text):,} characters")
        print(f"   Preview: {preview}...")
    print(f"\n{'='*60}")
    print(f"Total chapters: {len(chapters)}")


