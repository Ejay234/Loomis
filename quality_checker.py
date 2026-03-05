"""
quality_checker.py - Semantic Hallucination Detection

Compares a generated podcast script agianst the original source text
to identify lines that may be hallucinated (not supported by the source)

How it works:
1. Split both the source text and script into sentences
2. Convert all sentences into vector embeddings using a pre-trained model
3. For each script sentence, find the most similar source sentence
4. If the best match score is below the threshold, flag for suspicous
5. Calculate an overall "fidelity score" for the script
"""

from sentence_transformers import SentenceTransformer, util
import json
import os
import re
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [QUALITY] %(message)s")

# Load the embedding model
logging.info("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def split_into_sentences(text: str) -> list[str]:
    """
    Splits text into sentences using punctuation as delimiters
    
    We split on period, question mark, and exclamation mark,
    but only when followed by a space or end of string
    This avoids splitting on abbreviations like Dr. or U.S.

    Args:
        text: A block of text

    Returns:
        A list of individual sentences
    """

    # Split on sentence_ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Filter out empty strings and very short fragments
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences

def strip_speaker_prefix(lines: str) -> str:
    """
    Removes the Alex: or Jordan: prefix from a script line
    We need the raw dialouge text for comparison

    Args:
        line: A line from the podcast script

    Returns:
        The line with the speaker prefix removed
    """
    
    # Match Alex: or Jordan: at the start
    cleaned = re.sub(r'(Alex|Jordan)\s*:\s*', '', lines, flags=re.IGNORECASE)
    return cleaned.strip()

def check_quality(source_text: str, script: str, flag_threshold: float = 0.50) -> dict:
    """
    Compares the script agianst the source text using semantic similarity.

    For each sentence in the script, we find the most similar sentence in the source text.
    If the best match is below the threshold, that line is flagged as potentially hallucinated

    Args:
        source_text: The original chapter text
        script: The generated podcast script
        flag_threshold: Similarity score below which a line is flagged 
        (0.50 is a good default to prevent flagging creative paraphrasing)

    Returns:
        A dictionary with:
        -"fidelity_score": float (0-1, higher is better)
        -"flagged_lines": list of dicts with line and score
        -"total_lines": int
        -"flagged_count": int
    """

    logging.info("Runnign quality check...")

    # Split score and script into sentences
    source_sentences = split_into_sentences(source_text)
    script_lines = script.strip().split("\n")

    # Clean script lines
    script_sentences = []
    for line in script_lines:
        cleaned = strip_speaker_prefix(line)
        if len(cleaned) > 10:
            script_sentences.append(cleaned)
    
    if not script_sentences or not source_sentences:
        logging.warning("Not enough text to compare")
        return {
            "fidelity_score": 0.0,
            "flagged_lines": [],
            "passed": False,
            "total_lines": 0,
            "flagged_count": 0
        }
    
    logging.info(f"Comparing {len(script_sentences)} script sentences"
                f"against {len(source_sentences)} source sentences")

    # Encode all sentences into vectors
    source_embeddings = model.encode(source_sentences, convert_to_tensor=True)
    script_embeddings = model.encode(script_sentences, convert_to_tensor=True)

    # For each script sentence, find the most similar source sentence
    flagged_lines = []
    similarity_scores = []

    for i, script_emb in enumerate(script_embeddings):
        # Compute cosine similarity between the script sentence and all source sentences
        # take the highest match
        similarities = util.cos_sim(script_emb, source_embeddings)[0]
        best_score = float(similarities.max())
        similarity_scores.append(best_score)
        logging.info(f"  [{best_score:.3f}] {script_sentences[i][:80]}")


        if best_score < flag_threshold:
            flagged_lines.append({
                "line": script_sentences[i],
                "score": round(best_score, 3)
            })
    
    # Calculate overall fidelity score (average of all best scores)
    fidelity_score = sum(similarity_scores) / len(similarity_scores)

    result = {
        "fidelity_score": round(fidelity_score, 3),
        "flagged_lines": flagged_lines,
        "passed": fidelity_score >= 0.70,
        "total_lines": len(script_sentences),
        "flagged_count": len(flagged_lines)
    }

    logging.info(f"Fidelity score: {result['fidelity_score']}")
    logging.info(f"Flagged lines: {result['flagged_count']}/{result['total_lines']}")
    return result

def save_quality_report(report: dict, chapter_title: str, output_dir: str="outputs/quality_reports"):
    """
    Saves the quality report to a JSON file

    Args:
        report: The quality report dictionary
        chapter_title: Used to name the output file
        output_dir: The directory to save the report in
    """
    os.makedirs(output_dir, exist_ok=True)

    # Clean the title for use as a filename
    safe_title = re.sub(r'[^\w\s-]', '', chapter_title).strip().replace(' ', '_')
    filepath = os.path.join(output_dir, f"{safe_title}_quality.json")
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"Quality report saved to {filepath}")
