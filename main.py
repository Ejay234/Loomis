"""
main.py - Loomis

Runs the full textbook-to-podcast pipeline:
PDF -> Chapter Extraction -> Script Generation -> Quality Check -> Audio

Usage:
    python main.py textbooks/test_file.pdf
    python main.py textbooks/test_file.pdf --chapter "Ecosystems"
    python main.py textbooks/test_file.pdf --no-elevenlabs
"""

import argparse
import os
import json
from dotenv import load_dotenv

# Import out pipeline modules
from extractor import extract_chapters
from script_generator import generate_script
from tts import tts
from quality_check import quality_check

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [MAIN] %(message)s")
load_dotenv()

def run_pipeline(pdf_path: str, chapter_filter: str = None, use_elevenlabs: bool = True):
    """
    Runs the full pipeline

    Args:
        pdf_path (str): Path to the PDF file
        chapter_filter (str, optional): Chapter to extract. Defaults to None.
        use_elevenlabs (bool, optional): Whether to use ElevenLabs for TTS. Defaults to True.
    """

    logging.info(f"Extracting chapter from {pdf_path}")
    # Extract chapters from the PDF
    chapters = extract_chapters(pdf_path)

    if not chapters:
        logging.error("No chapters found.")
        return

    logging.info(f"Found {len(chapters)} chapter(s)")
    # Filter to a specific chapter, if specficied
    
    if chapter_filter:
        chapters = {name: text for name, text in chapters.items() if chapter_filter.lower() in name.lower()}
        if not chapters:
            logging.error(f"No chapters matching '{chapter_filter}")
            return
        logging.info(f"Filtered to {len(chapters)} chapter(s)")

    # Create output directory
    os.mkdir("outputs", exist_ok=True)

    for i, (chapter_name, chapter_text) in enumerate(chapters.items(), 1):
        logging.info(f"Processing [{i}/{len(chapters)}]: {chapter_name}")

        # Create a filename from chapter name
        safe_name = chapter_name.lower().replace(" ", "_")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")

        # Generate the script
        script = generate_script(chapter_text, chapter_name)
        if not script:
            logging.warning(f"Script generation failed for {chapter_name}, skipping.")
            continue

        script_path = f"outputs/{safe_name}_script.txt"
        with open(script_path, "w") as f:
            f.write(script)
        logging.info(f"Script saved to {script_path}")   

        # Quality check
        report = check_quality(chapter_text, script)
        # Save the quality report
        report_path = f"outputs/{safe_name}_quality.json"
        save_quality_report(report, report_path)
        logging.info(f"Fidelity score: {report['fidelity_score']:.3f}")
        if report["flagged_lines"]:
            logging.warning(f"{report['flagged_count']} line(s) flagged for review")

        # Generate Audio
        audio_path = f"outputs/{safe_name}.mp3"
        generate_audio(script, audio_path, use_elevenlabs=use_elevenlabs)
    
    logging.info("Pipeline complete!")

# Usage: python main.py file_path/file.pdf

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Loomis: Textbook to Podcast")
    parser.add_argument("pdf", help="Path to the textbook PDF")
    parser.add_argument("--chapter", help="Only process chapters matching this text", default=None)
    parser.add_argument("--no-elevenlabs", action="store_true", help="Use gTTS instead of ElevenLabs")
    
    args = parser.parse_args()
    
    run_pipeline(
        pdf_path=args.pdf,
        chapter_filter=args.chapter,
        use_elevenlabs=not args.no_elevenlabs,
    )

        