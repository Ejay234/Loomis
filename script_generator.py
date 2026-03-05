"""
script_generator.py -- LLM-Powered Script Generation

Takes a chapters text content and uses Google's Gemini API to generate
a natural and engaging two podcast script

How it Works:
1. If the text is too long, it splits it into chunks
2. Send each chunk to Gemini with a crafted system prompt
3. Gemini returns a dialogue script between two hosts (Alex and Jordan)
4. If multiple chunks, combien the scripts into one continous dialogue
5. Retry up to 5 times if the API call fails
"""

import google.genai as genai
import os
import logging
import time

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [SCRIPT_GEN] %(message)s")

# The system prompt
# Tells Gemini exactly what kind of output wanted
SYSTEM_PROMPT = """
You are a podcast script writer for an educational show called "Chapter Chat."
You convert textbook content into a natural, engaging dialogue between two hosts:
- Alex: enthusiastic, asks clarifying questions, uses analogies
- Jordan: knowledgeable, explains concepts clearly, gives real-world examples
Rules:
- Preserve ALL key facts, definitions, and terminology from the source text exactly.
- Do NOT invent examples, statistics, or claims not present in the source.
- Format every line as: "Alex: [dialogue]" or "Jordan: [dialogue]"
- Aim for approximately 10 minutes of spoken content (~1,300 words of script).
- Begin with a brief introduction of the chapter topic.
- End with a short summary and a teaser for the next chapter.
- Output ONLY the script. No preamble, no metadata.
"""

def chunk_test(text: str, max_words: int = 2500, overlap_words: int = 200) -> list[str]:
    """
    Splits long text into overlapping chunks that fit within the LLM's processing sweet spot.

    Overlap because at a hard boundary, it might cut a concept in half.
    The overlap ensures the LLM sees the end of one chunk repeated at the start of the next,
    so it can maintain continutiy

    Args:
        text: the full section text
        max_words: Max words per chunk
        overlap_words: words of overlap between chunks
    
    Returns:
        A list of text chunks
    """
    words = text.split()

    # If the text is short enough, no chunking
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # Move forward but step back to create overlap
        start = end - overlap_words

    logging.info(f"Text split into {len(chunks)} chunks "
                f"({len(words)} total words, {max_words} per chunk)")

    return chunks

def generate_script(chapter_title: str, chapter_text: str) -> str:
    """
    Sends chapter text to Gemini and returns a podcast script

    Handles:
    - Automatic chunking if text is too long
    - Retry logic with exponential backoff
    - Combining scripts from multiple chunks

    Args:
        chapter_title: the title of the section
        chapter_text: the full text content of the section
    
    Returns:
        A string containing the podcast script with "Alex: ..." and "Jordan: ..." lines
    """

    # Load the API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key not found. Set it with: export GEMINI_API_KEY=your_key"
        )

    # Initialize the Gemini client
    client = genai.Client(api_key=api_key)

    # Split text into chunks
    chunks = chunk_test(chapter_text)
    all_scripts = []
    
    for i, chunk in enumerate(chunks):
        logging.info(f"Generating script for '{chapter_title} "
                    f"(chunk {i+1}/{len(chunks)})")
        
        # Build prompt
        if len(chunks) == 1:
            user_prompt = (
                f"Convert the following section titled '{chapter_title}' "
                f" into a podcast script:\n\n{chunk}"
            )
        else:
            user_prompt = (
                f"Convert the following text into a podcast script."
                f"This is part {i+1} of {len(chunks)} for the chapter "
                f"titled '{chapter_title}'."
                f"{'Start with an introduction.' if i == 0 else 'Continue the conversation naturally.'} "
                f"{'End with a summary.' if i == len(chunks)-1 else 'Do not end the conversation yet.'}"
                f"\n\n{chunk}"
            )
        
        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.7,
                        max_output_tokens=4096,
                    ),
                )

                script_text = response.text
                all_scripts.append(script_text)
                logging.info(f"Chunk {i+1} processed - "
                            f"{len(script_text)} chars generated")
                break

            except Exception as e:
                wait_time = 2 ** attempt
                logging.error(f"API error (attempt {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    logging.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Failed after {max_retries} attempts.")
                    raise
        
    full_script = "\n\n".join(all_scripts)
    logging.info(f"Script generation complete for '{chapter_title}'"
                f"{len(full_script)} total chars")

    return full_script

# Usage: python script_generator.py
if __name__ == "__main__":
    test_text = """
    Ecosystems are complex communities of living organisms interacting with
    their physical environment. An ecosystem includes all the living things
    (plants, animals, and organisms) in a given area, interacting with each
    other and with their non-living environments (weather, earth, sun, soil,
    climate, atmosphere). In an ecosystem, each organism has its own niche
    or role to play.
    """
    print("Generating test script...")
    print("(Make sure GEMINI_API_KEY is set in your environment)")
    print()
    try:
        script = generate_script("Test: Ecosystems", test_text)
        print("=" * 60)
        print("GENERATED SCRIPT")
        print("=" * 60)
        print(script)
    except ValueError as e:
        print(f"Error: {e}")

