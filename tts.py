"""
tts.py = Text to Speech for Poscast Generation

Converts a two host podcost script in an MP3 audio file.
Each host will have a distinct voices where lines will be
stiched together with natural pauses in between.

How it works:
1. Parse the script into pairs
2. For each line, call the TTS API with correct voice
3. Convert the returned autdio bytes into a pydub AudioSegment
4. Stict all segments together
5. Export as a MP3 file
"""

from elevenlabs.client import ElevenLabs
from pydub import AudioSegment
from gtts import gTTS
import os
import io
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [TTS] %(message)s")

# Voice assignments from Eleven Labs
ALEX_VOICE = "pNInz6obpgDQGcFmaJgB" # Adam
JORDAN_VOICE = "21m00Tcm4TlvDq8ikWAM" # Rachel

# Milliseconds of seconds betwee each spoken line
SLIENCE_MS = 400

def parse_script_lines(script:str) -> list[tuple[str, str]]:
    """
    Parses a podcast script into a list of speaker, dialouge tuples.

    Except lines formtted:
    Alex: Welcome to the show!
    Jordan: Thanks, Alex. Today we're covering...

    Lines without a speaker. prefix are speaker

    Args: 
        script: The full podcast script as a string
    
    Returns:
        A list of tuples: [("Alex", "Welcome ot the show!"), ...]
    """
    parsed = []
    for line in script.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Check for Alex or Jordan prefix
        if line.startswith("Alex:"):
            text = line[5:].strip().strip('"')
            if text:
                parsed.append(("Alex", text))
        elif line.lower().startswith("jordan:"):
            text = line[7:].strip().strip('"')
            if text:
                parsed.append(("Jordan", text))
    
    logging.info(f"Parsed {len(parsed)} script lines") 
    return parsed  

def generate_audio(script: str, output_path: str, use_elevenlabs: bool = True) -> str:
    """
    Converts a podcast script into a MP3 file

    Uses ElevenLabs for realistic dual-voice audio if available.
    Falls back to gTTS (Google Text to Sppech) if ElevenLabs fails or is disabled

    Args:
        script: The full podcast script string 
        output_path: where to save the mp3 file
        use_elevenlabs: set to False to force gTTS fallback
    
    Returns:
        The path to the saved MP3 File
    """ 

    # Parse script into pairs
    lines = parse_script_lines(script)

    if not lines:
        logging.warning("No lines to convert to audio!")
        return output_path

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create a slience segment for pauses
    silence = AudioSegment.silent(duration=SLIENCE_MS)

    # Start with an empty audio segement
    final_audio = AudioSegment.empty()

    if use_elevenlabs:
        try:
            # Initialize the ElevenLabs Client
            api_key = os.environ.get("ELEVENLABS_API_KEY")
            if not api_key:
                raise ValueError("ELEVENLABS_API_KEYS are not set")

            client = ElevenLabs(api_key=api_key)
            logging.info("Using ElevenLabs for TTS")

            for i, (speaker, text) in enumerate(lines):
                voice = ALEX_VOICE if speaker == "Alex" else JORDAN_VOICE
                logging.info(f"[{i+1}/{len(lines)}] {speaker}: {text[:50]}...")

                # Call ElevenLabs API 
                audio_generator = client.text_to_speech.convert(
                    text=text,
                    voice_id=voice,
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128",
                )
                
                # Collect all audio bytes
                audio_bytes = b"".join(audio_generator)
                
                # Convert raw bytes to pydub AudioSegement
                segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))

                # Add this segment + silence to the final audi
                final_audio += segment + silence
            logging.info("ElevenLabs TTS complete")

        except Exception as e:
            logging.warning(f"ElevenLabs failed: {e}")
            logging.info("Falling back to gTTS")

            use_elevenlabs = False

    # Fallback
    if not use_elevenlabs:
        logging.info("Ussing gTTS (single voice)")

        for i, (speaker, text) in enumerate(lines):
            logging.info(f"[{i+1}/{len(lines)}] {speaker}: {text[:50]}...") 

            # gTTS generates audio and saves to BytesIO buffer
            tts = gTTS(text=text, lang="en")
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            buffer.seek(0)

            # Convert to pydub AudioSegment
            segment = AudioSegment.from_mp3(buffer)

            # Add final audio with silence padding
            final_audio += segment + silence

        logging.info("gTTS complete")

    # Export the combined audio as an MP3
    final_audio.export(output_path, format="mp3")
    logging.info(f"Audio saved at {output_path}")

    return output_path

# python3 tts.py
if __name__ == "__main__":
    test_script = """
        Alex: Welcome back to Chapter Chat! Today we're exploring ecosystems.
        Jordan: That's right, Alex. An ecosystem is a community of living organisms interacting with their environment.
        Alex: So it's not just the animals, but the soil, water, and air too?
        Jordan: Exactly. Every organism has its own role, or niche, to play.
        Alex: Fascinating! Join us next time for more Chapter Chat!
    """
    print("Generating test audio...")

    # Use gTTS for testing (no API key needed)
    output = generate_audio(test_script, "outputs/test_audio.mp3", use_elevenlabs=True)
    print(f"Audio saved to: {output}")
