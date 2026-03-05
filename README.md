# Loomis - Text to Podcast Pipeline

Loomis is an end-to-end Python pipeline that transforms PDFs into engaging, dual-voice podcasts.
It combines PDF extraction, LLM-powered script generation, semantic hallucination detection, and text-to-speech synthesis.

Built as a demonstration of applied AI in education technology.

---

## Pipeline Overview
1. Textbook PDF
2. Text Extraction (extractor.py)
3. Script Generation (script_generator.py)
4. Quality Checking (quality_checker.py)
5. Text-to-Speech (tts.py)
6. Podcast Output (outputs/)

---

## Project Structure

```
Loomis/
├── extractor.py          # PDF text extraction
├── script_generator.py   # LLM script generation
├── quality_checker.py    # Semantic hallucination detection
├── tts.py                # Text-to-Speech synthesis
├── requirements.txt      # Dependencies
├── .env                  # Environment variables
├── outputs/
│   ├── quality_reports/  # Per-chapter JSON quality reports
│   ├── review/           # Scripts that failed the quality threshold
│   └── *.mp3             # Final podcast audio files
└── README.md             # Project documentation
```

## Setup

### 1. Clone the repo and create a virtual environment

```bash
git clone https://github.com/your-username/loomis.git
cd loomis
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API keys

Create a `.env` file in the project root (this is already in `.gitignore`):

```
GEMINI_API_KEY=your_anthropic_key_here
ELEVENLABS_API_KEY=your_elevenlabs_key_here
```

Then load it before running:
```bash
export $(cat .env | xargs)  # Mac/Linux
```

---

## Usage

### Run the full pipeline on a PDF

```bash
python main.py --pdf path/to/textbook.pdf
```

### Process specific chapters only

```bash
python main.py --pdf path/to/textbook.pdf --chapters 1,2,3
```

### Skip audio generation (output scripts only)

```bash
python main.py --pdf path/to/textbook.pdf --skip-tts
```

### Set a custom quality threshold

```bash
python main.py --pdf path/to/textbook.pdf --quality-threshold 0.80
```

---
