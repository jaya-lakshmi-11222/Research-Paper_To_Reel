# Research Paper to Reel

> Automatically convert research papers into short AI-generated explainer videos using PDF understanding, Large Language Models, Text-to-Speech, and Generative Video Models.

---

# Overview

Research Paper to Reel is an end-to-end AI pipeline that transforms academic research papers into engaging short-form video reels.

The system extracts content from a research paper PDF, understands the paper using Gemini AI, generates a structured storyboard, creates narration audio, produces AI-generated video clips using LTX-2.3, and finally merges everything into a professional explainer reel.

The goal is to make complex research papers easier to understand and share through visually engaging video content.

---

# Features

## Research Paper Understanding

Extracts text directly from PDF files

Automatically identifies:

- Title
- Abstract
- Introduction
- Methodology
- Architecture
- Experiments
- Results
- Conclusion

## AI Research Analysis

Uses Gemini 2.5 Flash for paper comprehension.

Generates:

- Problem statement
- Methodology summary
- Architecture explanation
- Experimental setup
- Results interpretation
- Impact analysis

## Storyboard Generation

Creates a complete video storyboard including:

- Narration script
- Scene descriptions
- Camera movements
- Audio cues
- Visual prompts

## AI Narration

Supports multiple TTS engines:

1. Kokoro TTS (Primary)
2. Edge-TTS (Fallback)
3. Google TTS (Final fallback)

## AI Video Generation

Uses LTX-2.3 Diffusion Video Model to generate:

- Cinematic scenes
- Motion-based storytelling
- Research-specific visualizations
- Scene transitions

## Final Video Production

- Crossfade scene transitions
- Background ambience mixing
- Narration synchronization
- Automatic video rendering

---

# System Architecture

```text
Research Paper PDF
        │
        ▼
PDF Text Extraction (PyMuPDF)
        │
        ▼
Section Detection
        │
        ▼
Gemini 2.5 Flash
(Research Understanding)
        │
        ▼
Research Understanding JSON
        │
        ▼
Gemini Storyboard Generation
        │
        ▼
Storyboard JSON
        │
        ├──────────────► Narration Script
        │                       │
        │                       ▼
        │                TTS Generation
        │
        ▼
Video Scene Prompts
        │
        ▼
LTX-2.3 Video Generation
        │
        ▼
Scene Videos
        │
        ▼
FFmpeg Compilation
        │
        ▼
Final Research Reel
```

---

# Tech Stack

## AI Models

| Component | Model |
|------------|---------|
| Research Understanding | Gemini 2.5 Flash |
| Storyboard Generation | Gemini 2.5 Flash |
| Video Generation | LTX-2.3 |
| Text-to-Speech | Kokoro TTS |
| Fallback TTS | Edge-TTS |
| Backup TTS | gTTS |

---

## Libraries

### Core

- Python 3.10+
- PyMuPDF
- Google Generative AI SDK
- Torch
- Transformers
- Diffusers

### Audio

- Kokoro
- Edge-TTS
- gTTS
- SoundFile

### Video

- LTX-2.3
- FFmpeg

### Utilities

- NumPy
- AsyncIO
- JSON
- Pathlib

---

#  Project Structure

```text
reel_project/
│
├── paper/
│   └── research_paper.pdf
│
├── extracted/
│   ├── paper_content.json
│   └── research_understanding.json
│
├── storyboard/
│   └── storyboard.json
│
├── narration/
│   └── full_narration.wav
│
├── scenes/
│   ├── clip_1.mp4
│   ├── clip_2.mp4
│   ├── clip_3.mp4
│   ├── clip_4.mp4
│   └── clip_5.mp4
│
├── output/
│   └── final_reel.mp4
│
└── research_paper_to_reel.py
```

---

# Workflow

## Step 1: PDF Extraction

The system loads the research paper and extracts:

- Title
- Abstract
- Introduction
- Methodology
- Architecture
- Experiments
- Results
- Conclusion

**Output:**

```text
paper_content.json
```

---

## Step 2: Research Understanding

Gemini analyzes the extracted content and creates:

```json
{
  "problem": "...",
  "methodology": "...",
  "architecture": "...",
  "experiments": "...",
  "results": "...",
  "conclusion_and_impact": "..."
}
```

**Output:**

```text
research_understanding.json
```

---

## Step 3: Storyboard Creation

Gemini generates:

- Narration script
- Video prompts
- Scene descriptions
- Audio cues
- Camera movements

**Output:**

```text
storyboard.json
```

---

## Step 4: Narration Generation

Priority order:

```text
Kokoro TTS
      ↓
Edge-TTS
      ↓
Google TTS
```

**Output:**

```text
full_narration.wav
```

---

## Step 5: Video Generation

```text
Clip 1 → Problem
Clip 2 → Methodology
Clip 3 → Experiments
Clip 4 → Results
Clip 5 → Conclusion
```

**Outputs:**

```text
clip_1.mp4
clip_2.mp4
clip_3.mp4
clip_4.mp4
clip_5.mp4
```

---

## Step 6: Final Rendering

FFmpeg performs:

- Scene merging
- Crossfade transitions
- Audio mixing
- Narration synchronization

**Output:**

```text
final_reel.mp4
```

---

#  Example Pipeline Output

```text
Research Paper PDF
       ↓
Paper Understanding JSON
       ↓
Storyboard JSON
       ↓
Narration Audio
       ↓
AI Video Clips
       ↓
Final Reel Video
```

---

# Future Improvements

-  Real-time video generation
-  Automatic citation visualization
-  NotebookLM-style conversational videos
-  Interactive educational reels
