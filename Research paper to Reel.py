import gc
import json
import math
import os
import re
import subprocess
import time
import asyncio
from pathlib import Path
import fitz                          
import google.generativeai as genai
import torch
import transformers

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")



GEMINI_API_KEY = " Paste your Gemini API Key"
GEMINI_MODEL   = "gemini-2.5-flash"

VIDEO_WIDTH      = 768
VIDEO_HEIGHT     = 512
FRAME_RATE       = 24.0
CLIP_COUNT       = 5
NUM_INFER_STEPS  = 22
GUIDANCE_SCALE   = 3.0
OVERLAP_SECONDS  = 0.5        
VISUAL_STYLE = "atmospheric_conceptual"
NARRATION_GAIN_DB = 0
AMBIENCE_GAIN_DB  = -14
DEVICE = "cuda"
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reel_project-6")

FOLDERS = {
    "paper"      : os.path.join(BASE_DIR, "paper"),
    "extracted"  : os.path.join(BASE_DIR, "extracted"),
    "storyboard" : os.path.join(BASE_DIR, "storyboard"),
    "narration"  : os.path.join(BASE_DIR, "narration"),
    "scenes"     : os.path.join(BASE_DIR, "scenes"),
    "output"     : os.path.join(BASE_DIR, "output"),
}
for folder in FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

print("Configuration loaded")
print(f"   Resolution      : {VIDEO_WIDTH}×{VIDEO_HEIGHT}")
print(f"   Segments        : {CLIP_COUNT} clips with {OVERLAP_SECONDS}s crossfades")
print(f"   Project dir      : {BASE_DIR}")



torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
transformers.logging.set_verbosity_error()
from diffusers import LTX2Pipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT

print(f"Libraries imported. Device: {DEVICE.upper()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram  = props.total_memory / 1024**3
    print(f"   GPU : {props.name}")
    print(f"   VRAM: {vram:.1f} GB")
    if vram < 12:
        print("Less than 12GB VRAM detected. Consider reducing resolution.")

paper_folder = FOLDERS["paper"]
pdf_files    = list(Path(paper_folder).glob("*.pdf"))

assert len(pdf_files) > 0, (
    f"No PDF found in {paper_folder}\n"
    f"Copy your research paper PDF into that folder and re-run."
)
if len(pdf_files) > 1:
    print(f"Multiple PDFs found — using: {pdf_files[0].name}")
PDF_PATH = str(pdf_files[0])
print(f"\nPDF: {PDF_PATH}")
doc = fitz.open(PDF_PATH)
assert not doc.is_encrypted, "PDF is password-protected. Please upload an unlocked PDF."
PAGE_COUNT = doc.page_count
doc.close()
print(f"PDF validated: {PAGE_COUNT} pages")



def extract_title(doc: fitz.Document) -> str:
    page = doc[0]
    blocks = page.get_text("dict")["blocks"]

    lines_info = []
    for block in blocks:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            spans = line["spans"]
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            max_size = round(max(s["size"] for s in spans), 1)
            y_pos = line["bbox"][1]
            lines_info.append((y_pos, max_size, text))

    if not lines_info:
        return "Unknown Title"

    lines_info.sort(key=lambda x: x[0])
    page_height = page.rect.height
    top_region = [l for l in lines_info if l[0] < page_height * 0.4]
    search_pool = top_region if top_region else lines_info
    title_size = max(l[1] for l in search_pool)

    title_lines = []
    for y_pos, size, text in lines_info:
        if abs(size - title_size) <= 1.5:
            if re.search(r'@|arxiv|^\d+$|^page\s+\d+', text, re.IGNORECASE):
                if title_lines:
                    break
                continue
            title_lines.append(text)
        elif title_lines:
            break

    if not title_lines:
        return "Unknown Title"

    full_title = " ".join(title_lines)
    return re.sub(r'\s+', ' ', full_title).strip()


def extract_section(full_text: str, section_names: list) -> str:
    numbering = r'(?:[IVXLC]+\.|\d+(?:\.\d+)*\.?)?\s*'
    name_alt = '|'.join(re.escape(s) for s in section_names)
    pattern = rf'(?im)^\s*{numbering}(?:{name_alt})(?:[:\s][A-Za-z ]{{0,40}})?\s*$'
    match = re.search(pattern, full_text)
    if not match:
        return ""

    start = match.end()
    all_headers = (
        r'(?im)^\s*(?:[IVXLC]+\.|\d+(?:\.\d+)*\.?)?\s*(?:abstract|introduction|'
        r'related work|background|literature review|'
        r'methodology|methods|approach|proposed (?:method|approach|system|framework)|'
        r'system (?:architecture|design)|architecture|design|'
        r'experiments?|evaluation|experimental (?:results|setup)|'
        r'results(?: and discussion)?|discussion|'
        r'conclusion[s]?|future work|concluding remarks|'
        r'references|acknowledg(?:e?ments?)|appendix)'
        r'(?:[:\s][A-Za-z ]{0,40})?\s*$'
    )
    next_match = re.search(all_headers, full_text[start:])
    end = start + next_match.start() if next_match else len(full_text)
    return full_text[start:end].strip()


def _fallback_slice(full_text: str, frac_start: float, frac_end: float) -> str:
    n = len(full_text)
    return full_text[int(n * frac_start):int(n * frac_end)].strip()


def trim(text: str, max_chars: int = 2500) -> str:
    return text[:max_chars] + "..." if len(text) > max_chars else text


print("\nExtracting text from PDF...")
doc        = fitz.open(PDF_PATH)
pages_text = [doc[i].get_text("text") for i in range(doc.page_count)]
full_text  = "\n".join(pages_text)

title         = extract_title(doc)
abstract      = extract_section(full_text, ["Abstract", "ABSTRACT"])
introduction  = extract_section(full_text, ["Introduction", "INTRODUCTION", "1. Introduction"])
methodology   = extract_section(full_text, [
    "Methodology", "METHODOLOGY", "Methods", "METHODS",
    "Approach", "APPROACH", "Proposed Method", "Proposed Approach"
])
architecture  = extract_section(full_text, [
    "Architecture", "ARCHITECTURE", "System Architecture", "SYSTEM ARCHITECTURE",
    "Proposed Architecture", "System Design", "Framework", "Proposed Framework",
    "Design", "DESIGN", "System model", "SYSTEM MODEL"
])
experiments   = extract_section(full_text, [
    "Experiments", "EXPERIMENTS", "Experimental Setup", "Experimental Design",
    "Evaluation Setup", "Implementation", "Implementation Details", "Setup"
])
results       = extract_section(full_text, [
    "Results", "RESULTS", "Evaluation", "EVALUATION",
    "Experimental Results", "Performance Evaluation", "Case Study",
    "Results and Discussion", "Discussion"
])
conclusion    = extract_section(full_text, [
    "Conclusion", "CONCLUSION", "Conclusions", "CONCLUSIONS",
    "Concluding Remarks", "Future Work"
])
doc.close()

if not abstract:
    abstract = _fallback_slice(full_text, 0.0, 0.06)
if not introduction:
    introduction = _fallback_slice(full_text, 0.05, 0.18)
if not methodology:
    methodology = _fallback_slice(full_text, 0.18, 0.40)
if not architecture:
    architecture = _fallback_slice(full_text, 0.40, 0.55)
if not experiments:
    experiments = _fallback_slice(full_text, 0.55, 0.68)
if not results:
    results = _fallback_slice(full_text, 0.68, 0.85)
if not conclusion:
    conclusion = _fallback_slice(full_text, 0.88, 1.0)

paper_content = {
    "title"             : title,
    "abstract"          : trim(abstract or full_text[:800]),
    "introduction"      : trim(introduction),
    "methodology"       : trim(methodology, 3000),
    "architecture"      : trim(architecture, 3000),
    "experiments"       : trim(experiments, 2000),
    "results"           : trim(results, 2500),
    "conclusion"        : trim(conclusion),
    "full_text_preview" : trim(full_text, 6000),
}

content_path = os.path.join(FOLDERS["extracted"], "paper_content.json")
with open(content_path, "w", encoding="utf-8") as f:
    json.dump(paper_content, f, indent=2, ensure_ascii=False)

print(f" Extraction complete → {content_path}")
print(f"   Title       : {title}")
print(f"   Abstract    : {len(paper_content['abstract'])} chars")
print(f"   Introduction: {len(paper_content['introduction'])} chars")
print(f"   Methodology : {len(paper_content['methodology'])} chars")
print(f"   Architecture: {len(paper_content['architecture'])} chars")
print(f"   Experiments : {len(paper_content['experiments'])} chars")
print(f"   Results     : {len(paper_content['results'])} chars")
print(f"   Conclusion  : {len(paper_content['conclusion'])} chars")



genai.configure(api_key=GEMINI_API_KEY)
gemini        = genai.GenerativeModel(GEMINI_MODEL)
GEMINI_CONFIG = genai.GenerationConfig(
    temperature=0.75,
    top_p=0.95,
    max_output_tokens=8192,
    response_mime_type="application/json"
)
def call_gemini(prompt: str, retries: int = 3) -> str:
    for attempt in range(1, retries + 1):
        try:
            return gemini.generate_content(prompt, generation_config=GEMINI_CONFIG).text
        except Exception as e:
            wait = 2 ** attempt
            print(f"Gemini attempt {attempt} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Gemini API failed after all retries.")
def extract_json(text: str):
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Primary JSON parse failed: {e}")
        for pat in (r'(\{[\s\S]+\})', r'(\[[\s\S]+\])'):
            m = re.search(pat, text)
            if m:
                try:
                    res = json.loads(m.group(1))
                    print("Regex fallback JSON parse succeeded.")
                    return res
                except json.JSONDecodeError as fallback_err:
                    print(f"Fallback parse failed: {fallback_err}")
        debug_path = "gemini_failed_response.txt"
        with open(debug_path, "w", encoding="utf-8") as fd:
            fd.write(text)
        print(f"Saved failed Gemini response to {debug_path}")
    raise ValueError(f"Could not parse JSON from Gemini response:\n{text[:500]}")
print(f"\n Gemini configured. Model: {GEMINI_MODEL}")



UNDERSTANDING_PROMPT = f"""
You are simultaneously:
  - A senior researcher in this exact subfield who has read this paper closely
  - An expert science communicator producing a technical explainer video
  - A content strategist who never sacrifices technical accuracy for simplicity

Read the paper content below and produce a Research Understanding JSON that
will be the SOLE source of truth for an explainer video covering EVERY major
section of this paper — abstract, introduction, methodology, architecture,
experiments, results, and conclusion. Every claim must be traceable to the
text below — do not invent or default to vague "AI helps with X" language.

PAPER TITLE: {paper_content['title']}

ABSTRACT:
{paper_content['abstract']}

INTRODUCTION:
{paper_content['introduction']}

METHODOLOGY / APPROACH:
{paper_content['methodology']}

ARCHITECTURE / SYSTEM DESIGN:
{paper_content['architecture']}

EXPERIMENTS / SETUP:
{paper_content['experiments']}

RESULTS:
{paper_content['results']}

CONCLUSION:
{paper_content['conclusion']}

ADDITIONAL CONTEXT (use if sections above are incomplete):
{paper_content['full_text_preview']}

---

RULES:
1. Use the paper's ACTUAL terminology, named methods, system names, protocols,
   architecture components, dataset names, and specific numbers wherever they
   appear above.
2. If the paper names a specific architecture/framework/algorithm/system, name
   it explicitly rather than describing it abstractly.
3. For "results", include actual figures/metrics/percentages if present in the text.
4. Each field: 3-4 sentences, dense with real content, no padding, no field left
   generic — if the paper's related work names specific prior systems or papers,
   name them; if experiments specify datasets/metrics/hardware, name those too.
5. Plain language for a general technical audience, but never drop a specific
   term, number, or named component from the paper.

Return ONLY valid JSON, no markdown fences, no preamble:

{{
  "abstract_summary": "A faithful 2-3 sentence summary of the paper's own abstract, in plain language but preserving its specific claims — this is what opens the video.",
  "problem": "The specific real-world problem/gap this paper addresses, named concretely, and why it's hard.",
  "methodology": "The actual technical pipeline/approach — name real steps or concepts mentioned in the paper.",
  "architecture": "The system architecture/design — name real components, layers, modules, or modules mentioned in the paper.",
  "experiments": "The actual experimental setup — datasets, baselines compared against, metrics used, hardware/scale, or test scenarios, as described in the paper.",
  "results": "Actual quantitative/qualitative findings, with real numbers, percentages, or comparisons if present in the text.",
  "conclusion_and_impact": "The paper's own concluding statement plus who concretely benefits and what becomes possible because of this specific work — this is what closes the video."
}}
"""

print("\n Gemini Pass 1 — Research understanding...")
research_understanding = extract_json(call_gemini(UNDERSTANDING_PROMPT))

required = {
    "abstract_summary", "problem", "methodology", "architecture",
    "experiments", "results", "conclusion_and_impact"
}
missing  = required - set(research_understanding.keys())
assert not missing, f"Missing fields: {missing}"
understanding_path = os.path.join(FOLDERS["extracted"], "research_understanding.json")
with open(understanding_path, "w") as f:
    json.dump(research_understanding, f, indent=2)
print(f" Saved → {understanding_path}")
for k, v in research_understanding.items():
    print(f"{k.upper():12s}: {v[:100]}...")



VISUAL_STYLE_INSTRUCTIONS = {
    "atmospheric_conceptual": (
        "Visual approach: every scene must show a CONCRETE, RECOGNIZABLE real-world "
        "subject specific to THIS paper's domain. Before writing each video_prompt, identify the "
        "actual physical things this paper's domain involves (e.g. for a transportation/"
        "digital-twin paper: a real city intersection, a traffic sensor mounted on a "
        "pole, a fleet of vehicles on a highway, a server rack, a control-room wall of "
        "monitors, a satellite view of road networks, a single car's dashboard view) and "
        "build the shot around ONE of those concrete subjects. Each scene's subject "
        "must be DIFFERENT from the scenes before and after it, so the video doesn't "
        "feel repetitive. Render style: cinematic realism or a clean stylized-realistic "
        "look (deep blues/cyan/amber lighting is fine as a mood accent on top of a real "
        "recognizable subject, but the subject itself must read as a real, identifiable thing). "
        "NEVER include: legible body text, paragraphs, labeled boxes, arrows-with-words, "
        "charts, graphs, or UI mockups — these always render as illegible garbled marks. "
        "Represent technical/numeric ideas through the real-world subject's physical behavior instead "
        "(e.g. 'data accuracy improves' → gridlocked traffic smoothly starts flowing; "
        "'real-time monitoring' → a sensor's light pulses in sync with passing cars; "
        "'the system scales across the city' → camera pulls back from one intersection to "
        "reveal the same pattern repeating across many intersections). AT MOST one short coined "
        "label of 2-3 blocky monospace/code-style characters may appear as a stylistic accent."
    ),
    "cartoon": (
        "Visual approach: warm, friendly 2D cartoon/explainer-video style throughout — "
        "but still grounded in THIS paper's actual real-world domain. Simple character designs "
        "acting out one concrete idea per shot, bright clean color palette, smooth simple shapes. "
        "Each scene's subject must differ from its neighbors. NEVER include legible text, "
        "labeled diagrams, or charts — represent technical ideas through physical metaphor and character "
        "action grounded in the paper's real domain instead."
    ),
}

STORYBOARD_PROMPT = f"""
You are a creative director AND domain expert in this paper's exact field,
building a research paper explainer video of under 50 seconds.
Your job is to generate:
1. A single continuous narration script (`audio_script`) of EXACTLY 95-110 words.
   This script must be written to flow naturally as a single voiceover, covering:
   - Abstract & Problem (first ~9 seconds)
   - Methodology & Architecture (next ~9 seconds)
   - Experiments & Testing (next ~9 seconds)
   - Quantitative Results (next ~9 seconds)
   - Conclusion & Future Impact (final ~9 seconds)
   Use the paper's actual terminology, named systems, and numbers. Do not use generic filler.
   Tone: Expressive, professional, yet engaging and clear, with an emotional arc.
   CRITICAL: The script MUST be between 95 and 110 words. If you write more than 110 words, the video generation will fail!
2. A sequence of exactly 5 video clips (each representing a ~9-10 second segment of the timeline).
   For each clip, you must provide:
   - `associated_narration`: The exact sentence or sentences from the `audio_script` that correspond to this clip's time segment.
   - `video_prompt`: A cinematic, text-free visual and motion description for LTX-2.3.
     IMPORTANT: LTX-2.3 cannot render text or diagrams. Describe real-world scenes, objects, or conceptual/atmospheric metaphors (e.g. data streams, light patterns, mechanical/digital systems in motion, cityscapes, or server racks). Do not include any text, charts, or UI elements.
   - `audio_cue`: Ambient background music or sound effects (no speech) that match the clip's mood.
   - `camera_motion`: One of zoom_in, zoom_out, pan_left, pan_right, dolly_forward, static.

RESEARCH BRIEF (primary source of truth):
{json.dumps(research_understanding, indent=2)}

PAPER TITLE: {paper_content['title']}
METHODOLOGY EXCERPT: {paper_content['methodology'][:1200]}
ARCHITECTURE EXCERPT: {paper_content['architecture'][:1200]}
EXPERIMENTS EXCERPT: {paper_content['experiments'][:800]}
RESULTS EXCERPT: {paper_content['results'][:800]}

{VISUAL_STYLE_INSTRUCTIONS[VISUAL_STYLE]}

Return ONLY valid JSON, with no markdown fences, no preamble, matching this schema:

{{
  "audio_script": "The complete, continuous 95-110 word narration script...",
  "clips": [
    {{
      "clip_number": 1,
      "segment": "Abstract & Problem (0s-10s)",
      "associated_narration": "The exact sentence(s) from the audio_script spoken during this segment.",
      "video_prompt": "Cinematic visual description for LTX-2.3. Atmospheric conceptual or realistic metaphor aligned to the abstract/problem. No text, diagrams, or charts. Under 80 words.",
      "audio_cue": "Ambience/music cue (e.g., tense, low synth pad, digital hum)",
      "camera_motion": "dolly_forward"
    }},
    {{
      "clip_number": 2,
      "segment": "Methodology & Architecture (10s-20s)",
      "associated_narration": "The exact sentence(s) from the audio_script spoken during this segment.",
      "video_prompt": "Cinematic visual description for LTX-2.3 representing the methodology/architecture. Under 80 words.",
      "audio_cue": "Ambience/music cue (e.g., rhythmic clean electronic pulses, mid-tempo synth)",
      "camera_motion": "pan_right"
    }},
    {{
      "clip_number": 3,
      "segment": "Experiments & Testing (20s-30s)",
      "associated_narration": "The exact sentence(s) from the audio_script spoken during this segment.",
      "video_prompt": "Cinematic visual description for LTX-2.3 representing testing or data processing. Under 80 words.",
      "audio_cue": "Ambience/music cue (e.g., active high-tech data ambience)",
      "camera_motion": "zoom_in"
    }},
    {{
      "clip_number": 4,
      "segment": "Results (30s-40s)",
      "associated_narration": "The exact sentence(s) from the audio_script spoken during this segment.",
      "video_prompt": "Cinematic visual description for LTX-2.3 representing successful outcomes/results. Under 80 words.",
      "audio_cue": "Ambience/music cue (e.g., rising bright chords, triumphant synth)",
      "camera_motion": "zoom_out"
    }},
    {{
      "clip_number": 5,
      "segment": "Conclusion & Impact (40s-50s)",
      "associated_narration": "The exact sentence(s) from the audio_script spoken during this segment.",
      "video_prompt": "Cinematic visual description for LTX-2.3 representing the final conclusion and future vision. Under 80 words.",
      "audio_cue": "Ambience/music cue (e.g., warm ambient swell, soft strings resolving)",
      "camera_motion": "static"
    }}
  ]
}}
"""

print("\n Gemini Pass 2 — Storyboard generation...")
storyboard = extract_json(call_gemini(STORYBOARD_PROMPT))
assert isinstance(storyboard, dict), "Storyboard must be a JSON object containing audio_script and clips."
assert len(storyboard["clips"]) == CLIP_COUNT, f"Expected {CLIP_COUNT} clips, got {len(storyboard['clips'])}."
storyboard_path = os.path.join(FOLDERS["storyboard"], "storyboard.json")
with open(storyboard_path, "w") as f:
    json.dump(storyboard, f, indent=2)
print(f"Storyboard saved → {storyboard_path}")
print(f"   Audio Script: {storyboard['audio_script'][:120]}...")
for clip in storyboard["clips"]:
    n   = clip['clip_number']
    mot = clip['camera_motion']
    print(f"Clip {n} [{clip['segment']:25s}] [{mot}] prompt: {clip['video_prompt'][:60]}...")

def generate_kokoro_audio(text: str, output_path: str, voice: str = "af_bella") -> float:
    from kokoro import KPipeline
    import soundfile as sf
    import numpy as np
    print(f" Generating Kokoro TTS audio (voice={voice})...")
    pipeline = KPipeline(lang_code='a')
    generator = pipeline(text, voice=voice, speed=1.0)
    audio_data = []
    for gs, ps, audio in generator:
        if audio is not None and len(audio) > 0:
            audio_data.append(audio)
    if not audio_data:
        raise ValueError("Kokoro did not generate any audio.")
    full_audio = np.concatenate(audio_data)
    sf.write(output_path, full_audio, 24000)
    duration = len(full_audio) / 24000
    return duration


def generate_edgetts_audio(text: str, output_path: str, voice: str = "en-US-EmmaNeural") -> float:
    import edge_tts
    
    print(f" Generating edge-tts audio (voice={voice})...")
    async def amain():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path) 
    asyncio.run(amain())
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", output_path],
        capture_output=True, text=True, check=True
    )
    return float(probe.stdout.strip())
def generate_gtts_audio(text: str, output_path: str) -> float:
    from gtts import gTTS  
    print(" Generating gTTS audio (fallback)...")
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(output_path)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", output_path],
        capture_output=True, text=True, check=True
    )
    return float(probe.stdout.strip())
def generate_narration(text: str, output_path: str) -> float:
    try:
        return generate_kokoro_audio(text, output_path)
    except Exception as e:
        print(f"Kokoro TTS failed: {e}. Trying edge-tts...")
    try:
        return generate_edgetts_audio(text, output_path)
    except Exception as e:
        print(f"edge-tts failed: {e}. Trying gTTS...")
    return generate_gtts_audio(text, output_path)
narration_path = os.path.join(FOLDERS["narration"], "full_narration.wav")
audio_duration = generate_narration(storyboard["audio_script"], narration_path)
print(f"Narration audio generated. Duration: {audio_duration:.2f} seconds")



def seconds_to_valid_frame_count(seconds: float, fps: float) -> int:
    target_frames = seconds * fps
    n = math.ceil((target_frames - 1) / 8)
    n = max(n, 3)
    frames = int(n * 8 + 1)
    return frames
ideal_clip_duration = (audio_duration + (CLIP_COUNT - 1) * OVERLAP_SECONDS) / CLIP_COUNT
num_frames_per_clip = seconds_to_valid_frame_count(ideal_clip_duration, FRAME_RATE)
actual_clip_duration = num_frames_per_clip / FRAME_RATE
expected_video_duration = CLIP_COUNT * actual_clip_duration - (CLIP_COUNT - 1) * OVERLAP_SECONDS
print(f"\n Per-clip timing configuration:")
print(f"   Audio Duration         : {audio_duration:.2f}s")
print(f"   Ideal Clip Duration    : {ideal_clip_duration:.2f}s")
print(f"   Frame Count per Clip   : {num_frames_per_clip} frames ({(num_frames_per_clip - 1) // 8} × 8 + 1)")
print(f"   Actual Clip Duration   : {actual_clip_duration:.2f}s")
print(f"   Expected Video Duration: {expected_video_duration:.2f}s (difference of {expected_video_duration - audio_duration:+.2f}s)")
print("\n  Loading LTX-2.3 pipeline...")
pipe = LTX2Pipeline.from_pretrained(
    "diffusers/LTX-2.3-Diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing("max")

if hasattr(pipe, "vae") and pipe.vae is not None:
    try:
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.vae.tile_sample_min_size = 256
        print("VAE tiling + slicing enabled (reduces decode memory spikes)")
    except AttributeError:
        print("This VAE doesn't expose enable_tiling/enable_slicing")

AUDIO_SAMPLE_RATE = pipe.vocoder.config.output_sampling_rate
print(f"LTX-2.3 loaded with sequential CPU offload")
print(f"   Audio sample rate: {AUDIO_SAMPLE_RATE} Hz")

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

NEG_PROMPT = DEFAULT_NEGATIVE_PROMPT + (
    ", shaky camera, motion blur, jitter, flickering, "
    "poor lighting, inconsistent style, low resolution, "
    "text, words, letters, writing, subtitles, captions, text overlay, "
    "labels, labeled diagram, chart, graph, infographic, UI, dashboard, "
    "illegible text, garbled text, gibberish symbols, watermark, logo"
)

LTX_STYLE_SUFFIX = {
    "atmospheric_conceptual": (
        "cinematic motion graphics aesthetic, deep blues and cyan with warm amber "
        "accents, dramatic lighting on dark backgrounds, smooth purposeful camera "
        "movement, no photorealistic faces, no readable text, no diagrams, no charts, "
        "no UI elements — pure atmosphere and metaphor"
    ),
    "cartoon": (
        "2D cartoon explainer-video animation style, simple friendly character "
        "design, bright clean colors, smooth rounded shapes, no photorealism, "
        "no readable text, no diagrams, no charts, no UI elements"
    ),
}
def build_ltx_prompt(clip: dict) -> str:
    video_desc = clip["video_prompt"]
    audio_cue  = clip.get("audio_cue", "soft ambient background music, no speech")
    motion     = clip["camera_motion"].replace("_", " ")
    style      = LTX_STYLE_SUFFIX[VISUAL_STYLE]

    prompt = (
        f"{video_desc}. "
        f"Visual style: {style}. "
        f"Camera movement: {motion}. "
        f"Background audio: {audio_cue}, instrumental only, no dialogue, no voice, no spoken words. "
        f"Cinematic quality, sharp focus, professional lighting, smooth motion, high detail."
    )
    return prompt


clip_paths = []
SEEDS = [42, 123, 777, 1337, 2024]
print(f"\n Generating {CLIP_COUNT} clip videos with LTX-2.3...")
print(f"   Resolution : {VIDEO_WIDTH}×{VIDEO_HEIGHT}")
print(f"   Steps      : {NUM_INFER_STEPS}\n")
for i, clip in enumerate(storyboard["clips"]):
    n = clip["clip_number"]
    seed = SEEDS[i % len(SEEDS)]
    prompt = build_ltx_prompt(clip)
    print(f"[{n}/{CLIP_COUNT}] Generating ({num_frames_per_clip} frames, seed={seed})...")
    print(f"Prompt: {prompt[:120]}...")
    t0 = time.time()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    try:
        with torch.inference_mode():
            video, audio = pipe(
                prompt              = prompt,
                negative_prompt     = NEG_PROMPT,
                width               = VIDEO_WIDTH,
                height              = VIDEO_HEIGHT,
                num_frames          = num_frames_per_clip,
                frame_rate          = FRAME_RATE,
                num_inference_steps = NUM_INFER_STEPS,
                guidance_scale      = GUIDANCE_SCALE,
                generator           = generator,
                output_type         = "np",
                return_dict         = False,
            )
    except torch.OutOfMemoryError:
        print(f"OOM at {num_frames_per_clip} frames — retrying with reduced resolution...")
        del generator
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        reduced_frames = max(25, int(num_frames_per_clip * 0.8))
        reduced_frames = max(3, (reduced_frames - 1) // 8) * 8 + 1
        print(f"           Retrying with {reduced_frames} frames...") 
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        with torch.inference_mode():
            video, audio = pipe(
                prompt              = prompt,
                negative_prompt     = NEG_PROMPT,
                width               = VIDEO_WIDTH,
                height              = VIDEO_HEIGHT,
                num_frames          = reduced_frames,
                frame_rate          = FRAME_RATE,
                num_inference_steps = NUM_INFER_STEPS - 10,
                guidance_scale      = GUIDANCE_SCALE,
                generator           = generator,
                output_type         = "np",
                return_dict         = False,
            )

    clip_path = os.path.join(FOLDERS["scenes"], f"clip_{n}.mp4")
    encode_video(
        video             = video[0],
        fps               = FRAME_RATE,
        audio             = audio[0].float().cpu(),
        audio_sample_rate = AUDIO_SAMPLE_RATE,
        output_path       = clip_path,
    )
    clip_paths.append(clip_path)

    elapsed = time.time() - t0
    vram    = torch.cuda.memory_allocated() / 1024**3
    print(f"{elapsed:.1f}s → {clip_path}  (VRAM: {vram:.1f}GB)")

print("\nAll video clips generated.")



final_output_path = os.path.join(FOLDERS["output"], "final_reel.mp4")
print("\nCompiling clips and merging narration...")
def build_ffmpeg_merge_command(clip_paths: list, narration_path: str, output_path: str, clip_duration: float, overlap: float = 0.5) -> list:
    N = len(clip_paths)
    filter_parts = []
    current_v = "[0:v]"
    current_offset = clip_duration - overlap
    for i in range(1, N):
        next_v = f"[v{i}]" if i < N - 1 else "[vout]"
        filter_parts.append(f"{current_v}[{i}:v]xfade=transition=fade:duration={overlap}:offset={current_offset:.3f}{next_v}")
        current_v = next_v
        current_offset += clip_duration - overlap
    current_a = "[0:a]"
    for i in range(1, N):
        next_a = f"[a{i}]" if i < N - 1 else "[aamb]"
        filter_parts.append(f"{current_a}[{i}:a]acrossfade=d={overlap}:c1=tri:c2=tri{next_a}")
        current_a = next_a
    filter_parts.append(f"[aamb]volume={AMBIENCE_GAIN_DB}dB[amb]")
    filter_parts.append(f"[{N}:a]volume={NARRATION_GAIN_DB}dB[narr]")
    filter_parts.append(f"[amb][narr]amix=inputs=2:duration=first:dropout_transition=0[aout]")
    filter_complex = ";".join(filter_parts)
    
    cmd = ["ffmpeg", "-y"]
    for path in clip_paths:
        cmd.extend(["-i", path])
    cmd.extend(["-i", narration_path])
    cmd.extend(["-filter_complex", filter_complex])
    cmd.extend(["-map", "[vout]", "-map", "[aout]"])
    cmd.extend(["-shortest"])
    cmd.extend(["-c:v", "libx264", "-preset", "medium", "-b:v", "8000k"])
    cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    cmd.extend(["-r", str(int(FRAME_RATE))])
    cmd.extend([output_path])
    return cmd
ffmpeg_cmd = build_ffmpeg_merge_command(
    clip_paths     = clip_paths,
    narration_path = narration_path,
    output_path    = final_output_path,
    clip_duration  = actual_clip_duration,
    overlap        = OVERLAP_SECONDS
)
print("   Executing ffmpeg merge command:")
print("   " + " ".join(ffmpeg_cmd[:20]) + " ...")
res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
if res.returncode != 0:
    print(f"ffmpeg failed!\nError output:\n{res.stderr}")
    raise RuntimeError("ffmpeg merge failed.")
probe = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
     "-of", "default=noprint_wrappers=1:nokey=1", final_output_path],
    capture_output=True, text=True, check=True
)
final_duration = float(probe.stdout.strip())
file_mb = os.path.getsize(final_output_path) / (1024**2)

print(f"\nFINAL REEL: {final_output_path}")
print(f"   Duration  : {final_duration:.1f}s")
print(f"   File size : {file_mb:.1f} MB")
print(f"   Resolution: {VIDEO_WIDTH}×{VIDEO_HEIGHT} @ {int(FRAME_RATE)}fps")

print("GENERATED ASSETS")


groups = [
    ("Extracted JSON",  FOLDERS["extracted"],  "*.json"),
    ("Storyboard",      FOLDERS["storyboard"], "*.json"),
    ("Narration Audio", FOLDERS["narration"],  "*.wav"),
    ("Scene Videos",    FOLDERS["scenes"],     "*.mp4"),
    ("Final Output",    FOLDERS["output"],     "*.mp4"),
]

for label, folder, pattern in groups:
    files = sorted(Path(folder).glob(pattern))
    print(f"\n  {label}:")
    for fp in files:
        size = fp.stat().st_size
        unit = "KB" if size < 1_000_000 else "MB"
        val  = size // 1024 if size < 1_000_000 else size // (1024**2)
        print(f"    {fp.name:40s} {val:6d} {unit}")

print(f"\nDone! Open your video at:\n   {final_output_path}\n")
