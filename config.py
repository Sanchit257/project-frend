"""
config.py - Central place for every tunable setting in Project Frend.

Keeping all "magic numbers" here means you can tune the device (camera
resolution, alert sensitivity, voice, Claude model, ...) without touching the
logic in the other modules. Any value can also be overridden with an
environment variable or a line in the `.env` file next to this script.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / ".cache"          # gTTS mp3 cache lives here

# Load secrets (ANTHROPIC_API_KEY) and overrides from ./.env if it exists.
load_dotenv(BASE_DIR / ".env")


def _env(name, default, cast=str):
    """Read an environment variable and convert it to the type of `default`."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return cast(raw)


# --------------------------------------------------------------------------- #
# Camera (vision.py)
# --------------------------------------------------------------------------- #
CAMERA_INDEX = _env("FREND_CAMERA_INDEX", 0, int)
# 640x480 is the sweet spot for the Logitech C270: the detector downsizes to
# 416x416 anyway, and larger frames only cost USB bandwidth and CPU.
CAMERA_WIDTH = _env("FREND_CAMERA_WIDTH", 640, int)
CAMERA_HEIGHT = _env("FREND_CAMERA_HEIGHT", 480, int)
CAMERA_FPS = _env("FREND_CAMERA_FPS", 30, int)
# "YUYV" (raw) or "MJPG" (compressed; use it for resolutions above 640x480).
CAMERA_FOURCC = _env("FREND_CAMERA_FOURCC", "YUYV")

# --------------------------------------------------------------------------- #
# Local hazard detector (detector.py)
# --------------------------------------------------------------------------- #
DETECTOR_MODEL = MODELS_DIR / "object_detection_nanodet_2022nov.onnx"
DETECTOR_CONFIDENCE = _env("FREND_DETECTOR_CONFIDENCE", 0.45, float)
DETECTOR_NMS_IOU = 0.6
# Max detector passes per second. The Pi can do more, but a wearable runs on a
# battery and heats up; ~8 FPS is plenty for walking-speed hazard alerts.
DETECTOR_MAX_FPS = _env("FREND_DETECTOR_MAX_FPS", 8.0, float)
# OpenCV worker threads. Leave one core free for audio + speech recognition.
OPENCV_THREADS = _env("FREND_OPENCV_THREADS", 3, int)

# An object must be seen in this many consecutive detector passes before it is
# announced. Filters out single-frame false positives ("flicker").
HAZARD_CONFIRM_FRAMES = 2
# Gap in sightings (s) still counted as "the same continuous sighting", so one
# missed frame doesn't restart the confirmation count.
HAZARD_DROPOUT_SECONDS = 0.6
# An object unseen for this long is forgotten; if it comes back, it is
# announced again as new.
HAZARD_FORGET_SECONDS = _env("FREND_HAZARD_FORGET_SECONDS", 3.0, float)
# Re-announce an already-announced object as "getting closer" once its box has
# grown by this factor (it is approaching without changing distance level).
HAZARD_APPROACH_GROWTH = 1.6
# Minimum confidence for an alert (memory uses detector.TRUSTED_CONFIDENCE).
HAZARD_MIN_CONFIDENCE = _env("FREND_HAZARD_MIN_CONFIDENCE", 0.50, float)
# Minimum gap between any two hazard announcements, so alerts don't pile up.
HAZARD_MIN_GAP_SECONDS = 1.5
# A "very close" alert jumps the speech queue, so it needs more evidence:
# seen this many passes in a row (~0.5 s at 8 FPS) with at least this
# confidence. Weaker very-close sightings are announced as plain "close".
HAZARD_URGENT_CONFIRM_FRAMES = 4
HAZARD_URGENT_MIN_CONFIDENCE = 0.55
# Mean pixel change between processed frames above which the camera counts as
# swinging, and alerts pause until it steadies. A still camera scores ~0.5.
# Run with --debug to see live scores if you need to tune it.
CAMERA_MOTION_THRESHOLD = _env("FREND_CAMERA_MOTION_THRESHOLD", 12.0, float)

# Camera-only wall guess (vision.WallDetector): the middle of the view must be
# plain and edge-free for WALL_CONFIRM_FRAMES still frames in a row.
WALL_ALERTS = _env("FREND_WALL_ALERTS", True, bool)
WALL_MAX_EDGE_DENSITY = _env("FREND_WALL_MAX_EDGE_DENSITY", 0.02, float)  # fraction of edge pixels
WALL_MAX_TEXTURE = _env("FREND_WALL_MAX_TEXTURE", 14.0, float)            # grayscale std-dev
WALL_MAX_TONE_GAP = _env("FREND_WALL_MAX_TONE_GAP", 20.0, float)          # centre vs upper brightness
WALL_CONFIRM_FRAMES = _env("FREND_WALL_CONFIRM_FRAMES", 6, int)
WALL_COOLDOWN_SECONDS = _env("FREND_WALL_COOLDOWN_SECONDS", 8.0, float)

# --------------------------------------------------------------------------- #
# Text-to-speech (audio.py)
# --------------------------------------------------------------------------- #
# "auto"     -> Cartesia if a key is set, else gTTS; offline voice as last resort
# "cartesia" -> Cartesia Sonic (falls back to gTTS, then pyttsx3, on failure)
# "gtts"     -> Google voice (falls back to pyttsx3)
# "pyttsx3"  -> always offline voice (robotic; needs `sudo apt install espeak-ng`)
TTS_ENGINE = _env("FREND_TTS_ENGINE", "auto").lower()

# Cartesia Sonic: https://docs.cartesia.ai
CARTESIA_API_KEY = _env("CARTESIA_API_KEY", "")
# Sent on every request (the SDK's default is overridden so the date is pinned here).
CARTESIA_VERSION = "2026-08-14"
# "sonic-3.6" always tracks the latest stable Sonic 3.6 snapshot.
CARTESIA_MODEL_ID = _env("CARTESIA_MODEL_ID", "sonic-3.6")
# Skylar: approachable en-US female, recommended by Cartesia for assistants.
CARTESIA_VOICE_ID = _env("CARTESIA_VOICE_ID", "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4")
CARTESIA_LANGUAGE = "en"
# generation_config.speed, 0.6 - 1.5. Slightly fast suits screen-reader users.
CARTESIA_SPEED = _env("FREND_CARTESIA_SPEED", 1.1, float)
# Raw 16-bit mono PCM: no decoding on the Pi, and 24 kHz is plenty for speech.
CARTESIA_SAMPLE_RATE = 24000
# Phrases up to this length are cached on disk (hazard alerts repeat a lot),
# so they replay instantly, cost no credits, and work offline.
TTS_CACHE_MAX_CHARS = 80
GTTS_LANG = _env("FREND_GTTS_LANG", "en")
GTTS_TLD = _env("FREND_GTTS_TLD", "com")          # accent: com, co.uk, com.au...
GTTS_TIMEOUT_SECONDS = 4.0
# gTTS talks slowly; 1.25 is still very clear. Range 0.5 - 2.0.
TTS_SPEED = _env("FREND_TTS_SPEED", 1.25, float)
PYTTSX3_RATE = _env("FREND_PYTTSX3_RATE", 175, int)  # words per minute
# Hazard messages older than this are dropped instead of spoken late.
STALE_HAZARD_SECONDS = 2.5

# --------------------------------------------------------------------------- #
# Claude scene description (scene.py)
# --------------------------------------------------------------------------- #
ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY", "")
# Haiku is the fastest Claude model, which matters for a spoken assistant.
# Switch to e.g. "claude-sonnet-5-5" for more detailed descriptions.
CLAUDE_MODEL = _env("FREND_CLAUDE_MODEL", "claude-haiku-5-5")
CLAUDE_MAX_TOKENS = 500
CLAUDE_TIMEOUT_SECONDS = 25.0
# Frames are shrunk before upload: faster over Wi-Fi/hotspot, cheaper, and
# Claude does not need more than this to describe a scene.
CLAUDE_IMAGE_MAX_WIDTH = 768
CLAUDE_JPEG_QUALITY = 75

# --------------------------------------------------------------------------- #
# "Keep checking my surroundings" (watch.py)
# --------------------------------------------------------------------------- #
# Seconds between Claude safety checks. Each check costs one small API call
# (~800 input tokens with Haiku), so 10 s is ~360 calls per hour.
WATCH_INTERVAL_SECONDS = _env("FREND_WATCH_INTERVAL_SECONDS", 10.0, float)
# Clear-path estimates above this are spoken as "more than N feet"; a single
# eye-level camera can't judge longer distances reliably.
WATCH_MAX_CLEAR_FEET = 40

# --------------------------------------------------------------------------- #
# Detailed scan on a cloud GPU (vision.py -> Runpod Serverless)
# --------------------------------------------------------------------------- #
RUNPOD_API_KEY = _env("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = _env("RUNPOD_ENDPOINT_ID", "")
# Must match MODEL_NAME on the Runpod vLLM endpoint.
RUNPOD_MODEL = _env("FREND_RUNPOD_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
# Scans send a sharper image than Claude checks, for reading small text.
RUNPOD_IMAGE_MAX_WIDTH = 1280
# A scaled-to-zero endpoint must boot a GPU and load the model first, which
# can take a few minutes; warm requests take a few seconds.
RUNPOD_TIMEOUT_SECONDS = 300.0
RUNPOD_MAX_TOKENS = 400

# --------------------------------------------------------------------------- #
# Visual memory (memory.py -> Elasticsearch)
# --------------------------------------------------------------------------- #
ELASTIC_CLOUD_ID = _env("ELASTIC_CLOUD_ID", "")
ELASTIC_URL = _env("ELASTIC_URL", "")          # alternative to the Cloud ID
ELASTIC_API_KEY = _env("ELASTIC_API_KEY", "")
ELASTIC_INDEX = _env("FREND_ELASTIC_INDEX", "frend-memory")
# Events wait in RAM while offline; the oldest are dropped past this many.
MEMORY_MAX_BUFFERED = 2000
MEMORY_RECALL_RESULTS = 8
# Start every run with an empty memory (old runs are erased). Set
# FREND_MEMORY_CLEAR_ON_START=0 in .env, or pass --keep-memory, to keep history.
MEMORY_CLEAR_ON_START = _env("FREND_MEMORY_CLEAR_ON_START", "1") not in ("0", "false", "no")

# --------------------------------------------------------------------------- #
# Voice commands (voice.py)
# --------------------------------------------------------------------------- #
VOSK_MODEL_DIR = MODELS_DIR / "vosk-model-small-en-us-0.15"
# Substring of the mic's PipeWire node name (list them with `pw-cli ls Node`).
# The Logitech C270 is USB 046d:0825.
MIC_NAME = _env("FREND_MIC_NAME", "046d_0825")
# Seconds to wait for a question after a bare "Hey Frend".
FOLLOW_UP_SECONDS = 15.0
