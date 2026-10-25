<img width="942" height="551" alt="image" src="https://github.com/user-attachments/assets/10700ce0-88a3-483c-ae90-8a608a9a024b" />


PROJECT FREND - ENTIRE FLOW

STARTUP
main.py --web
  → stop any old Frend run + kill leftover speech players
  → erase last run's memory in Elasticsearch (unless --keep-memory)
  → start camera, NanoDet, speaker, Vosk mic, live web view
  → "Frend is ready. Say hey friend when you need me."

1) VISION LOOP (always on, on the Pi, ~8 FPS)
Webcam frame
  → Motion meter → camera swinging? → skip frame (no false alerts)
  → NanoDet-Plus → objects + position (left/ahead/right) + distance (far/close/very close)
      → Hazard announcer → new close object or getting closer? → alert text
      → Confident sightings → Elasticsearch memory (what, where, when)
  → Wall detector → plain surface filling the centre? → "Possible wall ahead"
  → Web view → annotated frame + status + log → browser at http://192.168.2.2:8080
Alerts → Speaker queue

2) VOICE LOOP (always on, on the Pi, offline)
Webcam mic → pw-record → Vosk speech recognition
  → Frend talking? → only "stop" accepted (ignores its own voice)
  → "Hey Frend" → hold alerts → "Yes?" → wait up to 15 s for the command
  → Low confidence / misheard → "Sorry, please say it again."
  → Intent parser → command type → release alerts

3) COMMAND ROUTING
"stop"                         → silence speech + clear queue
"alerts off / on"              → pause / resume hazard alerts
"examine surroundings"         → frame + detections → Claude → 2-sentence description
"check obstacles in my path"   → frame + detections → Claude → obstacles within 30 ft
"where is my water bottle"     → Elasticsearch search → memories + what's seen now → Claude → answer
"detailed scan"                → frame → Runpod Serverless GPU (Qwen2.5-VL-7B) → layout + all text
"watch my surroundings"        → periodic frames → Claude → updates on changes
any other question             → frame + detections → Claude → answer
Every command + answer         → Elasticsearch memory log
Answers                        → Speaker queue

4) SPEECH OUTPUT
Speaker queue (priority order, stale alerts dropped, only "stop" interrupts)
  → sentence by sentence
  → Cartesia Sonic TTS ("Parker" voice, streaming WebSocket, cached phrases)
      ↘ fallback: gTTS → fallback: offline espeak
  → pw-cat → PipeWire → network audio tunnel → MacBook PulseAudio → MacBook speakers → user

SHUTDOWN (Ctrl+C / closed SSH / kill)
  → speech player killed instantly → nothing carries over to the next run
