"""
voice.py - Offline "Hey Frend" voice commands via the webcam's microphone.

Uses Vosk (Kaldi) with its small 40 MB English model: it runs fully on the
Pi, streams audio in real time using a fraction of one CPU core, and needs no
internet or wake-word service.

Audio comes from PipeWire through `pw-record`, which resamples the C270's
48 kHz mic to the 16 kHz Vosk needs. Only the configured mic is used
(config.MIC_NAME): if it is unplugged, voice commands pause and resume when
it is back. (The AirPods mic isn't used: opening it would drop AirPods
playback to phone-call quality, and the Pi 5's built-in Bluetooth chip
doesn't deliver headset-mic audio anyway.)

This module only finds the wake phrase and hands over the words after it;
intents.py decides what they mean. The wake word can also be said alone:
    "Hey Frend, keep checking my surroundings"       (one breath)
    "Hey Frend" ... "Yes?" ... "is the door open?"   (two steps)

Speech recognisers hear "Frend" as "friend", so the wake phrase matches
"hey/hi/okay friend" or "friend" at the start of an utterance.

While Frend is talking (and for a second after), the mic also hears Frend's
own voice, e.g. its "Yes?" comes back as "yeah". So during that time the
only thing accepted is "stop" (with or without the wake word), and filler
words like "yeah" or "uh" never count as a question.
"""

import json
import logging
import re
import select
import subprocess
import threading
import time

import config

log = logging.getLogger("frend.voice")

# "hey friend ...", "hi friends ...", "okay frend ...", or "friend ..." at the start.
WAKE_RE = re.compile(r"(?:^|\b(?:hey|hay|hi|hello|okay|ok)\s+)(?:friend|friends|frend)\b\s*")

# Accepted mid-speech to cut Frend off. Short utterances only, so a long echo
# of Frend's own sentence that happens to contain "stop" doesn't count.
STOP_RE = re.compile(r"\b(?:stop|quiet|shut up|silence|enough)\b")
STOP_MAX_WORDS = 4
# What Vosk makes of background noise, coughs and our own "Yes?" echo.
FILLER_WORDS = {"yeah", "yes", "yep", "yup", "uh", "um", "uhm", "huh", "hmm", "mm", "ah",
                "oh", "the", "a", "an", "and", "so", "ok", "okay", "right", "he", "she",
                "it", "i", "said", "what", "well", "hey", "hi", "but", "is", "of", "to"}

# Below this average word confidence (0-1, from Vosk), the command is treated
# as misheard and the user is asked to repeat it.
MIN_COMMAND_CONFIDENCE = 0.65

RECOGNIZER_RATE = 16000   # Vosk small models are trained on 16 kHz audio
CHUNK_BYTES = RECOGNIZER_RATE // 4 * 2   # 0.25 s of 16-bit mono
# How often to look for the mic while it is unplugged.
RECONNECT_POLL_SECONDS = 3.0
# No audio for this long means the mic went away (pw-record doesn't exit).
STALL_SECONDS = 3.0
# Without these, PipeWire would quietly record from some other default mic
# when ours is missing or unplugged.
STREAM_PROPS = "{ node.dont-fallback = true node.dont-reconnect = true node.dont-move = true }"


class VoiceCommandListener:
    """Background thread: webcam mic -> Vosk -> `on_command(kind, text)`.

    kind is "wake" (wake word alone; a follow-up is expected) or "command"
    (text = the words spoken after the wake word).
    """

    def __init__(self, on_command, model_dir=config.VOSK_MODEL_DIR,
                 frend_is_talking=lambda: False):
        if not model_dir.exists():
            raise FileNotFoundError(f"Vosk model not found at {model_dir}. Run ./setup_models.sh")
        self.on_command = on_command
        self.frend_is_talking = frend_is_talking
        self.model_dir = model_dir
        self._stop = threading.Event()
        self._thread = None
        self._proc = None
        self._awaiting_question_until = 0.0   # > now after a bare "Hey Frend"

    def start(self):
        self._thread = threading.Thread(target=self._run, name="voice", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        if self._thread:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------ #
    def _run(self):
        import vosk

        vosk.SetLogLevel(-1)                      # silence Kaldi's chatter
        recognizer = vosk.KaldiRecognizer(vosk.Model(str(self.model_dir)), RECOGNIZER_RATE)
        recognizer.SetWords(True)                 # per-word confidence scores
        waiting_logged = False
        while not self._stop.is_set():
            mic = _find_mic()
            if mic is None:
                if not waiting_logged:
                    log.warning("Mic '%s' not found; voice commands paused until it is "
                                "plugged in", config.MIC_NAME)
                    waiting_logged = True
                self._stop.wait(RECONNECT_POLL_SECONDS)
                continue
            waiting_logged = False
            self._listen(*mic, recognizer)
            recognizer.Reset()
            self._stop.wait(1.0)

    def _listen(self, node, description, recognizer):
        """Feed the mic to Vosk until it is unplugged."""
        log.info("Listening on %s", description)
        self._proc = subprocess.Popen(
            ["pw-record", "--target", node, "-P", STREAM_PROPS,
             "--rate", str(RECOGNIZER_RATE), "--channels", "1", "--format", "s16",
             "--raw", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out = self._proc.stdout
        last_data = time.monotonic()
        try:
            while not self._stop.is_set():
                ready, _, _ = select.select([out], [], [], 1.0)
                if not ready:
                    if time.monotonic() - last_data > STALL_SECONDS and _find_mic() is None:
                        log.info("Mic unplugged")
                        return
                    continue
                chunk = out.read1(CHUNK_BYTES)
                if not chunk:
                    log.info("Mic stream ended")
                    return
                last_data = time.monotonic()
                if recognizer.AcceptWaveform(chunk):
                    result = json.loads(recognizer.Result())
                    if result.get("text"):
                        confs = [w.get("conf", 1.0) for w in result.get("result", [])]
                        self._handle(result["text"], confs)
        finally:
            if self._proc.poll() is None:
                self._proc.terminate()
            self._proc.wait()

    # ------------------------------------------------------------------ #
    def expect_follow_up(self):
        """Accept the next utterance without the wake word (after "say that again")."""
        self._awaiting_question_until = time.monotonic() + config.FOLLOW_UP_SECONDS

    def _handle(self, text, confs=()):
        """Interpret one finished utterance. confs: Vosk's per-word confidences."""
        text = text.lower().strip()
        log.debug("Heard: %s %s", text, [round(c, 2) for c in confs])

        if self.frend_is_talking():
            if len(text.split()) <= STOP_MAX_WORDS and STOP_RE.search(text):
                log.info("Voice command: 'stop' (while speaking)")
                self.on_command("command", "stop")
            return                                 # probably Frend's own voice

        match = WAKE_RE.search(text)
        if match:
            command = text[match.end():].strip()
            if _is_filler(command):
                command = ""                       # "hey friend uh" = wake word only
        elif time.monotonic() < self._awaiting_question_until:
            if _is_filler(text):
                return                             # noise; keep waiting for the question
            command = text                         # follow-up to a bare wake word
        else:
            return                                 # not talking to us
        self._awaiting_question_until = 0.0

        log.info("Voice command: %r", command or "(wake word only)")
        if not command:
            self.expect_follow_up()
            self.on_command("wake", "")
            return
        # Confidence of just the command words (the last ones in the utterance).
        n = len(command.split())
        command_confs = list(confs)[-n:] if len(confs) >= n else list(confs)
        if command_confs and sum(command_confs) / len(command_confs) < MIN_COMMAND_CONFIDENCE:
            log.info("Not sure I heard %r correctly (confidence %.2f)",
                     command, sum(command_confs) / len(command_confs))
            self.on_command("unclear", command)
        else:
            self.on_command("command", command)


def _is_filler(text):
    return all(w in FILLER_WORDS for w in text.split())


def _find_mic():
    """Return (node name, description) of the configured mic, or None."""
    try:
        nodes = json.loads(subprocess.run(["pw-dump"], capture_output=True, text=True,
                                          timeout=5).stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        log.warning("Could not list PipeWire devices: %s", exc)
        return None
    for obj in nodes:
        props = (obj.get("info") or {}).get("props") or {}
        name = props.get("node.name", "")
        if (props.get("media.class") == "Audio/Source"
                and config.MIC_NAME.lower() in name.lower()):
            return name, f"mic '{props.get('node.description') or name}'"
    return None


# Quick manual test:  python voice.py   then say "Hey Frend, what do you see?"
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    listener = VoiceCommandListener(lambda kind, text: print(f">>> {kind}: {text}"))
    listener.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        listener.stop()
