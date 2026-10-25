"""
audio.py - Non-blocking text-to-speech that plays through the AirPods.

Design
------
* Every module just calls `speaker.say("text")` and returns immediately.
  A single background thread owns the TTS engines and the audio output, so
  speech never stalls the camera or detector loops (and neither the Cartesia
  WebSocket nor pyttsx3, which are not thread-safe, is shared).

* Engines, tried in order until one works:
    1. Cartesia Sonic : hyper-realistic streaming voice over one long-lived
                        WebSocket. Audio starts ~80 ms after the request and
                        plays while it is still being generated.
    2. gTTS           : Google voice, if Cartesia is unreachable.
    3. pyttsx3        : offline espeak-ng voice, so alerts never go silent.
  Short phrases (hazard alerts repeat a lot) are cached on disk, so they
  replay instantly, cost no credits, and still work offline.

* Audio routing: playback goes to the *default PipeWire output* through
  `pw-cat` (raw PCM from Cartesia) or `ffplay` (gTTS mp3). When the AirPods
  are connected over Bluetooth they are that default, so no device names are
  hard-coded. (`wpctl status` marks the default sink with `*`.) PortAudio
  based players such as pyaudio only see the Pi's raw ALSA devices, not the
  Bluetooth sink, which is why they are not used here.

* Priorities: hazard alerts jump the queue and stale alerts are dropped (an
  alert about a car 5 seconds ago is worse than useless). Nothing cuts off
  what is currently being said except the user saying "stop"
  (`stop_speaking()`); alerts wait until the current message has finished.
"""

import ctypes
import hashlib
import logging
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field

import config

_PR_SET_PDEATHSIG = 1
try:
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
except OSError:
    _libc = None


def _die_with_parent():
    """Runs in the player child before exec: the kernel SIGKILLs it as soon as
    Frend exits, however Frend ends (Ctrl+C, closed SSH session, kill -9), so
    a cut-off sentence can never keep playing."""
    if _libc is not None:
        _libc.prctl(_PR_SET_PDEATHSIG, 9)


def _popen_player(cmd, stdin):
    return subprocess.Popen(cmd, stdin=stdin, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, preexec_fn=_die_with_parent)


def kill_leftover_players():
    """Kill speech players orphaned by an earlier run (they are the only
    pw-cat processes started with the "Assistant" media role)."""
    try:
        out = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", "pw-cat.*--media-role Assistant"],
                             capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return
    for pid in out.split():
        if int(pid) != os.getpid():
            try:
                os.kill(int(pid), 9)
                log.info("Killed leftover speech player from a previous run (pid %s)", pid)
            except OSError:
                pass

log = logging.getLogger("frend.audio")

# Lower number = spoken first.
PRIORITY_URGENT = 0     # very close hazard: spoken next
PRIORITY_HAZARD = 1     # other hazard alerts
PRIORITY_NORMAL = 2     # answers, descriptions, status messages

# After a Cartesia failure, use the fallback voices for this long before retrying.
CARTESIA_RETRY_SECONDS = 30.0


@dataclass
class _Utterance:
    text: str
    priority: int
    created: float = field(default_factory=time.monotonic)
    is_hazard: bool = False
    ttl: float = None       # seconds; dropped if it can't start in time

    @property
    def stale(self):
        return self.ttl is not None and time.monotonic() - self.created > self.ttl


class Speaker:
    """Queue-based TTS worker. Thread-safe: call `say()` from anywhere."""

    def __init__(self, engine=config.TTS_ENGINE):
        if engine == "auto":
            engine = "cartesia" if config.CARTESIA_API_KEY else "gtts"
        self.engine = engine
        self.is_speaking = threading.Event()

        self._queue = []                       # list of (priority, seq, _Utterance)
        self._seq = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._player = None                    # current pw-cat / ffplay subprocess
        self._interrupted = False
        self._cut_off = False
        self._last_spoke = 0.0                 # when speech last finished
        self._alerts_held_until = 0.0          # see hold_alerts()
        self._thread = None

        self._cartesia = CartesiaStream() if engine == "cartesia" else None
        self._cartesia_down_until = 0.0
        self._pyttsx3 = None                   # created lazily in worker thread
        self._pyttsx3_failed = False
        self._online = None
        self._online_checked = 0.0
        self._ffplay = shutil.which("ffplay")
        self._pwcat = shutil.which("pw-cat")
        self._cache_dir = config.CACHE_DIR / "tts"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        log.info("TTS engine: %s", engine)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def start(self):
        self._thread = threading.Thread(target=self._worker, name="speaker", daemon=True)
        self._thread.start()
        return self

    def say(self, text, priority=PRIORITY_NORMAL, is_hazard=False, ttl=None):
        """Queue `text` for speaking. Returns immediately.

        ttl: if given, the message is skipped when it can't start within that
        many seconds (for time-sensitive messages like "Clear").
        """
        text = (text or "").strip()
        if not text:
            return
        if (is_hazard or ttl is not None) and time.monotonic() < self._alerts_held_until:
            log.info("Alert held (listening to the user): %s", text)
            return
        log.info("SAY%s: %s", " (hazard)" if is_hazard else "", text)
        with self._cond:
            if is_hazard:
                # Only the newest hazard matters; drop older pending ones.
                self._queue = [q for q in self._queue if not q[2].is_hazard]
            self._push(_Utterance(text, priority, is_hazard=is_hazard, ttl=ttl))
            self._cond.notify()

    def say_hazard(self, text, urgent=False):
        self.say(text, PRIORITY_URGENT if urgent else PRIORITY_HAZARD, is_hazard=True,
                 ttl=config.STALE_HAZARD_SECONDS)

    def hold_alerts(self, seconds):
        """Mute time-sensitive messages (hazard alerts, watch updates) for a while,
        e.g. while waiting for the user's question. Queued ones are dropped."""
        self._alerts_held_until = time.monotonic() + seconds
        with self._cond:
            self._queue = [q for q in self._queue
                           if not (q[2].is_hazard or q[2].ttl is not None)]

    def release_alerts(self):
        self._alerts_held_until = 0.0

    def stop_speaking(self):
        """Silence everything now and clear the queue (the 'stop' command)."""
        with self._cond:
            self._queue.clear()
            self._interrupt_current()

    def shutdown(self):
        self._stop.set()
        self.stop_speaking()
        player = self._player
        if player and player.poll() is None:
            player.kill()             # terminate() lets pw-cat drain its buffer first
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cartesia:
            self._cartesia.close()

    # ------------------------------------------------------------------ #
    # Queue helpers (call with self._cond held)
    # ------------------------------------------------------------------ #
    def _push(self, utt):
        self._seq += 1
        self._queue.append((utt.priority, self._seq, utt))

    def _pop(self):
        best = min(self._queue, key=lambda q: q[:2])
        self._queue.remove(best)
        return best[2]

    def _interrupt_current(self):
        self._interrupted = True
        if self._player and self._player.poll() is None:
            self._cut_off = True          # sentence was not finished
            self._player.terminate()

    def spoke_recently(self, tail=1.0):
        """True while speaking and for `tail` seconds after, i.e. while the
        mic may still be hearing Frend's own voice."""
        return self.is_speaking.is_set() or time.monotonic() - self._last_spoke < tail

    # ------------------------------------------------------------------ #
    # Worker thread
    # ------------------------------------------------------------------ #
    def _worker(self):
        while not self._stop.is_set():
            with self._cond:
                self._cond.wait_for(lambda: self._queue or self._stop.is_set())
                if self._stop.is_set():
                    return
                utt = self._pop()
                self._interrupted = False
                self._cut_off = False

            if utt.stale:
                log.debug("Dropping stale message: %s", utt.text)
                continue

            self.is_speaking.set()
            try:
                self._speak_sentences(utt)
            except Exception:
                log.exception("Speech failed for %r", utt.text)
            finally:
                self._last_spoke = time.monotonic()
                with self._cond:
                    if not self._queue:
                        self.is_speaking.clear()
        self.is_speaking.clear()

    def _speak_sentences(self, utt):
        """Speak text one sentence at a time, so "stop" ends it promptly."""
        for sentence in _split_sentences(utt.text):
            self._speak_one(sentence)
            if self._interrupted:
                return               # stop_speaking(): drop the rest

    def _speak_one(self, text):
        """Speak one sentence with the best engine that works right now."""
        if self._cartesia and self._pwcat and time.monotonic() >= self._cartesia_down_until:
            try:
                self._speak_cartesia(text)
                return
            except Exception as exc:
                if self._interrupted:
                    return
                log.warning("Cartesia failed (%s); using fallback voice for %.0f s",
                            exc, CARTESIA_RETRY_SECONDS)
                self._cartesia_down_until = time.monotonic() + CARTESIA_RETRY_SECONDS
                self._cartesia.close()

        if self.engine != "pyttsx3" and self._gtts_reachable():
            mp3 = self._gtts_file(text)
            if mp3:
                self._play([self._ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet",
                            "-af", f"atempo={config.TTS_SPEED}", str(mp3)])
                return
        self._speak_pyttsx3(text)

    # ------------------------------------------------------------------ #
    # Cartesia (streaming PCM -> pw-cat -> default PipeWire sink)
    # ------------------------------------------------------------------ #
    def _speak_cartesia(self, text):
        cache = self._cache_path("cartesia", text, ".pcm")
        if cache and cache.exists():
            # pw-cat --raw only reads raw PCM from stdin, not from a path.
            with open(cache, "rb") as pcm:
                self._play(self._pwcat_cmd() + ["-"], stdin=pcm)
            return

        with self._cond:
            if self._interrupted:
                self._cut_off = True
                return
            self._player = _popen_player(self._pwcat_cmd() + ["-"], subprocess.PIPE)
        player = self._player
        audio = bytearray()
        try:
            for chunk in self._cartesia.stream(text, should_stop=lambda: self._interrupted):
                audio += chunk
                player.stdin.write(chunk)
        except BrokenPipeError:
            pass                       # player killed by an interrupt
        finally:
            try:
                player.stdin.close()
            except BrokenPipeError:
                pass
            player.wait()
            self._player = None

        if cache and not self._interrupted and audio:
            tmp = cache.with_suffix(".part")
            tmp.write_bytes(bytes(audio))
            tmp.replace(cache)

    @staticmethod
    def _pwcat_cmd():
        # --media-role lets PipeWire treat this as assistant speech (e.g. for
        # ducking music); "-" reads raw PCM from stdin instead of a file.
        return ["pw-cat", "--playback", "--raw", "--format", "s16",
                "--rate", str(config.CARTESIA_SAMPLE_RATE), "--channels", "1",
                "--media-role", "Assistant"]

    def _cache_path(self, engine, text, suffix):
        if len(text) > config.TTS_CACHE_MAX_CHARS:
            return None
        key = "|".join([engine, config.CARTESIA_MODEL_ID, config.CARTESIA_VOICE_ID,
                        str(config.CARTESIA_SPEED), text])
        return self._cache_dir / f"{hashlib.sha1(key.encode()).hexdigest()}{suffix}"

    # ------------------------------------------------------------------ #
    # gTTS (Google voice, fallback)
    # ------------------------------------------------------------------ #
    def _gtts_reachable(self):
        if self._ffplay is None:
            return False
        now = time.monotonic()
        if self._online is None or now - self._online_checked > 30:
            self._online_checked = now
            try:
                socket.create_connection(("translate.google.com", 443), timeout=1.5).close()
                self._online = True
            except OSError:
                self._online = False
        return self._online

    def _gtts_file(self, text):
        """Return a path to an mp3 of `text` (cached), or None on failure."""
        from gtts import gTTS   # imported lazily: unused when Cartesia works

        key = hashlib.sha1(f"{config.GTTS_LANG}|{config.GTTS_TLD}|{text}".encode()).hexdigest()
        path = self._cache_dir / f"{key}.mp3"
        if path.exists():
            return path
        try:
            tmp = path.with_suffix(".part")
            gTTS(text, lang=config.GTTS_LANG, tld=config.GTTS_TLD,
                 timeout=config.GTTS_TIMEOUT_SECONDS).save(str(tmp))
            tmp.replace(path)
            return path
        except Exception as exc:
            log.warning("gTTS failed (%s); using offline voice", exc)
            self._online = False
            self._online_checked = time.monotonic()
            return None

    def _play(self, cmd, stdin=subprocess.DEVNULL):
        """Run a player to completion unless interrupted."""
        with self._cond:
            if self._interrupted:
                self._cut_off = True
                return
            self._player = _popen_player(cmd, stdin)
        self._player.wait()
        self._player = None

    # ------------------------------------------------------------------ #
    # pyttsx3 (offline, espeak-ng)
    # ------------------------------------------------------------------ #
    def _speak_pyttsx3(self, text):
        if self._pyttsx3 is None and not self._pyttsx3_failed:
            try:
                import pyttsx3
                self._pyttsx3 = pyttsx3.init()
                self._pyttsx3.setProperty("rate", config.PYTTSX3_RATE)
            except Exception as exc:
                self._pyttsx3_failed = True
                log.error("pyttsx3 unavailable (%s). Install it with: "
                          "sudo apt install espeak-ng", exc)
        if self._pyttsx3 is None:
            print(f"[FREND would say] {text}", flush=True)
            return
        self._pyttsx3.say(text)
        self._pyttsx3.runAndWait()


class CartesiaError(Exception):
    pass


class CartesiaStream:
    """One long-lived Cartesia TTS WebSocket (lowest latency across many turns).

    Used only from the speaker thread. Each utterance gets its own context on
    the shared connection; the SDK reconnects automatically if the server has
    closed an idle socket.
    """

    def __init__(self):
        self._client = None
        self._conn = None

    def _connect(self):
        if self._conn is None:
            import cartesia
            if self._client is None:
                # The SDK sends a default cartesia-version header; overriding the
                # same (lower-case) key pins it instead of sending a duplicate.
                self._client = cartesia.Cartesia(
                    api_key=config.CARTESIA_API_KEY,
                    default_headers={"cartesia-version": config.CARTESIA_VERSION})
            self._conn = self._client.tts.websocket_connect().__enter__()
        return self._conn

    def stream(self, text, should_stop=lambda: False):
        """Yield raw PCM chunks for `text` as they are generated."""
        for attempt in range(2):
            started = False
            try:
                for chunk in self._stream_once(text, should_stop):
                    started = True
                    yield chunk
                return
            except CartesiaError:
                raise                          # API said no: retrying won't help
            except Exception as exc:
                self.close()
                # Reconnect once, but only if nothing was played yet;
                # otherwise the start of the sentence would repeat.
                if attempt == 1 or started:
                    raise
                log.debug("Cartesia connection lost (%s); reconnecting", exc)

    def _stream_once(self, text, should_stop):
        ctx = self._connect().context(
            timeout=10.0,
            model_id=config.CARTESIA_MODEL_ID,
            voice={"mode": "id", "id": config.CARTESIA_VOICE_ID},
            output_format={"container": "raw", "encoding": "pcm_s16le",
                           "sample_rate": config.CARTESIA_SAMPLE_RATE},
            language=config.CARTESIA_LANGUAGE,
            generation_config={"speed": config.CARTESIA_SPEED},
        )
        ctx.push(text, continue_=False)
        for event in ctx.receive():
            if should_stop():
                ctx.cancel()
                return
            if event.type == "chunk" and event.audio:
                yield event.audio
            elif event.type == "error":
                # Structured error (Cartesia-Version >= 2026-03-01).
                raise CartesiaError(f"{getattr(event, 'error_code', None)}: "
                                    f"{getattr(event, 'title', '')} {getattr(event, 'message', '')}")

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


def _split_sentences(text):
    """Split on sentence punctuation; very short fragments are merged."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?;])\s+", text) if p.strip()]
    merged = []
    for p in parts:
        if merged and len(merged[-1]) < 25:
            merged[-1] += " " + p
        else:
            merged.append(p)
    return merged or [text]


# Quick manual test:  python audio.py "Hello from Frend"
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    s = Speaker().start()
    s.say(" ".join(sys.argv[1:]) or "Hello, I am Frend. Your audio is working.")
    time.sleep(0.5)
    while s.is_speaking.is_set():
        time.sleep(0.1)
    s.shutdown()
