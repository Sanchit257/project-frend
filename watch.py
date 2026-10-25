"""
watch.py - "Keep checking my surroundings" sessions.

While a session is active, Claude looks at the camera every few seconds
(config.WATCH_INTERVAL_SECONDS) and the user hears either a short hazard
warning ("Curb drop about two steps ahead.") or "Clear."

This adds to the always-on local detector: Claude spots things the
detector's 80 object classes can't, like stairs, curbs, poles, holes and
construction.

A session ends when:
  * its time limit runs out ("for the next 5 minutes"),
  * the user cancels it ("stop watching", "cancel").
Without a time limit it runs until cancelled.
"""

import logging
import threading
import time

import config
from audio import PRIORITY_NORMAL

log = logging.getLogger("frend.watch")


class SurroundingsWatch:
    def __init__(self, camera, describer, speaker, memory=None):
        self.camera = camera
        self.describer = describer
        self.speaker = speaker
        self.memory = memory          # memory.VisualMemory, optional
        self._lock = threading.Lock()
        self._session = None          # dict describing the running session

    @property
    def active(self):
        return self._session is not None

    def start(self, duration_s=None):
        """Start (or replace) a session. duration_s=None means until stopped."""
        with self._lock:
            if self._session:
                self._session["stop"].set()
            session = {
                "stop": threading.Event(),
                "deadline": time.monotonic() + duration_s if duration_s else None,
            }
            self._session = session
        threading.Thread(target=self._run, args=(session,), name="watch", daemon=True).start()

        if duration_s:
            msg = f"Okay, I'll check your surroundings for the next {speak_duration(duration_s)}."
        else:
            msg = "Okay, I'll keep checking your surroundings until you tell me to stop."
        self.speaker.say(msg)
        log.info("Watch started (duration=%s)", duration_s)

    def stop(self, message="I've stopped checking your surroundings."):
        with self._lock:
            session, self._session = self._session, None
        if session is None:
            return False
        session["stop"].set()
        if message:
            self.speaker.say(message)
        log.info("Watch stopped")
        return True

    # ------------------------------------------------------------------ #
    def _run(self, session):
        stop = session["stop"]
        offline_warned = False
        blocked_warned = False      # say "can't see" once, not every check
        while not stop.is_set():
            started = time.monotonic()
            if session["deadline"] and started >= session["deadline"]:
                with self._lock:
                    if self._session is session:
                        self._session = None
                self.speaker.say("Time's up. I've stopped checking your surroundings.")
                return

            _, frame = self.camera.read()
            if frame is not None:
                try:
                    status, value = self.describer.check_hazards(frame)
                    offline_warned = False
                    if stop.is_set():
                        return
                    if self.memory:
                        text = clear_message(value) if status == "clear" else value or "View blocked."
                        self.memory.log("watch", text, source="claude", status=status)
                    if status == "blocked":
                        if not blocked_warned:
                            blocked_warned = True
                            self.speaker.say("I can't see well. Check the camera isn't covered.",
                                             priority=PRIORITY_NORMAL, ttl=5.0)
                    elif status == "clear":
                        blocked_warned = False
                        # Short TTL: a "Clear" that would play late is misleading.
                        self.speaker.say(clear_message(value), priority=PRIORITY_NORMAL, ttl=3.0)
                    else:
                        blocked_warned = False
                        self.speaker.say_hazard(value)
                except Exception as exc:
                    log.warning("Watch check failed: %s", exc)
                    if not offline_warned:
                        offline_warned = True
                        self.speaker.say("I can't reach the internet, so only basic "
                                         "obstacle alerts are working.")

            # Wait out the rest of the interval, but wake up exactly at the
            # time limit (and immediately if stopped).
            wait = config.WATCH_INTERVAL_SECONDS - (time.monotonic() - started)
            if session["deadline"]:
                wait = min(wait, session["deadline"] - time.monotonic())
            stop.wait(max(0.1, wait))


def clear_message(feet):
    """'Path looks clear for about 15 feet.' A single camera can only estimate
    distance, so it is rounded to 5 ft and capped at what the camera can judge."""
    if not feet or feet <= 0:
        return "Clear."
    if feet > config.WATCH_MAX_CLEAR_FEET:
        return f"Path looks clear for more than {config.WATCH_MAX_CLEAR_FEET} feet."
    feet = max(3, int(round(feet / 5) * 5)) if feet >= 5 else int(feet)
    return f"Path looks clear for about {feet} feet."


def speak_duration(seconds):
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} seconds"
    minutes, rest = divmod(seconds, 60)
    if minutes >= 60 and minutes % 60 == 0 and rest == 0:
        hours = minutes // 60
        return "hour" if hours == 1 else f"{hours} hours"
    if rest == 0:
        return "minute" if minutes == 1 else f"{minutes} minutes"
    return f"{minutes} minutes and {rest} seconds"
