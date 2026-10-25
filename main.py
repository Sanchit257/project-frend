#!/usr/bin/env python3
"""
main.py - Project Frend: assistive smart-glasses entry point.

Thread layout (each runs independently, so nothing blocks anything else):

    [camera]      vision.Camera            grabs webcam frames, keeps newest
    [main]        hazard loop              NanoDet detection -> instant alerts (always on)
    [watch]       watch.SurroundingsWatch  Claude safety check every ~10 s, during sessions
    [voice]       voice.VoiceCommandListener  "Hey Frend ..." -> intents.parse()
    [keyboard]    typed commands over SSH (same commands as voice)
    [claude]      one-off scene descriptions, questions and memory answers
    [scan]        vision.DetailedScanner   "scan in detail" on a Runpod cloud GPU
    [memory]      memory.VisualMemory      batches everything seen/said to Elasticsearch
    [speaker]     audio.Speaker            the only thread that talks (Cartesia) -> AirPods

Run:
    python main.py              # headless, as on the glasses
    python main.py --show       # with a debug preview window (needs a desktop)
    python main.py --help       # all options
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time

import cv2

import config
import intents
from audio import Speaker, kill_leftover_players
from detector import (PROXIMITY_NAMES, HazardAnnouncer, ObjectDetector, draw_detections,
                      summarize_detections)
from memory import VisualMemory
from vision import Camera, DetailedScanner, MotionMeter, RunpodError, WallDetector

log = logging.getLogger("frend")


class FrendApp:
    def __init__(self, args):
        self.args = args
        self.running = threading.Event()
        self.alerts_enabled = not args.no_alerts

        self.speaker = Speaker(engine=args.tts or config.TTS_ENGINE)
        self.camera = Camera(index=args.camera)
        self.detector = None
        self.announcer = HazardAnnouncer()
        self.motion = MotionMeter()
        self.walls = WallDetector()
        self.describer = None       # scene.SceneDescriber (Claude)
        self.watch = None           # watch.SurroundingsWatch
        self.listener = None        # voice.VoiceCommandListener
        self.memory = VisualMemory(clear_on_start=config.MEMORY_CLEAR_ON_START
                                   and not args.keep_memory)
        self.scanner = DetailedScanner()
        self.web = None             # webview.LiveView (--web)
        self._fps = 0.0

        self._latest_detections = []          # shared with the Claude thread
        self._busy = threading.Lock()         # one cloud request at a time
        self._stop_count = 0                  # bumped by every "stop" command

    # ------------------------------------------------------------------ #
    # Startup / shutdown
    # ------------------------------------------------------------------ #
    def start(self):
        self.running.set()
        self.speaker.start()
        self.speaker.say("Frend is starting.")
        self.memory.start()

        try:
            self.camera.start()
        except RuntimeError as exc:
            log.error("%s", exc)
            self.speaker.say("I can't find the camera. Please check the cable.")
            self._wait_for_speech()
            return False

        self.detector = ObjectDetector()

        if self.args.web:
            from webview import LiveView
            try:
                self.web = LiveView(port=self.args.web).start()
            except OSError as exc:
                log.warning("Live view disabled: %s", exc)

        # Each optional feature fails on its own without taking the app down.
        if not self.args.no_claude:
            try:
                from scene import SceneDescriber
                from watch import SurroundingsWatch
                self.describer = SceneDescriber()
                self.watch = SurroundingsWatch(self.camera, self.describer, self.speaker,
                                               memory=self.memory)
            except Exception as exc:
                log.warning("Claude features disabled: %s", exc)

        if not self.args.no_voice:
            try:
                from voice import VoiceCommandListener
                self.listener = VoiceCommandListener(
                    self.handle_command, frend_is_talking=self.speaker.spoke_recently).start()
            except Exception as exc:
                log.warning("Voice commands disabled: %s", exc)

        if sys.stdin.isatty():
            threading.Thread(target=self._keyboard_loop, name="keyboard", daemon=True).start()

        ready = "Frend is ready."
        if self.listener:
            ready += " Say hey friend when you need me."
        if self.describer is None and not self.args.no_claude:
            ready += " Scene descriptions are unavailable."
        self.speaker.say(ready)
        return True

    def shutdown(self):
        log.info("Shutting down")
        self.running.clear()
        if self.watch:
            self.watch.stop(message=None)
        if self.listener:
            self.listener.stop()
        self.camera.stop()
        if self.web:
            self.web.stop()
        self.memory.shutdown()
        self.speaker.shutdown()
        if self.args.show:
            cv2.destroyAllWindows()

    def _wait_for_speech(self, timeout=10.0):
        end = time.monotonic() + timeout
        time.sleep(0.3)
        while self.speaker.is_speaking.is_set() and time.monotonic() < end:
            time.sleep(0.1)

    # ------------------------------------------------------------------ #
    # Main real-time loop: local detection + hazard alerts (always on)
    # ------------------------------------------------------------------ #
    def run(self):
        min_interval = 1.0 / max(config.DETECTOR_MAX_FPS, 0.1)
        last_id, last_run = 0, 0.0
        fps_count, fps_t0 = 0, time.monotonic()

        while self.running.is_set():
            # Battery/heat saver: cap the detection rate.
            wait = min_interval - (time.monotonic() - last_run)
            if wait > 0:
                time.sleep(wait)

            # Always process the *newest* frame; any frames in between are
            # skipped on purpose, which is what keeps alerts real-time.
            got = self.camera.wait_for_frame(after_id=last_id, timeout=1.0)
            if got is None:
                continue
            last_id, frame = got
            last_run = time.monotonic()

            detections = self.detector.detect(frame)
            self._latest_detections = detections
            moving = self.motion.update(frame)
            if not moving:
                self.memory.note_sightings(detections)

            if self.alerts_enabled:
                alert = self.announcer.update(detections, camera_moving=moving)
                if alert:
                    text, urgent = alert
                    self.speaker.say_hazard(text, urgent=urgent)
                    top = max(detections, key=lambda d: d.confidence, default=None)
                    self.memory.log("hazard", text, labels=[d.label for d in detections],
                                    position=top and top.position,
                                    proximity=top and PROXIMITY_NAMES[top.proximity],
                                    source="nanodet", urgent=urgent)

                if config.WALL_ALERTS:
                    wall = self.walls.update(frame, detections, camera_moving=moving)
                    if wall:
                        self.speaker.say_hazard(wall)
                        self.memory.log("hazard", wall, labels=["wall"], position="ahead",
                                        proximity="close", source="wall-heuristic")

            if self.web:
                self.web.publish(frame, detections, fps=self._fps, motion=self.motion.score,
                                 moving=moving, walls=self.walls, alerts=self.alerts_enabled)

            if self.args.show:
                cv2.imshow("Frend (debug)", draw_detections(frame, detections))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            fps_count += 1
            if time.monotonic() - fps_t0 >= 2:
                self._fps = fps_count / (time.monotonic() - fps_t0)
            if time.monotonic() - fps_t0 >= 10:
                log.debug("Detector: %.1f FPS, %d objects, motion %.1f",
                          fps_count / (time.monotonic() - fps_t0), len(detections),
                          self.motion.score)
                fps_count, fps_t0 = 0, time.monotonic()

    # ------------------------------------------------------------------ #
    # Commands (from the voice thread or the keyboard thread)
    # ------------------------------------------------------------------ #
    def handle_command(self, kind, text=""):
        if kind == "wake":
            # Quiet while the user thinks of the question; ends at the
            # command or when the listening window runs out.
            self.speaker.hold_alerts(config.FOLLOW_UP_SECONDS)
            self.speaker.say("Yes?")
        elif kind == "unclear":
            self._ask_to_repeat()
        elif kind == "command":
            self.speaker.release_alerts()
            intent = intents.parse(text)
            # A lone word that isn't a known command ("three", "hello") is
            # almost always a mishearing, not a real question.
            if intent.kind == "ask" and len(text.split()) < 2:
                log.info("Treating %r as misheard", text)
                self._ask_to_repeat()
                return
            log.info("Intent: %s", intent)
            self.memory.log("command", text, source="user", intent=intent.kind)
            self.dispatch(intent)

    def _ask_to_repeat(self):
        self.speaker.hold_alerts(config.FOLLOW_UP_SECONDS)
        self.speaker.say("Sorry, I didn't catch that. Please say it again.")
        if self.listener:
            self.listener.expect_follow_up()

    def dispatch(self, intent):
        k = intent.kind
        if k == "stop":
            self._stop_count += 1          # answers still on their way are dropped too
            self.speaker.stop_speaking()
        elif k == "alerts_off":
            self.alerts_enabled = False
            self.speaker.say("Obstacle alerts paused.")
        elif k == "alerts_on":
            self.alerts_enabled = True
            self.speaker.say("Obstacle alerts on.")
        elif k == "stop_watch":
            if not (self.watch and self.watch.stop()):
                self.speaker.say("I wasn't checking your surroundings.")
        elif k == "watch":
            self._start_watch(intent)
        elif k == "scan":
            self.detailed_scan(intent.text)
        elif k == "recall":
            self.recall(intent.text)
        elif k == "obstacles":
            self.ask_claude(intent.text, mode="obstacles")
        elif k == "describe":
            self.ask_claude("Describe my surroundings.", mode="describe")
        else:
            self.ask_claude(intent.text)

    def _say_answer(self, asked_at_stop_count, text):
        """Speak a slow answer unless the user said "stop" since asking."""
        if self._stop_count == asked_at_stop_count:
            self.speaker.say(text)
        else:
            log.info("Answer dropped (user said stop): %s", text)

    def _start_watch(self, intent):
        if self.watch is None:
            self.speaker.say("Surroundings checking needs Claude, which isn't available. "
                             "Basic obstacle alerts are still on.")
            return
        self.watch.start(duration_s=intent.duration_s)

    def ask_claude(self, question="", mode="ask"):
        """mode: "ask" (any question), "describe" (2 sentences max) or "obstacles"."""
        if self.describer is None:
            self.speaker.say("Scene descriptions are not available. Check the API key and internet.")
            return
        if not self._busy.acquire(blocking=False):
            self.speaker.say("Still working on your last question.")
            return
        # Grab the frame *now*, at the moment the user asked.
        _, frame = self.camera.read()
        context = summarize_detections(self._latest_detections)
        labels = [d.label for d in self._latest_detections]
        self.speaker.say("Checking your path." if mode == "obstacles" else "Let me look.")
        threading.Thread(target=self._claude_worker,
                         args=(frame, question, context, labels, mode, self._stop_count),
                         name="claude", daemon=True).start()

    def _claude_worker(self, frame, question, context, labels, mode, stop_count):
        try:
            t0 = time.monotonic()
            if mode == "obstacles":
                reply = self.describer.list_obstacles(frame, context)
            elif mode == "describe":
                reply = self.describer.describe(frame, question, context, max_sentences=2)
            else:
                reply = self.describer.describe(frame, question, context)
            log.info("Claude answered in %.1f s", time.monotonic() - t0)
            self._say_answer(stop_count, reply)
            self.memory.log("description", reply, question=question or None,
                            labels=labels, source="claude")
        except Exception as exc:
            self._say_claude_error(exc, "Sorry, something went wrong describing the scene.")
        finally:
            self._busy.release()

    def _say_claude_error(self, exc, generic):
        import anthropic
        if isinstance(exc, anthropic.AuthenticationError):
            self.speaker.say("My Anthropic API key was rejected.")
        elif isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
            self.speaker.say("I can't reach the internet right now.")
        elif isinstance(exc, anthropic.RateLimitError):
            self.speaker.say("I'm being rate limited. Try again in a moment.")
        else:
            log.error("Claude request failed", exc_info=exc)
            self.speaker.say(generic)

    # ------------------------------------------------------------------ #
    # Detailed scan (Runpod GPU)
    # ------------------------------------------------------------------ #
    def detailed_scan(self, question=""):
        if not self.scanner.available:
            # Not deployed yet: Claude can still give a thorough description.
            log.info("Runpod endpoint not configured; using Claude for the scan")
            self.ask_claude("Describe everything in detail and read out all visible text.")
            return
        if not self._busy.acquire(blocking=False):
            self.speaker.say("Still working on your last question.")
            return
        _, frame = self.camera.read()
        labels = [d.label for d in self._latest_detections]
        threading.Thread(target=self._scan_worker,
                         args=(frame, question, labels, self._stop_count),
                         name="scan", daemon=True).start()

    def _scan_worker(self, frame, question, labels, stop_count):
        # Runpod's worker counts can't tell a loaded GPU from a cold one, so
        # explain the wait only once the scan is actually slow.
        self.speaker.say("Scanning in detail.")
        slow = threading.Timer(8.0, self._say_answer, args=(
            stop_count, "The cloud GPU is waking up, so this can take a couple of minutes."))
        slow.daemon = True
        slow.start()
        try:
            t0 = time.monotonic()
            try:
                reply = self.scanner.scan(frame)
            finally:
                slow.cancel()
            log.info("Detailed scan answered in %.1f s", time.monotonic() - t0)
            self._say_answer(stop_count, reply)
            self.memory.log("scan", reply, question=question or None, labels=labels,
                            source="runpod")
        except RunpodError as exc:
            log.warning("Detailed scan failed: %s", exc)
            self.speaker.say("The detailed scan didn't work. Ask me what I see for a "
                             "quick description instead.")
        except Exception:
            log.exception("Detailed scan failed")
            self.speaker.say("Sorry, something went wrong with the detailed scan.")
        finally:
            self._busy.release()

    # ------------------------------------------------------------------ #
    # Memory recall (Elasticsearch)
    # ------------------------------------------------------------------ #
    def recall(self, question):
        if not self.memory.enabled:
            self.speaker.say("My memory isn't set up. Add the Elasticsearch details to the settings.")
            return
        if not self._busy.acquire(blocking=False):
            self.speaker.say("Still working on your last question.")
            return
        seen_now = summarize_detections(self._latest_detections)
        threading.Thread(target=self._recall_worker,
                         args=(question, seen_now, self._stop_count),
                         name="recall", daemon=True).start()

    def _recall_worker(self, question, seen_now, stop_count):
        try:
            try:
                memories = self.memory.search(question)
            except Exception as exc:
                log.warning("Memory search failed: %s", exc)
                self.speaker.say("I can't reach my memory right now.")
                return
            log.info("Recall found %d memories", len(memories))
            if self.describer is not None:
                try:
                    self._say_answer(stop_count, self.describer.answer_from_memory(
                        question, memories, seen_now))
                    return
                except Exception as exc:
                    log.warning("Claude memory answer failed (%s); reading top match", exc)
            if memories:
                m = memories[0]
                self._say_answer(stop_count, f"The most recent match was {m.when()}: {m.text}")
            else:
                self._say_answer(stop_count, "I don't remember seeing that.")
        finally:
            self._busy.release()

    def _keyboard_loop(self):
        """Typed commands over SSH: the same phrases as voice, without 'hey friend'."""
        print("\nType any voice command (e.g. 'keep checking my surroundings for 5 minutes',\n"
              "'stop watching', 'what does this sign say', 'scan in detail',\n"
              "'check any obstacles in my path', 'when did I last see stairs') + Enter.\n"
              "[Enter] alone = describe scene | s = stop speech | q = quit\n", flush=True)
        while self.running.is_set():
            try:
                line = sys.stdin.readline()
            except (EOFError, OSError):
                return
            if not line:
                return                              # stdin closed
            cmd = line.strip()
            if cmd.lower() == "q":
                self.running.clear()
            elif cmd.lower() == "s":
                self.dispatch(intents.Intent("stop"))
            elif not cmd:
                self.dispatch(intents.Intent("describe"))
            else:
                self.handle_command("command", cmd)


def parse_args():
    p = argparse.ArgumentParser(description="Project Frend assistive smart glasses")
    p.add_argument("--show", action="store_true", help="show a debug preview window")
    p.add_argument("--web", nargs="?", type=int, const=8080, metavar="PORT",
                   help="serve a live camera + detections view in the browser (default port 8080)")
    p.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="webcam index")
    p.add_argument("--tts", choices=["auto", "cartesia", "gtts", "pyttsx3"], help="speech engine")
    p.add_argument("--no-voice", action="store_true", help="disable 'Hey Frend' listening")
    p.add_argument("--no-claude", action="store_true", help="disable Claude features")
    p.add_argument("--no-alerts", action="store_true", help="start with obstacle alerts paused")
    p.add_argument("--keep-memory", action="store_true",
                   help="keep memory from previous runs instead of erasing it at startup")
    p.add_argument("--debug", action="store_true", help="verbose logging")
    return p.parse_args()


PID_FILE = config.CACHE_DIR / "frend.pid"


def stop_previous_instance():
    """A run left in the background (Ctrl+Z, a detached SSH session) would keep
    talking and hold the camera and mic; end it before starting fresh."""
    try:
        pid = int(PID_FILE.read_text())
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            is_frend = b"main.py" in f.read()
    except (OSError, ValueError):
        is_frend = False
    if is_frend and pid != os.getpid():
        log.info("Stopping a previous Frend run (pid %d)", pid)
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        time.sleep(0.3)
    kill_leftover_players()
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)-14s %(message)s",
                        datefmt="%H:%M:%S")
    # Third-party libraries are very chatty at DEBUG level.
    for noisy in ("gtts", "urllib3", "httpx", "httpx2", "httpcore", "anthropic",
                  "elastic_transport", "elasticsearch", "websockets", "cartesia"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    stop_previous_instance()
    app = FrendApp(args)
    # systemd sends SIGTERM on stop, and closing the SSH window sends SIGHUP;
    # both become a clean shutdown that silences the speaker immediately.
    def _quit(*_):
        app.running.clear()
        app.speaker.shutdown()
    signal.signal(signal.SIGTERM, _quit)
    signal.signal(signal.SIGHUP, _quit)

    try:
        if app.start():
            app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
