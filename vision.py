"""
vision.py - Threaded webcam capture, plus detailed scans on a Runpod cloud GPU.

Camera: why a thread?
    cv2.VideoCapture.read() blocks until the camera delivers the next frame,
    and OpenCV/V4L2 keep a small queue of old frames. If the main loop is
    busy (running the detector, encoding a JPEG for Claude, ...), frames pile
    up and everything the device "sees" is delayed by a second or more.

    Instead, a dedicated thread reads frames as fast as the camera produces
    them and keeps ONLY the newest one. Consumers call `Camera.read()` and
    always get the most recent frame instantly, never a stale one.

DetailedScanner: "Hey Frend, scan in detail"
    Sends the full-resolution frame to Qwen2.5-VL-7B running on a Runpod
    Serverless GPU (the official vLLM worker, which speaks the OpenAI chat
    completions API). It is slower and less polished than Claude, but reads
    small text and describes layouts more thoroughly. The endpoint scales to
    zero when idle, so it costs nothing until used, but the first scan after
    a pause has to boot a GPU (a "cold start", up to a few minutes).
"""

import base64
import logging
import threading
import time

import cv2
import requests

import config

log = logging.getLogger("frend.vision")


class Camera:
    """Continuously grabs frames in a background thread; `read()` returns the latest."""

    def __init__(self, index=config.CAMERA_INDEX, width=config.CAMERA_WIDTH,
                 height=config.CAMERA_HEIGHT, fps=config.CAMERA_FPS):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps

        self._cap = None
        self._frame = None            # newest frame (numpy BGR image)
        self._frame_id = 0            # increments on every new frame
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self):
        """Open the camera and start the capture thread. Raises if no camera."""
        self._open()
        self._thread = threading.Thread(target=self._capture_loop,
                                        name="camera", daemon=True)
        self._thread.start()
        # Wait (briefly) for the first frame so callers can rely on read().
        if not self.wait_for_frame(timeout=5.0):
            raise RuntimeError("Camera opened but delivered no frames")
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        log.info("Camera stopped")

    def _open(self):
        # CAP_V4L2 talks to the Linux video driver directly. It is faster and
        # more predictable on the Pi than letting OpenCV pick GStreamer.
        cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam /dev/video{self.index}")

        # YUYV (uncompressed) skips JPEG decoding on the Pi's CPU, and the
        # C270's MJPG stream emits "Corrupt JPEG data" warnings on every frame.
        # 640x480 YUYV still fits USB 2.0 at up to 30 FPS.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.CAMERA_FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Ask the driver for the smallest possible internal queue (less lag).
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                  cap.get(cv2.CAP_PROP_FPS))
        log.info("Camera /dev/video%d opened at %dx%d @ %.0f FPS",
                 self.index, *actual)
        self._cap = cap

    # ------------------------------------------------------------------ #
    # Capture thread
    # ------------------------------------------------------------------ #
    def _capture_loop(self):
        failures = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                failures += 1
                # A loose USB cable on a wearable is a real possibility:
                # after repeated failures, try to reopen instead of dying.
                if failures >= 30:
                    log.warning("Camera stopped delivering frames; reconnecting")
                    self._reconnect()
                    failures = 0
                else:
                    time.sleep(0.02)
                continue

            failures = 0
            with self._new_frame:
                self._frame = frame
                self._frame_id += 1
                self._new_frame.notify_all()

    def _reconnect(self):
        try:
            self._cap.release()
        except Exception:
            pass
        while not self._stop.is_set():
            try:
                self._open()
                return
            except RuntimeError as exc:
                log.warning("%s; retrying in 2 s", exc)
                time.sleep(2.0)

    # ------------------------------------------------------------------ #
    # Consumer API
    # ------------------------------------------------------------------ #
    def read(self):
        """Return (frame_id, frame) for the newest frame, or (0, None) if none yet.

        The frame is shared, not copied, so treat it as read-only. The capture
        thread replaces the reference rather than writing into the array, so
        this is safe and avoids a 900 KB copy for every consumer.
        """
        with self._lock:
            return self._frame_id, self._frame

    def wait_for_frame(self, after_id=0, timeout=1.0):
        """Block until a frame newer than `after_id` exists. Returns it, or None."""
        with self._new_frame:
            if self._frame_id <= after_id:
                self._new_frame.wait_for(lambda: self._frame_id > after_id,
                                         timeout=timeout)
            if self._frame_id > after_id:
                return self._frame_id, self._frame
            return None


class MotionMeter:
    """Detects the camera being swung around (head turns, picking it up).

    Frames taken mid-swing are smeared, and the detector "sees" large phantom
    objects in them (beds, trains, tables filling the view), which would set
    off false "very close" alerts. The score is the mean pixel change between
    two tiny grayscale thumbnails: about 0.5 for a still camera, far higher
    while swinging. Costs well under a millisecond per frame.
    """

    def __init__(self, threshold=config.CAMERA_MOTION_THRESHOLD):
        self.threshold = threshold
        self.score = 0.0
        self._prev = None

    def update(self, frame):
        """Feed the next processed frame; returns True if the camera is moving fast."""
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (80, 60),
                          interpolation=cv2.INTER_AREA)
        self.score = 0.0 if self._prev is None else float(cv2.absdiff(gray, self._prev).mean())
        self._prev = gray
        return self.score > self.threshold


class WallDetector:
    """Camera-only guess at "a wall (or door, or big flat surface) right ahead".

    The object detector has no "wall" class. Up close, though, a wall fills
    the middle of the view with one plain, edge-free surface. This flags that:
    the centre and upper-centre of a small grayscale thumbnail must both have
    very few edges, little brightness variation and a similar tone, for
    several consecutive still frames, with no detected object covering the
    centre. It misses patterned walls and can mistake a plain door or a blank
    screen for a wall, so it only says "possible wall".
    """

    def __init__(self):
        self.edge_score = 0.0
        self.texture_score = 0.0
        self._streak = 0
        self._clear_streak = 0
        self._armed = True
        self._last_alert = float("-inf")

    def _looks_like_wall(self, frame):
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120),
                          interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 30, 90)
        centre = (slice(30, 90), slice(40, 120))
        upper = (slice(10, 50), slice(40, 120))
        self.edge_score = float(edges[centre].mean() / 255.0)
        upper_edges = float(edges[upper].mean() / 255.0)
        self.texture_score = float(gray[centre].std())
        tone_gap = abs(float(gray[centre].mean()) - float(gray[upper].mean()))
        return (self.edge_score < config.WALL_MAX_EDGE_DENSITY
                and upper_edges < config.WALL_MAX_EDGE_DENSITY
                and self.texture_score < config.WALL_MAX_TEXTURE
                and float(gray[upper].std()) < config.WALL_MAX_TEXTURE
                and tone_gap < config.WALL_MAX_TONE_GAP)

    def update(self, frame, detections, camera_moving, now=None):
        """Returns an alert string when a wall-like surface has been ahead for a
        while, else None. Re-arms only after the view has cleared."""
        now = time.monotonic() if now is None else now
        if camera_moving:
            self._streak = 0
            return None

        h, w = frame.shape[:2]
        covered = any(d.box[0] < w * 0.6 and d.box[2] > w * 0.4
                      and d.box[1] < h * 0.6 and d.box[3] > h * 0.4
                      for d in detections)
        if not covered and self._looks_like_wall(frame):
            self._streak += 1
            self._clear_streak = 0
        else:
            self._streak = 0
            self._clear_streak += 1
            if self._clear_streak >= config.WALL_CONFIRM_FRAMES:
                self._armed = True

        if (self._streak >= config.WALL_CONFIRM_FRAMES and self._armed
                and now - self._last_alert >= config.WALL_COOLDOWN_SECONDS):
            self._armed = False
            self._last_alert = now
            return "Possible wall or flat surface straight ahead, close."
        return None


# ---------------------------------------------------------------------- #
# Runpod detailed scan
# ---------------------------------------------------------------------- #
SCAN_PROMPT = """\
You are the detailed-scan mode of smart glasses worn by a blind person. The \
photo comes from a forward-facing camera at eye level and your reply is read \
aloud.

1. First sentence: any hazard in the walking path (steps, curbs, obstacles, \
traffic), or say the path ahead looks clear.
2. Read out ALL legible text exactly (signs, labels, screens, documents), \
saying where each one is.
3. Describe the layout from left to right with rough distances, and name \
anything a blind person would want to find (doors, seats, counters, buttons).

Plain spoken English only, no markdown, lists or symbols. At most 8 sentences.\
"""


class RunpodError(Exception):
    pass


class DetailedScanner:
    """Client for a Runpod Serverless vLLM endpoint serving a vision-language model."""

    def __init__(self, api_key=config.RUNPOD_API_KEY, endpoint_id=config.RUNPOD_ENDPOINT_ID,
                 model=config.RUNPOD_MODEL):
        self.available = bool(api_key and endpoint_id)
        self.model = model
        self._base = f"https://api.runpod.ai/v2/{endpoint_id}"
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {api_key}"

    def is_warm(self):
        """True if a worker is already running (the scan will take seconds, not minutes)."""
        try:
            resp = self._session.get(f"{self._base}/health", timeout=5)
            resp.raise_for_status()
            # "idle"/"ready" count standby workers that still have to load the
            # model; only a running worker answers in seconds.
            return resp.json().get("workers", {}).get("running", 0) > 0
        except (requests.RequestException, ValueError) as exc:
            log.debug("Runpod health check failed: %s", exc)
            return False

    def scan(self, frame, question=None):
        """Return the model's detailed description of `frame` (blocking)."""
        if not self.available:
            raise RunpodError("RUNPOD_ENDPOINT_ID is not set")
        prompt = SCAN_PROMPT
        if question:
            prompt += f'\n\nThe wearer also asked: "{question}"'
        image = encode_jpeg(frame, config.RUNPOD_IMAGE_MAX_WIDTH, quality=85)

        # The queue API (/run + /status) is used instead of the endpoint's
        # OpenAI-compatible route: that route holds one HTTP request open, and
        # Runpod's gateway drops it (HTTP 502) after ~150 s, which a cold start
        # loading the model can exceed.
        job = {"input": {
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                {"type": "text", "text": prompt},
            ]}],
            "sampling_params": {"max_tokens": config.RUNPOD_MAX_TOKENS, "temperature": 0.2},
        }}
        started = time.monotonic()
        job_id = None
        try:
            resp = self._session.post(f"{self._base}/run", json=job, timeout=30)
            if resp.status_code != 200:
                raise RunpodError(f"Runpod returned HTTP {resp.status_code}: {resp.text[:200]}")
            job_id = resp.json()["id"]
            while True:
                status = self._session.get(f"{self._base}/status/{job_id}", timeout=30).json()
                state = status.get("status")
                if state == "COMPLETED":
                    break
                if state not in ("IN_QUEUE", "IN_PROGRESS"):
                    job_id = None
                    raise RunpodError(f"scan job {state}: {str(status.get('error'))[:200]}")
                if time.monotonic() - started > config.RUNPOD_TIMEOUT_SECONDS:
                    raise RunpodError("the cloud GPU took too long to respond")
                time.sleep(1.0)
        except requests.RequestException as exc:
            raise RunpodError(f"could not reach Runpod ({type(exc).__name__})") from None
        except (ValueError, KeyError):
            raise RunpodError("unexpected Runpod response") from None
        except RunpodError:
            if job_id:
                self._cancel(job_id)
            raise

        try:
            output = status["output"]
            data = output[0] if isinstance(output, list) else output
            reply = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            raise RunpodError(f"unexpected Runpod output: {str(status)[:200]}") from None
        usage = data.get("usage", {})
        log.info("Runpod scan in %.1f s (queued %.1f s; %s in / %s out tokens): %s",
                 time.monotonic() - started, status.get("delayTime", 0) / 1000,
                 usage.get("prompt_tokens"), usage.get("completion_tokens"), reply)
        return reply.replace("*", "").replace("#", "")

    def _cancel(self, job_id):
        """Don't leave a GPU working on an answer nobody will hear."""
        try:
            self._session.post(f"{self._base}/cancel/{job_id}", timeout=5)
        except requests.RequestException:
            pass


def encode_jpeg(frame, max_width, quality):
    """Downscale (if wider than max_width) and JPEG-encode a frame as base64."""
    h, w = frame.shape[:2]
    if w > max_width:
        frame = cv2.resize(frame, (max_width, int(h * max_width / w)),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# Quick manual test:  python vision.py path/to/photo.jpg
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    scanner = DetailedScanner()
    print("warm worker:", scanner.is_warm())
    print(scanner.scan(cv2.imread(sys.argv[1])))
