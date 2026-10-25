"""
detector.py - Fast, fully offline hazard / obstacle detection.

Two parts:

1. `ObjectDetector` runs NanoDet-Plus (a 3.8 MB ONNX model from the official
   OpenCV model zoo, trained on the 80 COCO classes) through OpenCV's DNN
   module. On a Raspberry Pi 5 one pass takes ~50 ms; on a Pi 4 ~120-150 ms.
   No internet is needed, so hazard alerts keep working anywhere.

2. `HazardAnnouncer` turns raw detections into short, non-repetitive spoken
   alerts like "Car on your left, close". It decides *what is worth saying*:
   only hazard-relevant classes, confirmed over several frames, ranked by how
   close/dangerous they are, and rate-limited so the user isn't spammed.

Distance is estimated from how much of the frame an object fills. A single
camera can't measure depth, but "bigger in the image = closer" is a robust
rule of thumb for obstacles at walking distance.
"""

import logging
import time
from dataclasses import dataclass, replace

import cv2
import numpy as np

import config

log = logging.getLogger("frend.detector")

# The 80 COCO class names, in the order the model outputs them.
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

# Classes worth interrupting the user for, with a danger weight used for
# ranking (higher = announced first). Anything not listed is ignored by the
# alert system; Claude covers the rest when the user asks for a description.
HAZARD_WEIGHTS = {
    # Moving vehicles: most dangerous, announced even when still far away.
    "car": 10, "bus": 10, "truck": 10, "motorcycle": 10, "train": 10,
    "bicycle": 8,
    # Moving living things.
    "person": 5, "dog": 6, "horse": 6, "cow": 6,
    # Street furniture and signals.
    "traffic light": 4, "stop sign": 4, "fire hydrant": 5,
    "parking meter": 5, "bench": 5,
    # Indoor obstacles you can walk into or trip over.
    "chair": 4, "couch": 4, "dining table": 5, "bed": 3, "toilet": 3,
    "potted plant": 4, "suitcase": 5, "refrigerator": 3,
    "skateboard": 5,
}
# These are reported at any distance; everything else only when close.
FAR_ALERT_CLASSES = {"car", "bus", "truck", "motorcycle", "train", "bicycle",
                     "traffic light", "stop sign"}

PROXIMITY_NAMES = ("far", "close", "very close")   # indexed by Detection.proximity

# Confidence needed before a detection is spoken as an alert or saved to
# memory. The detector itself keeps weaker guesses (they're still a useful
# hint for Claude). With the C270's soft image, NanoDet's wrong guesses
# mostly score 0.45-0.55, and large flat surfaces (carpet, desks, floors)
# often come out as big furniture or animals, so those classes need more.
TRUSTED_CONFIDENCE = 0.55
SUSPICIOUS_CLASSES_CONFIDENCE = 0.70
SUSPICIOUS_CLASSES = {"bed", "couch", "toilet", "refrigerator", "oven", "sink", "train",
                      "airplane", "boat", "cat", "horse", "cow", "sheep", "elephant", "bear",
                      "zebra", "giraffe", "kite", "surfboard", "skis", "snowboard"}


def is_trusted(det):
    """True if `det` is confident enough to announce or remember."""
    need = (SUSPICIOUS_CLASSES_CONFIDENCE if det.label in SUSPICIOUS_CLASSES
            else TRUSTED_CONFIDENCE)
    return det.confidence >= need

# Spoken names, where the COCO label reads awkwardly.
SPOKEN_NAMES = {"dining table": "table", "motorcycle": "motorbike",
                "potted plant": "plant", "tv": "TV"}


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple            # (x1, y1, x2, y2) in original frame pixels
    position: str         # "left" | "ahead" | "right"
    proximity: int        # 0 = far, 1 = close, 2 = very close

    @property
    def key(self):
        """Identity used for de-duplicating announcements."""
        return (self.label, self.position)


class ObjectDetector:
    """NanoDet-Plus (416x416, COCO) via OpenCV DNN, with letterbox pre-processing."""

    INPUT_SIZE = 416
    STRIDES = (8, 16, 32)
    REG_MAX = 7   # each box side is predicted as a distribution over 8 bins

    def __init__(self, model_path=config.DETECTOR_MODEL,
                 conf_threshold=config.DETECTOR_CONFIDENCE,
                 nms_iou=config.DETECTOR_NMS_IOU):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Detector model not found at {model_path}. Run ./setup_models.sh")
        cv2.setNumThreads(config.OPENCV_THREADS)
        self.net = cv2.dnn.readNet(str(model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.out_names = self.net.getUnconnectedOutLayersNames()
        self.conf_threshold = conf_threshold
        self.nms_iou = nms_iou

        # Normalisation constants the model was trained with (BGR order).
        self.mean = np.array([103.53, 116.28, 123.675], np.float32)
        self.std = np.array([57.375, 57.12, 58.395], np.float32)

        # Pre-compute the centre point of every grid cell at every stride.
        # These never change, so doing it once saves work on every frame.
        self.anchors = []
        for stride in self.STRIDES:
            n = self.INPUT_SIZE // stride
            ys, xs = np.mgrid[0:n, 0:n]
            centres = np.stack([xs.ravel(), ys.ravel()], axis=1) * stride
            self.anchors.append(centres.astype(np.float32) + 0.5 * (stride - 1))
        self.anchors = np.concatenate(self.anchors)       # (3549, 2)
        self.anchor_strides = np.concatenate(
            [np.full((self.INPUT_SIZE // s) ** 2, s, np.float32) for s in self.STRIDES])
        self.bins = np.arange(self.REG_MAX + 1, dtype=np.float32)
        log.info("Loaded detector %s", model_path.name)

    # ------------------------------------------------------------------ #
    def detect(self, frame):
        """Run the model on a BGR frame and return a list of `Detection`s."""
        h, w = frame.shape[:2]
        blob, scale, pad_x, pad_y = self._preprocess(frame)

        self.net.setInput(blob)
        outputs = self.net.forward(self.out_names)
        boxes, scores, class_ids = self._decode(outputs)
        if len(scores) == 0:
            return []

        # Undo the letterbox: model coordinates -> original frame pixels.
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - pad_x) / scale).clip(0, w)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pad_y) / scale).clip(0, h)

        # Non-maximum suppression removes overlapping duplicates of one object.
        xywh = np.column_stack([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]])
        keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(),
                                self.conf_threshold, self.nms_iou)

        detections = []
        for i in np.array(keep).flatten():
            x1, y1, x2, y2 = boxes[i]
            detections.append(Detection(
                label=COCO_CLASSES[class_ids[i]],
                confidence=float(scores[i]),
                box=(int(x1), int(y1), int(x2), int(y2)),
                position=_position(x1, x2, w),
                proximity=_proximity(x1, y1, x2, y2, w, h),
            ))
        return detections

    def _preprocess(self, frame):
        """Letterbox to 416x416 (keeps aspect ratio, pads the rest) + normalise."""
        h, w = frame.shape[:2]
        scale = self.INPUT_SIZE / max(h, w)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.INPUT_SIZE - nw) // 2
        pad_y = (self.INPUT_SIZE - nh) // 2
        canvas = np.zeros((self.INPUT_SIZE, self.INPUT_SIZE, 3), np.uint8)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        img = (canvas.astype(np.float32) - self.mean) / self.std
        blob = cv2.dnn.blobFromImage(img)                  # HWC -> NCHW
        return blob, scale, pad_x, pad_y

    def _decode(self, outputs):
        """Turn the raw network outputs into boxes, scores and class ids."""
        # The model has 6 outputs: per stride, one (N, 80) class-score map and
        # one (N, 32) box map. Sort by grid size so strides line up with the
        # pre-computed anchors regardless of the order OpenCV returns them.
        outs = [o.reshape(o.shape[-2], o.shape[-1]) for o in outputs]
        cls = sorted([o for o in outs if o.shape[1] == len(COCO_CLASSES)],
                     key=lambda o: -o.shape[0])
        reg = sorted([o for o in outs if o.shape[1] == 4 * (self.REG_MAX + 1)],
                     key=lambda o: -o.shape[0])
        cls = np.concatenate(cls)                        # (3549, 80), sigmoid
        reg = np.concatenate(reg)                        # (3549, 32)

        class_ids = cls.argmax(axis=1)
        scores = cls[np.arange(len(cls)), class_ids]

        # Throw away weak candidates *before* the expensive box maths.
        mask = scores >= self.conf_threshold
        if not mask.any():
            return np.empty((0, 4)), np.empty(0), np.empty(0, int)
        reg, scores, class_ids = reg[mask], scores[mask], class_ids[mask]
        anchors, strides = self.anchors[mask], self.anchor_strides[mask]

        # Each side's distance = expected value of a softmax over 8 bins.
        reg = reg.reshape(-1, 4, self.REG_MAX + 1)
        reg = np.exp(reg - reg.max(axis=2, keepdims=True))
        reg /= reg.sum(axis=2, keepdims=True)
        dist = (reg @ self.bins) * strides[:, None]      # (N, 4): l, t, r, b

        boxes = np.column_stack([anchors[:, 0] - dist[:, 0],
                                 anchors[:, 1] - dist[:, 1],
                                 anchors[:, 0] + dist[:, 2],
                                 anchors[:, 1] + dist[:, 3]])
        return boxes, scores, class_ids


def _position(x1, x2, frame_w):
    """Split the view into thirds. The camera faces forward, so image left = user's left."""
    cx = (x1 + x2) / 2 / frame_w
    if cx < 0.33:
        return "left"
    if cx > 0.67:
        return "right"
    return "ahead"


def _proximity(x1, y1, x2, y2, frame_w, frame_h):
    """Rough distance class from apparent size: 2 very close, 1 close, 0 far."""
    area = (x2 - x1) * (y2 - y1) / (frame_w * frame_h)
    height = (y2 - y1) / frame_h
    if area > 0.30 or height > 0.85:
        return 2
    if area > 0.08 or height > 0.45:
        return 1
    return 0


def _alert_confident(det):
    """Alerts accept slightly weaker sightings than memory does: the frame
    confirmation below already filters one-off false positives."""
    need = (SUSPICIOUS_CLASSES_CONFIDENCE if det.label in SUSPICIOUS_CLASSES
            else config.HAZARD_MIN_CONFIDENCE)
    return det.confidence >= need


def _area(det):
    x1, y1, x2, y2 = det.box
    return max(1, (x2 - x1) * (y2 - y1))


class _Track:
    __slots__ = ("hits", "urgent_hits", "last_seen", "said_proximity", "said_area")

    def __init__(self):
        self.hits = 0
        self.urgent_hits = 0
        self.last_seen = 0.0
        self.said_proximity = None    # None = not announced yet
        self.said_area = 0


class HazardAnnouncer:
    """Event-based alerts: each object is announced once when it first shows
    up close, and again only when it gets closer (a nearer distance level, or
    clearly bigger in view). Nothing is repeated just because time passed; an
    object that leaves the view for HAZARD_FORGET_SECONDS counts as new when
    it comes back."""

    def __init__(self):
        self._tracks = {}          # Detection.key -> _Track
        self._last_any = float("-inf")

    def update(self, detections, now=None, camera_moving=False):
        """Feed one frame's detections. Returns (text, urgent) or None."""
        now = time.monotonic() if now is None else now
        self._tracks = {k: t for k, t in self._tracks.items()
                        if now - t.last_seen < config.HAZARD_FORGET_SECONDS}
        if camera_moving:
            # Smeared frames produce phantom objects: skip them, but keep the
            # tracks so walking jitter doesn't make objects "new" again.
            return None

        relevant = [d for d in detections
                    if d.label in HAZARD_WEIGHTS and _alert_confident(d)
                    and (d.proximity > 0 or d.label in FAR_ALERT_CLASSES)]

        candidates = []
        for d in relevant:
            t = self._tracks.get(d.key)
            if t is None:
                t = self._tracks[d.key] = _Track()
            # A short dropout (one missed frame) doesn't restart the count.
            if now - t.last_seen > config.HAZARD_DROPOUT_SECONDS:
                t.hits = t.urgent_hits = 0
            t.last_seen = now
            t.hits += 1
            confident_close = (d.proximity == 2
                               and d.confidence >= config.HAZARD_URGENT_MIN_CONFIDENCE)
            t.urgent_hits = t.urgent_hits + 1 if confident_close else 0

            # "Very close" jumps ahead of everything, so it needs more evidence;
            # until then (or if weak) the object counts as plain "close".
            if d.proximity == 2 and t.urgent_hits < config.HAZARD_URGENT_CONFIRM_FRAMES:
                d = replace(d, proximity=1)
            if t.hits < config.HAZARD_CONFIRM_FRAMES:
                continue

            if t.said_proximity is None or d.proximity > t.said_proximity:
                candidates.append((d, t, False))
            elif (d.proximity >= 1
                  and _area(d) >= t.said_area * config.HAZARD_APPROACH_GROWTH):
                candidates.append((d, t, True))

        if not candidates or now - self._last_any < config.HAZARD_MIN_GAP_SECONDS:
            return None

        # Most important first: closest, then most dangerous, then centred.
        d, t, approaching = max(candidates, key=lambda c: (
            c[0].proximity, HAZARD_WEIGHTS[c[0].label], c[0].position == "ahead"))
        t.said_proximity = d.proximity
        t.said_area = _area(d)
        self._last_any = now
        return describe_detection(d, approaching), d.proximity == 2


def describe_detection(det, approaching=False):
    """'Car on your left, close' style phrase. Kept short so it's spoken fast."""
    name = SPOKEN_NAMES.get(det.label, det.label)
    where = "ahead" if det.position == "ahead" else f"on your {det.position}"
    if approaching:
        return f"{name.capitalize()} {where}, getting closer"
    if det.proximity == 2:
        return f"Careful, {name} {where}, very close"
    if det.proximity == 1:
        return f"{name.capitalize()} {where}, close"
    return f"{name.capitalize()} {where}"


def summarize_detections(detections):
    """Short comma-separated list for giving Claude extra context."""
    return ", ".join(f"{d.label} ({d.position}, {PROXIMITY_NAMES[d.proximity]})"
                     for d in detections) or "nothing"


def draw_detections(frame, detections):
    """Debug overlay for the optional --show preview window."""
    out = frame.copy()
    colours = [(0, 200, 0), (0, 200, 255), (0, 0, 255)]   # far / close / very close
    for d in detections:
        x1, y1, x2, y2 = d.box
        c = colours[d.proximity]
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        cv2.putText(out, f"{d.label} {d.confidence:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    return out
