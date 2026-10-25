"""
memory.py - Long-term visual memory in Elasticsearch.

Everything Frend sees or says is logged as a small JSON document:

    {"@timestamp": "...", "event_type": "hazard", "text": "Careful, stairs ahead",
     "labels": ["person"], "position": "ahead", "proximity": "close",
     "source": "nanodet", "question": null}

event_type is one of:
    sighting    - an object class NanoDet started seeing (at most once a minute per class)
    hazard      - a hazard alert that was spoken
    description - a Claude answer to "what do you see" / a question
    watch       - a surroundings-watch result (clear / hazard / blocked)
    scan        - a detailed Runpod scan
    command     - a voice or keyboard command

That makes questions like "Hey Frend, when did I last see stairs?" answerable:
`search()` turns the question into a full-text query plus an optional time
window ("today", "in the last 10 minutes") and returns the best matches with
how long ago they happened.

Logging never blocks the caller: documents go into an in-memory buffer that a
background thread sends with the bulk API. While offline, the buffer simply
grows (up to MEMORY_MAX_BUFFERED) and is flushed when the network returns.
"""

import collections
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import config

log = logging.getLogger("frend.memory")

# Mappings only: Elastic Cloud Serverless manages shards and replicas itself.
# The "english" analyzer stems words, so "stair" also matches "stairs".
MAPPINGS = {
    "properties": {
        "@timestamp": {"type": "date"},
        "event_type": {"type": "keyword"},
        "text": {"type": "text", "analyzer": "english"},
        "question": {"type": "text", "analyzer": "english"},
        "labels": {"type": "keyword",
                   "fields": {"text": {"type": "text", "analyzer": "english"}}},
        "position": {"type": "keyword"},
        "proximity": {"type": "keyword"},
        "source": {"type": "keyword"},
        "extra": {"type": "object", "enabled": False},
    }
}

SIGHTING_REPEAT_SECONDS = 60.0
FLUSH_INTERVAL_SECONDS = 2.0

# Words that carry no meaning for the search itself.
STOPWORDS = set("""
a an the i me my we you your it its is are was were be been am do did does done
have has had see saw seen seeing look looked notice noticed spot spotted find found
when where what which who how last time times ever any some there here that this
those these of in on at to for from with by about near around ago today yesterday
tonight morning afternoon evening week minute minutes hour hours second seconds
remember recall tell show can could would should please frend friend hey ok okay
lately recently earlier previous previously before past just
lost put leave drop dropped place placed set know spotted our keep kept store stored had
""".split())

# Claude and NanoDet name things differently; widen the search a little.
SYNONYMS = {
    "stairs": ["steps", "staircase", "stairway", "stair"],
    "steps": ["stairs", "staircase", "step"],
    "car": ["vehicle", "cars", "truck"],
    "person": ["people", "man", "woman", "pedestrian"],
    "people": ["person", "man", "woman", "pedestrian"],
    "door": ["doorway", "entrance"],
    "bike": ["bicycle"],
    "bicycle": ["bike"],
    "sign": ["signs", "signage"],
    "dog": ["dogs", "puppy"],
}

NUMBER_WORDS = {"a": 1, "an": 1, "couple": 2, "few": 3, "several": 5, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15,
                "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60}


@dataclass
class Memory:
    timestamp: datetime          # timezone-aware, local time
    event_type: str
    text: str
    question: str = None

    def when(self):
        return relative_time(self.timestamp)

    def __str__(self):
        q = f' (asked "{self.question}")' if self.question else ""
        return f"[{self.when()}] {self.event_type}{q}: {self.text}"


class VisualMemory:
    """Background logger + searcher. Safe to call from any thread."""

    def __init__(self, clear_on_start=config.MEMORY_CLEAR_ON_START):
        self.enabled = bool((config.ELASTIC_CLOUD_ID or config.ELASTIC_URL)
                            and config.ELASTIC_API_KEY)
        self.clear_on_start = clear_on_start
        self._buffer = collections.deque(maxlen=config.MEMORY_MAX_BUFFERED)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._last_sighting = {}
        self._index_ready = False
        self._thread = None
        self._es = None
        if not self.enabled:
            log.warning("Elasticsearch not configured; visual memory is off")
            return
        from elasticsearch import Elasticsearch
        kwargs = {"api_key": config.ELASTIC_API_KEY, "request_timeout": 15}
        if config.ELASTIC_CLOUD_ID:
            kwargs["cloud_id"] = config.ELASTIC_CLOUD_ID
        else:
            kwargs["hosts"] = [config.ELASTIC_URL]
        self._es = Elasticsearch(**kwargs)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self):
        if self.enabled:
            self._thread = threading.Thread(target=self._run, name="memory", daemon=True)
            self._wake.set()                   # prepare (and clear) the index right away
            self._thread.start()
        return self

    def shutdown(self):
        """Send whatever is still buffered (briefly), then stop."""
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------ #
    # Logging (non-blocking)
    # ------------------------------------------------------------------ #
    def log(self, event_type, text, *, labels=None, position=None, proximity=None,
            source=None, question=None, **extra):
        if not self.enabled or not text:
            return
        doc = {
            "@timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "text": text,
            "labels": sorted(set(labels or [])),
            "position": position,
            "proximity": proximity,
            "source": source,
            "question": question,
        }
        if extra:
            doc["extra"] = extra
        self._buffer.append(doc)
        self._wake.set()

    def note_sightings(self, detections):
        """Log object classes as they come into view (rate-limited per class)."""
        if not self.enabled:
            return
        from detector import PROXIMITY_NAMES, is_trusted
        now = time.monotonic()
        for d in detections:
            if not is_trusted(d):
                continue
            last = self._last_sighting.get(d.label)
            self._last_sighting[d.label] = now
            if last is None or now - last > SIGHTING_REPEAT_SECONDS:
                distance = PROXIMITY_NAMES[d.proximity]
                self.log("sighting", f"{d.label} {d.position}, {distance}",
                         labels=[d.label], position=d.position, proximity=distance,
                         source="nanodet", confidence=round(d.confidence, 2))

    # ------------------------------------------------------------------ #
    # Background sender
    # ------------------------------------------------------------------ #
    def _run(self):
        backoff = FLUSH_INTERVAL_SECONDS
        while True:
            self._wake.wait(timeout=backoff)
            self._wake.clear()
            stopping = self._stop.is_set()
            try:
                self._ensure_index()
                self._flush()
                backoff = FLUSH_INTERVAL_SECONDS
            except Exception as exc:
                backoff = min(backoff * 2, 60.0)
                log.warning("Memory upload failed (%s); %d events buffered, retry in %.0f s",
                            _short(exc), len(self._buffer), backoff)
            if stopping:
                return
            if not self._buffer:
                # Batch a few events together instead of one request each.
                time.sleep(FLUSH_INTERVAL_SECONDS)

    def _ensure_index(self):
        if self._index_ready:
            return
        exists = self._es.indices.exists(index=config.ELASTIC_INDEX)
        if exists and self.clear_on_start:
            # Events logged since startup are still in the buffer, so nothing
            # from this run is lost.
            self._es.indices.delete(index=config.ELASTIC_INDEX)
            log.info("Erased memory from previous runs")
            exists = False
        if not exists:
            from elasticsearch import BadRequestError
            try:
                self._es.indices.create(index=config.ELASTIC_INDEX, mappings=MAPPINGS)
                log.info("Created Elasticsearch index %s", config.ELASTIC_INDEX)
            except BadRequestError as exc:
                if "resource_already_exists" not in str(exc):
                    raise
        self._index_ready = True

    def _flush(self):
        while self._buffer:
            batch = [self._buffer.popleft() for _ in range(min(200, len(self._buffer)))]
            ops = []
            for doc in batch:
                ops += [{"index": {"_index": config.ELASTIC_INDEX}}, doc]
            try:
                resp = self._es.bulk(operations=ops)
            except Exception:
                self._buffer.extendleft(reversed(batch))   # keep order, retry later
                raise
            if resp.get("errors"):
                bad = [i["index"].get("error") for i in resp["items"] if i["index"].get("error")]
                log.warning("%d memory events rejected, e.g. %s", len(bad), bad[0])
            log.debug("Logged %d memory events", len(batch))

    # ------------------------------------------------------------------ #
    # Recall
    # ------------------------------------------------------------------ #
    def search(self, question, size=config.MEMORY_RECALL_RESULTS):
        """Return the Memory entries most relevant to `question`, newest first.

        Raises on network errors so the caller can say memory is unreachable.
        """
        if not self.enabled or not self._index_ready:
            return []                          # nothing stored yet this run
        start, end = parse_time_window(question)
        keywords = extract_keywords(question)
        terms = keywords + [s for k in keywords for s in SYNONYMS.get(k, [])]

        filters = [{"terms": {"event_type": ["sighting", "hazard", "description",
                                             "watch", "scan"]}}]
        if start or end:
            rng = {}
            if start:
                rng["gte"] = start.isoformat()
            if end:
                rng["lt"] = end.isoformat()
            filters.append({"range": {"@timestamp": rng}})

        if terms:
            query = {"bool": {"filter": filters, "must": [{"multi_match": {
                "query": " ".join(terms),
                "fields": ["text", "labels.text^2", "question"],
                "fuzziness": "AUTO",
            }}]}}
            resp = self._es.search(index=config.ELASTIC_INDEX, query=query, size=30)
            hits = resp["hits"]["hits"]
            if hits:
                # Keep the clearly relevant matches, then order them by time so
                # "when did I last..." gets the most recent one first.
                best = hits[0]["_score"] or 1.0
                hits = [h for h in hits if (h["_score"] or 0) >= 0.4 * best]
                hits.sort(key=lambda h: h["_source"]["@timestamp"], reverse=True)
        else:
            resp = self._es.search(index=config.ELASTIC_INDEX, size=size,
                                   query={"bool": {"filter": filters}},
                                   sort=[{"@timestamp": "desc"}])
            hits = resp["hits"]["hits"]

        return [_to_memory(h["_source"]) for h in hits[:size]]


# ---------------------------------------------------------------------- #
# Question parsing helpers
# ---------------------------------------------------------------------- #
def extract_keywords(question):
    words = re.findall(r"[a-z]+", question.lower())
    return [w for w in words if w not in STOPWORDS and w not in NUMBER_WORDS and len(w) > 1]


def parse_time_window(question, now=None):
    """Return (start, end) local datetimes implied by the question, or (None, None)."""
    q = question.lower()
    now = now or datetime.now().astimezone()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

    m = re.search(r"(?:last|past)\s+(\d+|[a-z]+)?\s*(second|minute|hour|day|week)s?", q) \
        or re.search(r"(\d+|[a-z]+)\s+(second|minute|hour|day|week)s?\s+ago", q)
    if m:
        raw = m.group(1)
        n = int(raw) if raw and raw.isdigit() else NUMBER_WORDS.get(raw, 1)
        delta = timedelta(**{m.group(2) + "s": n})
        if "ago" in m.group(0):
            # "five minutes ago": a window around that moment.
            centre = now - delta
            return centre - delta / 2, centre + delta / 2
        return now - delta, None
    if "yesterday" in q:
        return midnight - timedelta(days=1), midnight
    if "this morning" in q:
        return midnight, midnight.replace(hour=12)
    if "this afternoon" in q:
        return midnight.replace(hour=12), midnight.replace(hour=17)
    if "tonight" in q or "this evening" in q:
        return midnight.replace(hour=17), None
    if "today" in q:
        return midnight, None
    if "this week" in q:
        return midnight - timedelta(days=now.weekday()), None
    return None, None


def relative_time(ts, now=None):
    """'just now', '4 minutes ago', 'today at 3:05 PM', 'yesterday at 9:10 AM'..."""
    now = now or datetime.now().astimezone()
    secs = (now - ts).total_seconds()
    if secs < 45:
        return "just now"
    if secs < 90:
        return "a minute ago"
    if secs < 3600:
        return f"{round(secs / 60)} minutes ago"
    clock = ts.strftime("%I:%M %p").lstrip("0")
    if ts.date() == now.date():
        hours = round(secs / 3600)
        return f"about {hours} hour{'s' if hours != 1 else ''} ago, at {clock}"
    if ts.date() == (now - timedelta(days=1)).date():
        return f"yesterday at {clock}"
    if secs < 7 * 86400:
        return f"{ts.strftime('%A')} at {clock}"
    return f"on {ts.strftime('%B')} {ts.day} at {clock}"


def _to_memory(src):
    ts = datetime.fromisoformat(src["@timestamp"].replace("Z", "+00:00")).astimezone()
    return Memory(ts, src.get("event_type", ""), src.get("text", ""), src.get("question"))


def _short(exc):
    return f"{type(exc).__name__}: {str(exc)[:120]}"


# Quick manual test:  python memory.py "when did I last see stairs"
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    mem = VisualMemory().start()
    q = " ".join(sys.argv[1:]) or "what did I see today"
    print("keywords:", extract_keywords(q), "window:", parse_time_window(q))
    for m in mem.search(q):
        print(" ", m)
    mem.shutdown()
