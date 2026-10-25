"""
scene.py - Rich scene descriptions from Claude (Anthropic API).

The local detector only knows 80 object types and can't read text, judge
stairs, describe a room, or answer questions. When the user asks
("Hey Frend, describe what's in front of me"), the current frame is sent to
Claude together with the question, and the answer is spoken back.

This needs internet, so it is *on demand only*; real-time safety alerts never
depend on it.
"""

import logging
import re

import config
from vision import encode_jpeg

log = logging.getLogger("frend.scene")

SYSTEM_PROMPT = """\
You are Frend, a voice assistant built into smart glasses worn by a blind or \
low-vision person. You receive a photo from a forward-facing camera at eye \
level, and your reply is read aloud through earbuds.

Rules:
- Safety first: if there is any hazard (traffic, steps, stairs, curbs, poles, \
wet floor, obstacles at head height, open doors), mention it in the first sentence \
(just describe it; don't announce that you are putting it first).
- Use directions relative to the wearer: "ahead", "on your left", "slightly \
right", and rough distances in steps or metres.
- Be brief: 2-4 short sentences unless the user asks for detail or asks you \
to read text, in which case read the text exactly.
- Plain spoken English only: no markdown, lists, emojis or symbols.
- Never claim certainty you don't have; say "looks like" when unsure. If the \
image is too dark or blurry, say so and suggest turning slightly or moving.\
"""

DEFAULT_QUESTION = "Describe what is in front of me."

WATCH_PROMPT = """\
You are the safety spotter in smart glasses worn by a blind person who is \
walking. You get one photo from a forward-facing camera at eye level every \
few seconds.

Look only at the walking path ahead (roughly the next 5 metres) and anything \
moving toward it. Hazards include: steps, stairs, curbs, drop-offs, holes, \
uneven or broken pavement, poles, signs, low branches or anything at head \
height, construction, vehicles or bicycles in or entering the path, wet or \
icy floor, closed doors, glass walls, and people or objects directly in the way.

Reply format, nothing else:
- If the path looks safe: CLEAR followed by your best estimate, in whole \
feet, of how far ahead the walkable path is visibly free of hazards, e.g. \
"CLEAR 20". Judge distance from perspective cues like pavement joints, door \
and person sizes; be conservative and never exceed what you can actually see.
- Otherwise: one short spoken sentence, at most 12 words, naming the most \
important hazard with direction and distance, e.g. "Curb drop about two steps \
ahead." or "Low branch at head height, slightly left."
- If the image is too dark, blurry or blocked to judge: BLOCKED

Use exactly one of these three formats. Never explain, hedge or add a \
second sentence; your reply is spoken aloud as-is.\
"""

OBSTACLE_PROMPT = """\
You check the path for a blind person wearing smart glasses. You get one \
photo from a forward-facing camera at eye level, and your reply is read aloud.

List EVERY obstacle within about 30 feet (9 metres) of the wearer along the \
path directly ahead: anything on the floor, furniture, bags, cables, steps, \
curbs, edges or drops, poles, doors, people, pets, vehicles, and anything \
sticking out or hanging at head height. Nearest first, one short sentence \
each, saying what it is, where (ahead, slightly left, slightly right) and \
roughly how far in feet, e.g. "Chair about 6 feet ahead, slightly left."

Ignore things further than about 30 feet away or well off to the side of the \
path. If nothing is within 30 feet, reply exactly: No obstacles in the next \
30 feet. If you can't see that far (a wall, a corner, darkness), say how far \
you can see. If the image is too dark or \
blurry to judge, say so in one sentence. Plain spoken English only: no lists, \
numbering, markdown or symbols.\
"""

MEMORY_PROMPT = """\
You are Frend, the voice assistant in smart glasses worn by a blind person. \
The wearer is asking about something they passed earlier. You get their \
question and entries from the glasses' memory log, each with how long ago it \
happened. Entries come from an object detector ("sighting", "hazard") and \
from scene descriptions ("description", "watch", "scan").

Answer in one or two short spoken sentences using only the log: say when \
(using the relative time given) and where it was, including what it was \
next to or on if the log says so. If several entries match, use the most \
recent one unless the question asks otherwise. For "where is my ..." \
questions, say where it was last seen and that it is probably still there. \
You may also be told what the camera sees right now; if the thing is in \
view now, say so first. A "bottle" sighting can be the wearer's water \
bottle. If nothing answers the question, say you don't remember seeing \
that. No markdown or symbols.\
"""


class SceneDescriber:
    def __init__(self, api_key=config.ANTHROPIC_API_KEY, model=config.CLAUDE_MODEL):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set (add it to .env)")
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key,
                                          timeout=config.CLAUDE_TIMEOUT_SECONDS,
                                          max_retries=1)
        self.model = model

    def describe(self, frame, question=None, local_detections=None, max_sentences=None):
        """Send one BGR frame + question to Claude; return the reply text.

        max_sentences: hard limit on the spoken reply (extra sentences are cut).
        """
        question = (question or "").strip() or DEFAULT_QUESTION
        image_b64 = _encode_jpeg(frame)

        text = f"The wearer asks: \"{question}\""
        if max_sentences:
            text += (f"\nAnswer in at most {max_sentences} short sentences; "
                     "mention only what matters most.")
        if local_detections:
            # A hint from the on-device detector; Claude still trusts the image.
            text += f"\n(On-device detector currently sees: {local_detections}.)"

        log.info("Asking %s: %s", self.model, question)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=config.CLAUDE_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            # Extended thinking roughly doubles latency and can use up the
            # whole token budget before any spoken answer is written.
            thinking={"type": "disabled"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": "image/jpeg",
                                                 "data": image_b64}},
                    {"type": "text", "text": text},
                ],
            }],
        )
        reply = " ".join(b.text for b in response.content if b.type == "text").strip()
        log.info("Claude replied (%d in / %d out tokens): %s",
                 response.usage.input_tokens, response.usage.output_tokens, reply)
        if max_sentences:
            reply = " ".join(_sentences(reply)[:max_sentences])
        return reply or "Sorry, I couldn't make out the scene."

    def list_obstacles(self, frame, local_detections=None):
        """Every obstacle within ~30 feet of the path ahead, nearest first."""
        text = "Check the next 30 feet of my path."
        if local_detections:
            text += f"\n(On-device detector currently sees: {local_detections}.)"
        response = self.client.messages.create(
            model=self.model,
            max_tokens=250,
            system=OBSTACLE_PROMPT,
            thinking={"type": "disabled"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": "image/jpeg",
                                                 "data": _encode_jpeg(frame)}},
                    {"type": "text", "text": text},
                ],
            }],
        )
        reply = " ".join(b.text for b in response.content if b.type == "text").strip()
        log.info("Obstacle check (%d in / %d out tokens): %s",
                 response.usage.input_tokens, response.usage.output_tokens, reply)
        return reply or "Sorry, I couldn't check your path."

    def check_hazards(self, frame):
        """Quick safety check for watch mode.

        Returns one of:
            ("clear", feet)   feet = estimated clear distance, or None if not given
            ("blocked", None) view too dark, blurry or covered to judge
            ("hazard", text)  short hazard sentence to speak
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=60,
            system=WATCH_PROMPT,
            thinking={"type": "disabled"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": "image/jpeg",
                                                 "data": _encode_jpeg(frame)}},
                    {"type": "text", "text": "Check the path."},
                ],
            }],
        )
        reply = " ".join(b.text for b in response.content if b.type == "text").strip()
        log.info("Watch check (%d in / %d out tokens): %s",
                 response.usage.input_tokens, response.usage.output_tokens, reply)
        word = reply.strip(" .!").upper()
        if not reply:
            return ("clear", None)
        if word.startswith("CLEAR"):
            m = re.search(r"\d+", word)
            return ("clear", int(m.group()) if m else None)
        if re.search(r"\bBLOCKED\b", reply):
            # Also catches mixed replies like "Table surface, not a path; BLOCKED".
            return ("blocked", None)
        # Keep alerts short even if the model ignores the format or is cut
        # off by max_tokens: speak only the first sentence.
        first = re.split(r"(?<=[.!?])\s", reply, maxsplit=1)[0]
        return ("hazard", first)


    def answer_from_memory(self, question, memories, seen_now=None):
        """Answer a recall question ("when did I last see stairs?") from memory hits.

        seen_now: what the on-device detector sees at this moment, if anything.
        Text only (no image), so it costs a few hundred tokens.
        """
        log_lines = "\n".join(str(m) for m in memories) or "(no matching memories)"
        content = f'The wearer asks: "{question}"\n\nMemory log, newest first:\n{log_lines}'
        if seen_now:
            content += f"\n\nThe camera sees right now: {seen_now}."
        response = self.client.messages.create(
            model=self.model,
            max_tokens=150,
            system=MEMORY_PROMPT,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": content}],
        )
        reply = " ".join(b.text for b in response.content if b.type == "text").strip()
        log.info("Memory answer (%d in / %d out tokens): %s",
                 response.usage.input_tokens, response.usage.output_tokens, reply)
        return reply


def _sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _encode_jpeg(frame):
    return encode_jpeg(frame, config.CLAUDE_IMAGE_MAX_WIDTH, config.CLAUDE_JPEG_QUALITY)
