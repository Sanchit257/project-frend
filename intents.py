"""
intents.py - Turns a spoken (or typed) command into an action.

The voice recogniser gives us lower-case text with no punctuation, e.g.
    "keep checking my surroundings for the next five minutes"
    "stop watching"
    "what does this sign say"
`parse()` maps that to an `Intent(kind, ...)`. Plain rules are used rather
than an AI model so that commands work offline and instantly; anything
that doesn't match a rule is treated as a question for Claude.
"""

import re
from dataclasses import dataclass


@dataclass
class Intent:
    kind: str                  # see the list in parse()
    text: str = ""             # question for Claude
    duration_s: float = None   # watch: time limit (None = until cancelled)


_CANCEL = re.compile(r"^(?:cancel\w*|cancel\w* it|cancel\w* everything|cancel\w* all)$")
_STOP_WATCH = re.compile(
    r"\b(?:stop\w*|cancel\w*|end|quit|pause)\b.*\b(?:watch|watching|monitor|monitoring|"
    r"check|checking|scan|scanning|surroundings|looking)\b")
_WATCH = re.compile(
    r"\b(?:keep|start|begin|continue|continuously)\b.*\b(?:check|checking|watch|"
    r"watching|monitor|monitoring|scan|scanning|look|looking|eye)\b|"
    r"\b(?:watch|monitor|check)\s+(?:my|the)\s+(?:surroundings|path|way|area)\b")
_ALERTS_OFF = re.compile(r"\b(?:pause|mute|stop|disable|turn off)\b.*\b(?:alerts?|warnings?)\b")
_ALERTS_ON = re.compile(r"\b(?:resume|unmute|start|enable|turn on)\b.*\b(?:alerts?|warnings?)\b")
_STOP_SPEECH = re.compile(r"^(?:stop|quiet|be quiet|shut up|silence|never mind|"
                          r"stop talking)$")
# Detailed scan on the Runpod GPU.
_SCAN = re.compile(
    r"\b(?:scan|look|check|describe)\w*\b.*\b(?:in detail|detailed|closely|carefully|"
    r"thoroughly|everything)\b|\b(?:detailed|full|deep|close)\s+(?:scan|look|description)\b|"
    r"\bread (?:everything|all|all the text)\b")
# Questions about the past, answered from Elasticsearch memory.
_RECALL = re.compile(
    r"\b(?:when|where|what|how many times)\b.*\b(?:did|have|had) (?:i|we)\b.*"
    r"\b(?:see|seen|saw|pass|passed|walk|walked|meet|met|come across|encounter\w*)\b|"
    r"\b(?:did|have|had) (?:i|we) (?:\w+ )?(?:see|seen|pass|passed|walk past|come across|"
    r"encounter\w*)\b|"
    r"\blast time (?:i|we) (?:saw|passed|was|were)\b|"
    r"\b(?:do you )?remember\b|\b(?:recall|history)\b|"
    r"\bwhat (?:was|were) (?:there|around|ahead|in front)\b.*\b(?:ago|earlier|before|"
    r"yesterday|today)\b|"
    # Looking for your own things: "where is my water bottle", "find my keys".
    r"\bwhere\b(?:(?!\bmy (?:left|right)\b).)*\bmy\b(?! (?:left|right)\b)|"
    r"\bwhere\b.*\b(?:i|we)\b.*\b(?:keep|kept|put|leave|left|place|placed|set|drop|dropped|"
    r"store|stored|had|last had)\b|"
    r"\b(?:have|did) you (?:see|seen|spot|spotted)\s+my\b|\bfind my\b|\blost my\b|"
    r"\bdo you (?:know|remember) where\b")
# One-off check of the next ~30 feet ("check any obstacles in my path").
_OBSTACLES = re.compile(
    r"\bobstacles?\b|\b(?:anything|something)\b.*\b(?:in my way|in the way|in my path|"
    r"in front of my feet|blocking)\b")
# General "what's around me" requests: answered in at most two sentences.
_DESCRIBE = re.compile(
    r"^$|\b(?:examine|describe|survey)\b.*\b(?:surroundings|around|area|room|scene|"
    r"environment|place)\b|^(?:describe|look around|describe what you see|"
    r"what do you see|what can you see|what'?s around me|what is around me|"
    r"what'?s in front of me|what is in front of me|where am i)$")


def parse(text):
    """Return the Intent for one utterance.

    kinds: alerts_off, alerts_on, stop_watch, recall, scan, watch, obstacles,
           stop, describe, ask
    """
    t = " ".join(text.lower().replace("’", "'").split())

    # Order matters: the more specific phrases are checked first, e.g.
    # "stop alerts" must not be a plain "stop", and "stop watching" must not
    # start a watch session just because it contains "watching".
    if _ALERTS_OFF.search(t):
        return Intent("alerts_off")
    if _ALERTS_ON.search(t):
        return Intent("alerts_on")
    if _CANCEL.search(t) or _STOP_WATCH.search(t):
        return Intent("stop_watch")
    if _RECALL.search(t):
        return Intent("recall", text=t)
    if _SCAN.search(t):
        return Intent("scan", text=t)
    if _WATCH.search(t):
        return Intent("watch", duration_s=parse_duration(t))
    if _OBSTACLES.search(t):
        return Intent("obstacles", text=t)
    if _STOP_SPEECH.search(t):
        return Intent("stop")
    if _DESCRIBE.search(t):
        return Intent("describe", text=t)
    return Intent("ask", text=t)


# --------------------------------------------------------------------------- #
# Durations: "for the next five minutes", "for 10 minutes", "half an hour"
# --------------------------------------------------------------------------- #
_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
          "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19, "a": 1, "an": 1}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60}
_SECONDS = {"second": 1, "minute": 60, "hour": 3600}
_DURATION = re.compile(
    r"\b(\d+(?:\.\d+)?|(?:(?:twenty|thirty|forty|fifty|sixty)(?:[\s-](?:one|two|three|"
    r"four|five|six|seven|eight|nine))?)|zero|one|two|three|four|five|six|seven|eight|"
    r"nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|a|an)\s+(?:more\s+)?(second|minute|hour)s?\b")


def parse_duration(text):
    """Seconds mentioned in `text`, or None. Handles digits and number words."""
    if re.search(r"\bhalf\s+(?:an?\s+)?hour\b", text):
        return 1800.0
    if re.search(r"\b(?:a\s+)?quarter\s+(?:of\s+)?an?\s+hour\b", text):
        return 900.0
    m = _DURATION.search(text)
    if not m:
        return None
    return _number(m.group(1)) * _SECONDS[m.group(2)]


def _number(word):
    try:
        return float(word)
    except ValueError:
        pass
    parts = re.split(r"[\s-]", word)
    total = _TENS.get(parts[0], _UNITS.get(parts[0], 0))
    if len(parts) > 1:
        total += _UNITS.get(parts[1], 0)
    return float(total)
