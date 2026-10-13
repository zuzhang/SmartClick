import re


TEXT_MAPPING = {
    "no thanks": "close_button",
    "cancel": "close_button",
    "dismiss": "close_button",
    "close": "close_button",
    "x": "close_button",
    "skip": "skip",
    "claim": "claim",
    "collect": "collect",
    "confirm": "confirm",
    "accept": "confirm",
    "yes": "confirm",
    "ok": "confirm",
    "continue": "continue",
    "done": "continue",
    "next": "next",
    "reward": "reward",
    "play": "play",
    "watch": "play",
    "go": "play",
}

SUPPORTED_LABELS = frozenset(TEXT_MAPPING.values())
_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(text: str) -> str:
    return " ".join(_NORMALIZE_RE.sub(" ", text.lower()).split())


def label_for_text(text: str) -> str | None:
    normalized = normalize_text(text)
    for phrase in sorted(TEXT_MAPPING, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized):
            return TEXT_MAPPING[phrase]
    return None
