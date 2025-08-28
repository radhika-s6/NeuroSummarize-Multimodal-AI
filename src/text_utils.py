import re

BASIC_CLEAN_RE = re.compile(r"[ \t]+")

def basic_clean(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = BASIC_CLEAN_RE.sub(" ", s)
    return s.strip()

def clip_for_model(s: str, max_chars: int = 4000) -> str:
    return s[:max_chars]
