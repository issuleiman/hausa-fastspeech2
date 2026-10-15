"""Orthographic normalisation for Hausa text.

Responsibilities
----------------
* lower-casing (including the hooked capitals Ɓ Ɗ Ƙ Ƴ),
* mapping the ASCII fallbacks used in informal typing  'b 'd 'k 'y  ->  ɓ ɗ ƙ ƴ,
* turning remaining apostrophes into the glottal stop ʔ (only between letters),
* unifying punctuation into three classes: word space, minor break (, ; :) and
  major break (. ? ! …),
* leaving tone / length diacritics (grave, circumflex, acute, macron) untouched
  as *combining* characters (NFD) so the G2P can read them.

Design choice: apostrophe + b/d/k/y is always read as the hooked letter. A true
glottal stop directly before b/d/k/y is rare in Hausa; if your data contains such
cases, write the hooked letters in Unicode to avoid ambiguity.
"""
import re
import unicodedata

_APOSTROPHES = "'’ʼ`´‘ʻ"
_HOOK = {"b": "ɓ", "d": "ɗ", "k": "ƙ", "y": "ƴ"}

MINOR_PUNCT = ",;:"
MAJOR_PUNCT = ".?!…"
EXPLICIT_BREAK = "|"          # optional manual phrase break marker, treated as minor


def normalize_text(text: str, strict: bool = False) -> str:
    """Return NFD-normalised lower-case text containing only: letters (+ combining
    marks), spaces, and the punctuation classes `,` (minor) `.` (major) `|`."""
    text = unicodedata.normalize("NFC", text).lower()
    for a in _APOSTROPHES:
        text = text.replace(a, "'")
    # ASCII hooked-letter fallback: 'b -> ɓ ...
    text = re.sub(r"'([bdky])", lambda m: _HOOK[m.group(1)], text)
    # remaining apostrophes: glottal stop if between two letters, else dropped
    text = re.sub(r"(?<=\w)'(?=\w)", "ʔ", text)
    text = text.replace("'", "")
    # punctuation classes
    text = re.sub(f"[{re.escape(MAJOR_PUNCT)}]+", " . ", text)
    text = re.sub(f"[{re.escape(MINOR_PUNCT)}]+", " , ", text)
    text = text.replace(EXPLICIT_BREAK, " | ")
    text = re.sub(r"[-–—_/]", " ", text)           # hyphens/dashes split words
    # drop anything that is not a letter, combining mark, ʔ, space or our punct
    allowed = []
    for ch in unicodedata.normalize("NFD", text):
        if ch.isalpha() or unicodedata.combining(ch) or ch in " .,|ʔ":
            allowed.append(ch)
        elif ch.isdigit():
            if strict:
                raise ValueError("Digits are not expanded by this frontend; spell numbers out.")
        elif ch.isspace():
            allowed.append(" ")
        elif strict:
            raise ValueError(f"Unsupported character {ch!r} in input text")
    text = "".join(allowed)
    return re.sub(r"\s+", " ", text).strip()


def strip_marks(word: str) -> str:
    """Remove all combining diacritics (tone + macron) from a word."""
    return "".join(c for c in unicodedata.normalize("NFD", word) if not unicodedata.combining(c))
