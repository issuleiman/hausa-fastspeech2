"""Tone / vowel-length lexicon.

Standard Hausa orthography (boko) does NOT mark tone or vowel length, so a TTS
frontend cannot recover them from plain text alone. This module lets you supply
that knowledge as a lexicon.

File format (UTF-8, tab separated, '#' starts a comment)::

    plain_form<TAB>annotated_form

* `plain_form`     the word as normally written (no diacritics);
* `annotated_form` the same word with diacritics:
      grave  ̀  = Low,   circumflex  ̂  = Falling,   acute  ́  = High (explicit),
      macron  ̄  = long vowel (doubling the letter also works).
  Unmarked vowels in an annotated word are High (the standard Hausa convention).

Homographs: the first entry wins (the lexicon is context-free).
The lexicon is only used when `tone_mode='auto'` and the input word carries no
tone diacritics of its own.
"""
from typing import Dict, Optional

from .normalize import normalize_text, strip_marks


class ToneLexicon:
    def __init__(self, entries: Optional[Dict[str, str]] = None):
        self.entries: Dict[str, str] = dict(entries or {})

    @classmethod
    def from_file(cls, path: str) -> "ToneLexicon":
        entries: Dict[str, str] = {}
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    raise ValueError(f"{path}:{ln}: expected 'plain<TAB>annotated', got {line!r}")
                plain = strip_marks(normalize_text(parts[0]))
                annotated = normalize_text(parts[1])
                entries.setdefault(plain, annotated)
        return cls(entries)

    def lookup(self, word: str) -> Optional[str]:
        return self.entries.get(strip_marks(word))

    def __len__(self):
        return len(self.entries)
