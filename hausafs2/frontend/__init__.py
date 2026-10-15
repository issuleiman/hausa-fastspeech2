"""Hausa linguistic frontend: text -> phone tokens with tone, vowel length,
syllable weight and a word/phrase prosodic hierarchy."""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .g2p import Seg, canonical_word, parse_word, syllabify
from .lexicon import ToneLexicon
from .normalize import normalize_text, strip_marks
from .symbols import *  # noqa: F401,F403
from .symbols import (
    B_MAJOR, B_MINOR, B_NONE, B_WORD, KIND_C, KIND_P, KIND_V, LEN_LONG,
    PHONE2ID, POS_FINAL, POS_INITIAL, POS_MEDIAL, POS_NONE, POS_SINGLE,
    ROLE_NONE, TONE_H, TONE_NAMES, TONE_NONE, TONE_UNK, W_NONE, WEIGHT_NAMES,
)

__all__ = ["HausaFrontend", "Utterance", "Token", "ToneLexicon", "canonical_word",
           "normalize_text", "strip_marks"]

_PAUSE_FOR_BOUNDARY = {B_WORD: "sp", B_MINOR: "pm", B_MAJOR: "sil"}


@dataclass
class Token:
    phone: str
    kind: int
    long: int
    tone: int
    role: int
    weight: int
    syl: int          # global syllable index, -1 for pauses
    word: int         # global word index, -1 for pauses
    phrase: int       # phrase index (pauses carry the phrase they close)
    boundary: int
    syl_pos: int
    word_pos: int
    phrase_pos: int

    @property
    def label(self) -> str:
        """Segment label as it appears in the forced-alignment dictionary."""
        if self.long and self.kind != KIND_P and self.phone not in ("ai", "au"):
            return self.phone + "ː"
        return self.phone


@dataclass
class Utterance:
    text: str                  # normalised text
    words: List[str]
    tokens: List[Token]

    def __len__(self):
        return len(self.tokens)

    @property
    def labels(self) -> List[str]:
        return [t.label for t in self.tokens]

    def arrays(self) -> Dict[str, np.ndarray]:
        """Integer feature arrays consumed by the model / dataset."""
        T = self.tokens
        a = lambda f: np.asarray([f(t) for t in T], dtype=np.int64)  # noqa: E731
        return dict(
            phone=a(lambda t: PHONE2ID[t.phone]),
            length=a(lambda t: t.long),
            tone=a(lambda t: t.tone),
            role=a(lambda t: t.role),
            weight=a(lambda t: t.weight),
            kind=a(lambda t: t.kind),
            boundary=a(lambda t: t.boundary),
            syl_pos=a(lambda t: t.syl_pos),
            word_pos=a(lambda t: t.word_pos),
            phrase_pos=a(lambda t: t.phrase_pos),
            syl_idx=a(lambda t: t.syl),
            word_idx=a(lambda t: t.word),
            phrase_idx=a(lambda t: t.phrase),
        )

    def describe(self) -> str:
        """Human readable syllabified view, e.g.  sar[H|heavy].ki[L|light] _ ..."""
        out, cur_syl, buf = [], None, []

        def flush():
            if buf:
                out.append("".join(buf))
            buf.clear()

        for t in self.tokens:
            if t.kind == KIND_P:
                flush()
                out.append(f"<{t.phone}>")
                cur_syl = None
                continue
            if t.syl != cur_syl:
                flush()
                cur_syl = t.syl
            buf.append(t.label)
            if t.kind == KIND_V:
                buf.append(f"[{TONE_NAMES[t.tone]}|{WEIGHT_NAMES[t.weight]}]")
        flush()
        return " ".join(out)


def _pos(i: int, n: int) -> int:
    if n == 1:
        return POS_SINGLE
    if i == 0:
        return POS_INITIAL
    if i == n - 1:
        return POS_FINAL
    return POS_MEDIAL


class HausaFrontend:
    """Text -> Utterance.

    Parameters
    ----------
    tone_mode : 'auto' | 'all_marked' | 'none'
        auto        words with any tone diacritic are fully annotated (unmarked
                    vowels = High); unmarked words use the lexicon if available,
                    otherwise tone = UNKNOWN.
        all_marked  every word is assumed fully annotated; unmarked vowels = High.
        none        ignore tone diacritics; every vowel gets tone UNKNOWN.
    lexicon : optional ToneLexicon (or path to a lexicon file)
    loan_map : letter -> Hausa letter fallback for non-Hausa letters (default p->f, v->b)
    """

    def __init__(self, tone_mode: str = "auto", lexicon=None, loan_map=None, strict: bool = False):
        assert tone_mode in ("auto", "all_marked", "none")
        self.tone_mode = tone_mode
        if isinstance(lexicon, str):
            lexicon = ToneLexicon.from_file(lexicon)
        self.lexicon: Optional[ToneLexicon] = lexicon
        self.loan_map = loan_map
        self.strict = strict

    # -- word level ----------------------------------------------------------
    def _word_segments(self, word: str) -> List[Seg]:
        segs = parse_word(word, self.loan_map, self.strict)
        if self.tone_mode == "none":
            for s in segs:
                s.tone = None
            default = TONE_UNK
        elif self.tone_mode == "all_marked":
            default = TONE_H
        else:
            marked = any(s.tone is not None for s in segs)
            if not marked and self.lexicon is not None:
                ann = self.lexicon.lookup(word)
                if ann:
                    asegs = parse_word(ann, self.loan_map, self.strict)
                    sig = lambda ss: [(s.phone, s.kind) for s in ss]  # noqa: E731
                    if sig(asegs) == sig(segs):
                        for s, a in zip(segs, asegs):
                            s.long = s.long or a.long
                            if s.kind == "V":
                                s.tone = a.tone if a.tone is not None else TONE_H
                        marked = True
            default = TONE_H if marked else TONE_UNK
        for s in segs:
            if s.kind == "V" and s.tone is None:
                s.tone = default
        return segs

    # -- utterance level -----------------------------------------------------
    def __call__(self, text: str) -> Utterance:
        norm = normalize_text(text, self.strict)
        words: List[str] = []
        bound_after: List[int] = []
        for item in norm.split(" "):
            if item in ("", ):
                continue
            if item in (",", "|"):
                if words:
                    bound_after[-1] = max(bound_after[-1], B_MINOR)
            elif item == ".":
                if words:
                    bound_after[-1] = max(bound_after[-1], B_MAJOR)
            else:
                words.append(item)
                bound_after.append(B_WORD)

        # phrase membership
        phrase_of_word, p = [], 0
        for b in bound_after:
            phrase_of_word.append(p)
            if b >= B_MINOR:
                p += 1
        n_phrases = (phrase_of_word[-1] + 1) if words else 0
        words_in_phrase: Dict[int, List[int]] = {}
        for w, ph in enumerate(phrase_of_word):
            words_in_phrase.setdefault(ph, []).append(w)

        tokens: List[Token] = []
        syl_counter = 0
        for w, word in enumerate(words):
            segs = self._word_segments(word)
            if not segs:
                continue
            syl_of, role_of, weights = syllabify(segs)
            n_syl = len(weights)
            ph = phrase_of_word[w]
            wlist = words_in_phrase[ph]
            wpos = _pos(wlist.index(w), len(wlist))
            ppos = _pos(ph, n_phrases)
            for s, si, role in zip(segs, syl_of, role_of):
                kind = KIND_V if s.kind == "V" else KIND_C
                tokens.append(Token(
                    phone=s.phone, kind=kind, long=int(s.long),
                    tone=(s.tone if kind == KIND_V else TONE_NONE),
                    role=role, weight=weights[si],
                    syl=syl_counter + si, word=w, phrase=ph, boundary=B_NONE,
                    syl_pos=_pos(si, n_syl), word_pos=wpos, phrase_pos=ppos,
                ))
            syl_counter += n_syl
            if w < len(words) - 1:                       # pause token between words
                b = bound_after[w]
                tokens.append(Token(
                    phone=_PAUSE_FOR_BOUNDARY[b], kind=KIND_P, long=0, tone=TONE_NONE,
                    role=ROLE_NONE, weight=W_NONE, syl=-1, word=-1, phrase=ph,
                    boundary=b, syl_pos=POS_NONE, word_pos=POS_NONE, phrase_pos=ppos,
                ))
        return Utterance(text=norm, words=words, tokens=tokens)
