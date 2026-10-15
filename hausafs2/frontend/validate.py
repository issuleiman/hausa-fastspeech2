"""Linguistic sanity checks on frontend output.

These are *diagnostics*, not hard constraints: Hausa has lexical exceptions and
annotation conventions vary, so everything here is reported as a message with a
severity level so you can audit a corpus before training.
"""
from collections import defaultdict
from typing import Dict, List, Tuple

from .symbols import (
    KIND_V, TONE_F, TONE_UNK, W_LIGHT, W_SUPER,
)

Message = Tuple[str, str]   # (level, text) with level in {"warn", "info"}


def check_utterance(utt) -> List[Message]:
    msgs: List[Message] = []
    by_word: Dict[int, list] = defaultdict(list)
    for t in utt.tokens:
        if t.kind != 2:
            by_word[t.word].append(t)

    n_unk = sum(1 for t in utt.tokens if t.kind == KIND_V and t.tone == TONE_UNK)
    n_v = sum(1 for t in utt.tokens if t.kind == KIND_V)
    if n_v and n_unk:
        msgs.append(("info", f"{n_unk}/{n_v} vowels have unknown tone (unmarked, not in lexicon)"))

    for w, toks in by_word.items():
        word = utt.words[w] if w < len(utt.words) else str(w)
        for t in toks:
            if t.kind == KIND_V and t.tone == TONE_F and t.weight == W_LIGHT:
                msgs.append(("warn", f"'{word}': falling tone on a light (CV) syllable; "
                                     "falling tone normally occurs on heavy syllables"))
            if t.kind == KIND_V and t.weight == W_SUPER:
                msgs.append(("info", f"'{word}': super-heavy syllable (>=3 morae) is rare in Hausa"))
        if not any(t.kind == KIND_V for t in toks):
            msgs.append(("warn", f"'{word}': no vowel found; treated as a nucleus-less syllable"))
    return msgs


def corpus_report(utts) -> Dict[str, int]:
    """Aggregate counts over many utterances (useful before training)."""
    stats = defaultdict(int)
    for u in utts:
        for t in u.tokens:
            if t.kind == KIND_V:
                stats[f"tone_{t.tone}"] += 1
                stats["long_vowels" if t.long else "short_vowels"] += 1
                stats[f"weight_{t.weight}"] += 1
        for lvl, _ in check_utterance(u):
            stats[f"msg_{lvl}"] += 1
    return dict(stats)
