"""Hausa grapheme-to-phoneme parsing and syllabification.

Hausa orthography is nearly phonemic, so G2P is a deterministic parse:

* digraphs  sh ts fy ky gy ƙy kw gw ƙw  are single phones;
* doubled vowels (aa) or a macron (ā)    -> long vowel;
* ai, au                                  -> diphthong (bimoraic nucleus);
* doubled consonants (bb, ss, ...)        -> geminate (length flag on one phone);
* any other vowel sequence                -> hiatus (two syllables).

Syllable structure is (C)V(V)(C): the last consonant of an intervocalic cluster
is the onset of the following syllable, earlier consonants are codas. A geminate
is the onset of the following syllable *and* adds a mora to the preceding
syllable (so  sab.bi  has a heavy first syllable).

Moraic weight: nucleus = 1 mora (2 if long/diphthong) + 1 per coda
  light (1 mora, CV) | heavy (2 morae: CVV, CVC) | super-heavy (>=3: CVVC, CVCC)
"""
import unicodedata
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .symbols import (
    CONSONANTS, DIPHTHONGS, MACRON, TONE_MARKS,
    W_HEAVY, W_LIGHT, W_SUPER,
    ROLE_CODA, ROLE_NUCLEUS, ROLE_ONSET,
)

_VOWEL_LETTERS = set("aeiou")
_DIGRAPHS = {"sh", "ts", "fy", "ky", "gy", "kw", "gw", "ƙy", "ƙw"}
_SINGLE = {c for c in CONSONANTS if len(c) == 1}
# Approximate adaptation of loan-word letters not in the Hausa alphabet.
# This is an assumption (override via HausaFrontend(loan_map=...)).
DEFAULT_LOAN_MAP = {"p": "f", "v": "b"}


@dataclass
class Seg:
    phone: str
    kind: str                       # 'C' or 'V'
    long: bool = False              # long vowel / diphthong / geminate consonant
    tone: Optional[int] = None      # explicit tone id from a diacritic, else None


def split_marks(word: str) -> List[Tuple[str, str]]:
    """Split an NFD word into (base_char, combining_marks) units."""
    units: List[List[str]] = []
    for ch in unicodedata.normalize("NFD", word):
        if unicodedata.combining(ch) and units:
            units[-1][1] += ch
        else:
            units.append([ch, ""])
    return [(b, m) for b, m in units]


def _tone_of(marks: str) -> Optional[int]:
    for ch in marks:
        if ch in TONE_MARKS:
            return TONE_MARKS[ch]
    return None


def parse_word(word: str, loan_map: Optional[Dict[str, str]] = None,
               strict: bool = False) -> List[Seg]:
    """Parse one orthographic word into segments (consonants/vowels with length
    and any explicitly marked tone)."""
    loan_map = DEFAULT_LOAN_MAP if loan_map is None else loan_map
    units = split_marks(word)
    n = len(units)
    segs: List[Seg] = []

    def base(k):
        ch = units[k][0]
        return loan_map.get(ch, ch)

    def match_cons(k):
        """Return (phone, next_index) for the consonant starting at k, or None."""
        ch = base(k)
        if k + 1 < n and (ch + base(k + 1)) in _DIGRAPHS:
            return ch + base(k + 1), k + 2
        if ch in _SINGLE:
            return ch, k + 1
        return None

    i = 0
    while i < n:
        ch = base(i)
        marks = units[i][1]
        if ch in _VOWEL_LETTERS:
            tone = _tone_of(marks)
            long = MACRON in marks
            j = i + 1
            if j < n and base(j) == ch:                       # aa, ee, ii, oo, uu
                long = True
                tone = tone if tone is not None else _tone_of(units[j][1])
                j += 1
                phone = ch
            elif j < n and (ch + base(j)) in DIPHTHONGS:      # ai, au
                long = True                                   # bimoraic nucleus
                tone = tone if tone is not None else _tone_of(units[j][1])
                phone = ch + base(j)
                j += 1
            else:
                phone = ch
            segs.append(Seg(phone, "V", long, tone))
            i = j
            continue
        m = match_cons(i)
        if m is None:
            if strict:
                raise ValueError(f"Cannot parse grapheme {units[i][0]!r} in word {word!r}")
            warnings.warn(f"Skipping unknown grapheme {units[i][0]!r} in word {word!r}")
            i += 1
            continue
        phone, j = m
        long = False
        m2 = match_cons(j) if j < n else None
        if m2 is not None and m2[0] == phone:                  # geminate
            long = True
            j = m2[1]
        segs.append(Seg(phone, "C", long, None))
        i = j
    return segs


def syllabify(segs: List[Seg]):
    """Assign each segment to a (word-local) syllable and role, compute weights.

    Returns
    -------
    syl_of : list[int]   syllable index of each seg
    role_of: list[int]   ROLE_ONSET / ROLE_NUCLEUS / ROLE_CODA
    weights: list[int]   weight class per syllable (W_LIGHT / W_HEAVY / W_SUPER)
    """
    nuclei = [k for k, s in enumerate(segs) if s.kind == "V"]
    n = len(segs)
    if not nuclei:                       # no vowel: one nucleus-less pseudo syllable
        return [0] * n, [ROLE_CODA] * n, [W_LIGHT]

    n_syl = len(nuclei)
    syl_of = [0] * n
    role_of = [ROLE_ONSET] * n
    moras = [0] * n_syl
    for si, k in enumerate(nuclei):
        syl_of[k] = si
        role_of[k] = ROLE_NUCLEUS
        moras[si] = 2 if segs[k].long else 1

    # word-initial consonants -> onset of syllable 0
    for k in range(0, nuclei[0]):
        syl_of[k], role_of[k] = 0, ROLE_ONSET
    # intervocalic clusters
    for si in range(n_syl - 1):
        cluster = list(range(nuclei[si] + 1, nuclei[si + 1]))
        if not cluster:
            continue
        *codas, onset = cluster
        for k in codas:
            syl_of[k], role_of[k] = si, ROLE_CODA
            moras[si] += 1
        syl_of[onset], role_of[onset] = si + 1, ROLE_ONSET
        if segs[onset].long:             # geminate closes the previous syllable
            moras[si] += 1
    # word-final consonants -> coda of last syllable
    for k in range(nuclei[-1] + 1, n):
        syl_of[k], role_of[k] = n_syl - 1, ROLE_CODA
        moras[n_syl - 1] += 1

    weights = [W_LIGHT if m <= 1 else (W_HEAVY if m == 2 else W_SUPER) for m in moras]
    return syl_of, role_of, weights


def canonical_word(word: str) -> str:
    """Tone-free canonical form with explicit vowel length (long vowels doubled).

    Used as the key for the MFA dictionary and `.lab` transcripts, so that the
    aligner sees every distinct *segmental* form once.
    """
    out = []
    for seg_char, marks in split_marks(word):
        c = seg_char
        out.append(c + (c if (MACRON in marks and c in _VOWEL_LETTERS) else ""))
    return "".join(out)
