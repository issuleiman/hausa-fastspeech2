import warnings

import pytest

from hausafs2.frontend import HausaFrontend, ToneLexicon
from hausafs2.frontend import symbols as S
from hausafs2.frontend.g2p import canonical_word, parse_word, syllabify
from hausafs2.frontend.normalize import normalize_text
from hausafs2.frontend.validate import check_utterance


def weights(word):
    segs = parse_word(word)
    _, _, w = syllabify(segs)
    return w


def test_ascii_hooked_letter_fallback():
    assert normalize_text("'Yan 'ba 'da 'ka") == normalize_text("ƴan ɓa ɗa ƙa")


def test_glottal_stop_between_vowels():
    assert "ʔ" in normalize_text("ta'azzara")


def test_digraphs_are_single_phones():
    ph = [s.phone for s in parse_word("shaatsakwaɗo")]
    assert ph[0] == "sh" and "ts" in ph and "kw" in ph


def test_long_vowel_by_doubling_and_macron():
    a, b = parse_word("baa")[1], parse_word("ba\u0304")[1]
    assert a.long and b.long and a.phone == b.phone == "a"
    assert not parse_word("ba")[1].long


def test_diphthong_is_bimoraic_heavy():
    segs = parse_word("kai")
    assert [s.phone for s in segs] == ["k", "ai"] and segs[1].long
    assert weights("kai") == [S.W_HEAVY]


@pytest.mark.parametrize("word,expected", [
    ("sarki", [S.W_HEAVY, S.W_LIGHT]),       # sar.ki  (CVC.CV)
    ("gida", [S.W_LIGHT, S.W_LIGHT]),        # gi.da
    ("mutum", [S.W_LIGHT, S.W_HEAVY]),       # mu.tum
    ("sabbi", [S.W_HEAVY, S.W_LIGHT]),       # geminate closes first syllable
    ("saako", [S.W_HEAVY, S.W_LIGHT]),       # long vowel
    ("kaanta", [S.W_SUPER, S.W_LIGHT]),      # kaan.ta: long vowel + coda = 3 morae
])
def test_syllable_weights(word, expected):
    assert weights(word) == expected


def test_geminate_is_single_long_phone():
    segs = parse_word("sabbi")
    assert [s.phone for s in segs] == ["s", "a", "b", "i"] and segs[2].long


def test_hiatus_gives_two_syllables():
    assert len(weights("kue")) == 2 and len(weights("kuea")) == 3


def test_tone_modes():
    marked = HausaFrontend("auto")("mùtûm gída")
    tones = [t.tone for t in marked.tokens if t.kind == S.KIND_V]
    assert tones == [S.TONE_L, S.TONE_F, S.TONE_H, S.TONE_H]
    unmarked = HausaFrontend("auto")("mutum")
    assert all(t.tone == S.TONE_UNK for t in unmarked.tokens if t.kind == S.KIND_V)
    allm = HausaFrontend("all_marked")("mutum")
    assert all(t.tone == S.TONE_H for t in allm.tokens if t.kind == S.KIND_V)
    none = HausaFrontend("none")("mùtûm")
    assert all(t.tone == S.TONE_UNK for t in none.tokens if t.kind == S.KIND_V)


def test_lexicon_overrides_unmarked_words_only():
    lex = ToneLexicon({"mutum": "mùtûm"})
    fe = HausaFrontend("auto", lexicon=lex)
    t = [x.tone for x in fe("mutum").tokens if x.kind == S.KIND_V]
    assert t == [S.TONE_L, S.TONE_F]
    t2 = [x.tone for x in fe("mútúm").tokens if x.kind == S.KIND_V]      # explicit marks win
    assert t2 == [S.TONE_H, S.TONE_H]


def test_lexicon_with_mismatched_segments_is_ignored():
    fe = HausaFrontend("auto", lexicon=ToneLexicon({"mutum": "gìda"}))
    assert all(t.tone == S.TONE_UNK for t in fe("mutum").tokens if t.kind == S.KIND_V)


def test_pause_tokens_and_hierarchy():
    u = HausaFrontend("auto")("gida ruwa, kai. sarki")
    pauses = [t for t in u.tokens if t.kind == S.KIND_P]
    assert [p.phone for p in pauses] == ["sp", "pm", "sil"]
    assert [p.boundary for p in pauses] == [S.B_WORD, S.B_MINOR, S.B_MAJOR]
    assert len({t.phrase for t in u.tokens}) == 3                   # phrases split at , and .
    assert all(t.syl == -1 and t.word == -1 for t in pauses)
    words = {t.word for t in u.tokens if t.kind != S.KIND_P}
    assert words == {0, 1, 2, 3}
    # no leading/trailing pause tokens
    assert u.tokens[0].kind != S.KIND_P and u.tokens[-1].kind != S.KIND_P


def test_positional_features():
    u = HausaFrontend("auto")("gida")
    syl_pos = [t.syl_pos for t in u.tokens]
    assert syl_pos[0] == S.POS_INITIAL and syl_pos[-1] == S.POS_FINAL
    assert u.tokens[0].word_pos == S.POS_SINGLE and u.tokens[0].phrase_pos == S.POS_SINGLE


def test_labels_for_alignment():
    u = HausaFrontend("auto")("saako kai")
    assert u.labels == ["s", "aː", "k", "o", "sp", "k", "ai"]


def test_canonical_word_strips_tone_and_doubles_macron():
    assert canonical_word("mùtûm") == "mutum"
    assert canonical_word("ba\u0304") == "baa"


def test_validator_flags_falling_on_light():
    u = HausaFrontend("auto")("sâ")          # falling tone on a light CV
    assert any(l == "warn" and "falling" in m for l, m in check_utterance(u))
    u2 = HausaFrontend("auto")("kâi")
    assert not any(l == "warn" for l, _ in check_utterance(u2))


def test_unknown_characters_warn_or_raise():
    with pytest.raises(ValueError):
        HausaFrontend(strict=True)("ba3a")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert len(HausaFrontend()("ba3a")) > 0
