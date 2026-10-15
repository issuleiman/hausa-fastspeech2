# Design notes

## 1. What was reviewed and what was taken from each source

| Source | What it contributed here |
|---|---|
| **FastSpeech 2 paper** (arXiv 2006.04558) | Overall architecture: FFT encoder/decoder (self-attention + 1-D conv FFN, kernels 9/1), variance adaptor with *parallel* duration / pitch / energy predictors trained with MSE on ground-truth targets (teacher forcing), pitch/energy quantised into 256 bins and embedded, duration in the log domain, F0 via WORLD (DIO + StoneMask) with unvoiced gaps interpolated, energy = L2 norm of STFT frame, Adam(0.9, 0.98, 1e-9) with Transformer warm-up schedule, forced-alignment durations (MFA) instead of a teacher model. |
| **ming024/FastSpeech2** | Practical details: MFA TextGrid → durations, per-phoneme pitch/energy (the paper's frame-level variant is not their default), Tacotron-2 PostNet, HiFi-GAN vocoding, speaker embedding added to encoder output for multi-speaker, `log(d+1)` duration target so zero-length pauses are representable, the pretrained HiFi-GAN generator weights (used here for the vocoder). |
| **rishikksh20/FastSpeech2** | Compact variance-predictor layout; confirmed the paper's CWT pitch is optional and that the reference implementations do not implement it. |
| **xcmyz/FastSpeech** (v1) | Length regulator and FFT block lineage; useful to confirm the FFT-block structure. |
| **HF papers page** | Abstract / citation context only. |

Not implemented from the paper: **CWT-based pitch spectrogram** (the paper's v3 pitch variant). The reference repos also skip it. It is replaced here by a K-point contour representation that targets the same problem (preserving pitch *shape*) in a way that is easier to tie to tone.

## 2. Hausa-specific modelling

### Linguistic facts the frontend encodes
* Three lexical tones: **High, Low, Falling** (HL). Tone-bearing unit = syllable nucleus.
* **Phonemic vowel length** (a vs aa …); diphthongs **ai, au** are treated as bimoraic nuclei.
* **Consonant length** (geminates) → length flag; a geminate closes the preceding syllable (sab.bi).
* Hooked consonants ɓ ɗ ƙ ƴ and digraph phonemes (sh, ts, fy, ky, gy, ƙy, kw, gw, ƙw).
* Syllables (C)V(V)(C) with **moraic weight**: light (CV) / heavy (CVV, CVC) / super-heavy.
* Falling tone is *typically* found on heavy syllables → reported by `validate.py` as a **warning**, not enforced.

### Model components
1. **Factored linguistic embeddings** (`model/encoder.py`): phone, length, **tone**, weight, role-in-syllable, boundary strength, syllable-in-word / word-in-phrase / phrase-in-utterance position, **downstep counter** — each its own small embedding, summed.
2. **Downstep counter**: number of L/F tones seen earlier in the current phrase (resets per phrase). Encodes the hypothesis that high tones are progressively lowered after lows (downdrift/downstep). Computed *inside the model* from whatever tone sequence it is given (annotated, dropped-out, or predicted).
3. **Hierarchical prosody encoder** (`model/hier_prosody.py`): phoneme states are pooled to syllable / word / phrase nodes, each level has its own FFT stack, results are broadcast back through zero-initialised projections (starts as identity). Pauses have no syllable/word but belong to a phrase.
4. **Tone-aware pitch contour** (`model/variance_adaptor.py`): pitch is predicted as **K points per phoneme** (default K=3) of *speaker-normalised* log-F0 and interpolated linearly to frames. A falling tone therefore keeps its slope inside one long vowel; K=1 reproduces phoneme-level pitch and `mode: frame` reproduces the paper's frame-level pitch.
5. **Explicit tone → pitch skip**: tone and downstep embeddings are also added directly to the pitch predictor input (`pitch_tone_skip`).
6. **Per-speaker normalisation** of log-F0 and energy: tones are realised relative to a speaker's register, so multi-speaker training shares one tone→pitch mapping; synthesis de-normalises with the target speaker's statistics.
7. **Optional tone dropout + tone head** (off by default): tones of random words are replaced with UNKNOWN in training and a probe predicts H/L/F; at inference `predict_tone=True` fills UNKNOWN tones from the probe (two-pass). Measured in `docs/RESULTS.md`: dropout *costs* accuracy on marked text and does not help unseen words in plain spelling, so the default is 0; use a small value only if you need to synthesise unmarked text for words seen in training.
8. **Pause tokens at every word boundary** (`sp`, `pm`, `sil`; duration may be 0) so pauses are modelled by the duration predictor identically in training and inference.
9. Optional **length-contrast loss** (default off) and diagnostics (`long/short ratio`, tone-probe accuracy).

### Tone annotation (the critical data question)
Standard Hausa orthography marks neither tone nor vowel length, so the frontend cannot derive them from plain text. Three supported routes (`frontend.tone_mode`):

* `all_marked` – your transcripts are fully annotated (unmarked vowel = High).
* `auto` – words carrying any tone diacritic are fully annotated; unmarked words are looked up in an optional **lexicon** (`lexicon/hausa_tone_lexicon.tsv`, shipped *empty*), else tone = UNKNOWN.
* `none` – ignore diacritics; rely entirely on the tone head.

The model can be trained with UNKNOWN tones, but tone accuracy at synthesis is then bounded by how well tone can be inferred from text. A tone lexicon or tone-annotated transcripts will matter more for naturalness than any architectural choice.

## 3. Assumptions and things deliberately NOT done
* **No linguistic data is shipped.** The lexicon is an empty template; I did not want to present unverified tone assignments as fact.
* **Not modelled**: tone sandhi, polar-tone suffixes (-n/-r linker), high-tone raising, intonational particles/question intonation, the tap/flap `r`/`ɽ` contrast (not written in the orthography), number/abbreviation expansion, vowel-initial glottal onset (left as onset-less syllable).
* The loan-letter mapping `p→f`, `v→b` is an **assumption** (override via `HausaFrontend(loan_map=…)`).
* Downstep and weight/tone co-occurrence are encoded as *features/hypotheses for the model to learn from data*, not as hard rules. Check them against your corpus with `scripts/audit_corpus.py`.
* Never validated on real Hausa speech (see README "What has and has not been verified").
