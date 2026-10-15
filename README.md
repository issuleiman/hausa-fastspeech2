# Hausa FastSpeech 2

A linguistically grounded extension of [FastSpeech 2](https://arxiv.org/abs/2006.04558) that explicitly models
**Hausa tone, vowel length, syllable weight and hierarchical prosody** for single- and multi-speaker synthesis.

> **Status: research code, verified on synthetic data only.** It has never been trained on real Hausa speech.
> No linguistic data is shipped (the tone lexicon is an empty template). See "What has and has not been verified".

Design rationale and the paper/repo mapping: [`docs/DESIGN.md`](docs/DESIGN.md).

## Quick start

```bash
pip install -r requirements.txt          # torch, librosa, pyworld, tgt, ...
python -m pytest -q tests                # 72 tests, ~15 s
python scripts/smoke_test.py             # end-to-end on a synthetic corpus, ~6 min on 1 CPU
bash scripts/fetch_vocoder.sh            # (already bundled) pretrained HiFi-GAN generator
```

### Real data

1. **Corpus layout**: `metadata.csv` (`id|speaker|text`, UTF-8) and `wavs/<speaker>/<id>.wav`.
2. **Audit annotation**: `python scripts/audit_corpus.py --metadata data/hausa_corpus/metadata.csv --tone_mode auto`
3. **Forced alignment** (Montreal Forced Aligner):
   `python scripts/prepare_mfa.py --corpus_dir data/hausa_corpus --out_dir data/mfa_input`, then train/align with MFA
   (check `mfa train --help` for your version's flags; I did not run MFA here). The TextGrid *phones* tier must end up in
   `alignments/<speaker>/<id>.TextGrid`. The dictionary comes from the same G2P as the TTS frontend, so labels match tokens.
4. `python scripts/preprocess.py -c configs/hausa_single.yaml` (or `hausa_multi.yaml`)
5. `python -m hausafs2.train -c configs/hausa_single.yaml` (`--resume output/.../last.pt` to continue)
6. ```bash
   python -m hausafs2.synthesize --ckpt output/hausa_single/last.pt --text "..." --speaker <name> \
       --speed 1.0 --pitch_ratio 1.0 --pitch_range 1.0 --plot --out out.wav
   ```

### Tone annotation (the key data question)
Standard Hausa spelling marks neither tone nor vowel length. `frontend.tone_mode`: `all_marked` (transcripts fully
annotated; unmarked vowel = High), `auto` (marked words annotated, unmarked words via `lexicon/hausa_tone_lexicon.tsv`
else UNKNOWN), `none`. Diacritics: grave = Low, circumflex = Falling, acute = High, macron or doubled letter = long.
Unknown tones are handled by tone dropout + a tone head, but accuracy is bounded by what can be inferred from text.

## Layout
`hausafs2/frontend` (normalise, G2P, syllables, hierarchy, lexicon, validator) · `hausafs2/model` (encoder,
hierarchical prosody, variance adaptor, decoder, loss) · `hausafs2/data` (audio, alignment, preprocess, dataset, MFA prep,
synthetic corpus) · `hausafs2/vocoder` (HiFi-GAN) · `configs/` · `scripts/` · `tests/` · `results/` (arm comparison) · `smoke_run/` (report, plots, wavs).

## Comparison with original FastSpeech 2 (synthetic data) - see [`docs/RESULTS.md`](docs/RESULTS.md)
Same code/data/seed/budget, evaluated on unseen words with true durations and *predicted* pitch/energy.
Mean of 2 seeds: original FS2 pitch RMSE 1.01 z; FS2 with tone/length inputs 0.57; this model with the old tone-dropout
default 0.59 (**no gain** - tone dropout was hurting); with dropout removed (single seed) **0.36** (about 36 % better than the
fair baseline), mel L1 0.370 vs 0.403. K=3 contours fix falling tones; downstep/skip/weight features fix H levels; the
hierarchical encoder showed no pitch gain. The default was changed accordingly (production configs: tone dropout 0).
**On plain-orthography text with unseen words no variant beats the no-tone baseline** - tone knowledge is the bottleneck.
Evidence is synthetic, ablations are single-seed, no listening tests.

## Kaggle experiment for the paper (resumable)
`kaggle/hausa_kaggle_experiment.py` is a single-cell runner: baselines + ablations + seeds + statistics, with all progress stored in a private
Hugging Face repo so a new Kaggle session continues where the last one stopped. Guide: [`docs/KAGGLE_GUIDE.md`](docs/KAGGLE_GUIDE.md).
Start with `MODE="smoke"`. Resume logic is tested offline (three simulated sessions); the real Hub / MMS-aligner / dataset calls are not.

## Real-data experiment (your machine)
`pip install datasets torchaudio && python scripts/hf_smoke_20.py --dataset suleiman2003/unified-hausa-speech --n 20 --out data/hf20`
downloads 20 clips, audits the text, aligns with MMS CTC (rough), preprocesses and runs the arm comparison. **Untested end to
end** (the authoring sandbox could not reach Hugging Face); 20 clips only check that the pipeline runs. The frontend was
checked on 25 real sentences from that dataset's preview: no unknown graphemes, but all vowels have unknown tone (no marks).

## What has and has not been verified
**Verified (synthetic pseudo-words, 2 speakers, 400 utterances, 1500 CPU steps):** 24/24 gating smoke checks
([`smoke_run/smoke_report.json`](smoke_run/smoke_report.json)); on **unseen words** H-L pitch gap 0.97 z (true ~1.2), F fall 2.15 z,
long/short vowel ratio 2.09 (true 2.0), downstep present, speaker F0 ratio 1.75 (true 1.75), speed/pitch/range controls as
specified. 72 tests pass; all CLI entry points were run end to end.

**Bugs this process found (fixed, with regression tests):** `tgt` drops silence intervals by default (misaligned every phone
after a pause); padded positions leaked into variance predictors/PostNet; `n_points=1` crashed; tone dropout silently cost
accuracy (default changed); my first smoke test only proved word memorisation (redesigned with open vocabulary).

**Known limitations**
* Tone-swap diagnostic still below its bar (+0.44 z vs 0.6, non-gating): contradicting the tone of a frequent word is only
  partly obeyed; tone input is used compositionally for unseen words, not as a hard override.
* Plain-orthography text: unseen words' tones cannot be inferred; a lexicon or tone-annotated transcripts are needed.
* **Not validated on real speech**; tone/weight co-occurrence and downstep are hypotheses encoded as features.
* Not modelled: tone sandhi, polar-tone suffixes, question intonation, number expansion, the r/ɽ contrast. Loan letters
  p->f, v->b are an assumption. Loanword clusters (e.g. "Botswana") syllabify approximately.
* No CWT pitch (paper's variant); K-point contour instead (`pitch.mode: frame` gives the paper's frame-level pitch).

## Credits / licences
FastSpeech 2 (Ren et al.), ming024/FastSpeech2 (MIT), HiFi-GAN (Kong et al., MIT; generator code adapted and the universal
generator checkpoint in `vocoder_assets/` is from ming024's repo). Check upstream licences before redistributing.
