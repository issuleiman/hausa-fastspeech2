# Comparison with original-style FastSpeech 2

**Read this first.** All numbers come from a *synthetic* corpus (invented Hausa-like words, 2 speakers, 360 training
utterances, 1500 CPU steps per model, rendered by a crude harmonic synthesiser whose F0 follows known tone rules). They
show what the architecture *can* learn when the data contains tone/length/downstep. They say **nothing** about speech
quality or about real Hausa. Raw results: `results/results_seed*.json`; table: `python scripts/summarize_results.py`;
reproduce: `scripts/compare_arms.py`.

## Protocol
Same code, data, seed, step budget and optimiser for every arm; arms differ only by switches.
Evaluation uses **ground-truth durations but predicted pitch and energy** (so the pitch/energy predictors are really
tested; a teacher-forced mel loss would hide them), on `val` and on `test_novel` = 80 utterances made only of words that never
occur in training. Metrics: mel L1; frame-level RMSE of speaker-normalised log-F0 (z), overall and on vowel frames of each tone;
duration MAE (frames); predicted long/short vowel ratio (true ≈ 2.0). Lower is better except the ratio.

Arms: `fs2_baseline` = original recipe (phone ids only, phoneme-level pitch, no tone/length input);
`fs2_tone_length` = baseline + tone and vowel-length inputs (the usual "tone in the phone symbol" trick) -- the **fair
baseline**; `hausa_full` = everything.

## Main result: unseen words, mean of 2 seeds
| arm | mel L1 | pitch RMSE | H | L | F | dur MAE | long/short | params |
|---|---|---|---|---|---|---|---|---|
| fs2_baseline | 0.504 | 1.006 | 0.943 | 0.980 | 0.966 | 1.65 | 1.09 | 1.42M |
| fs2_tone_length | 0.403 | 0.570 | 0.584 | **0.171** | 0.804 | 0.65 | 2.00 | 1.42M |
| hausa_full, tone dropout 0.25 (*old default*) | 0.435 | 0.591 | 0.651 | 0.397 | 0.438 | 0.64 | 1.98 | 1.79M |

1. Original → tone/length-aware baseline is a big gain, but that is just *having tone and length information*; on
   synthetic data it is true by construction, so it is not evidence for the architecture.
2. The old default of the full model did **not** beat the fair baseline overall (pitch 0.591 vs 0.570, mel L1 worse). It was
   much better on falling tones but worse on H/L levels. Seed-to-seed spread of pitch RMSE is about 0.03.

## Diagnosis: tone dropout was costing accuracy (single seed 1234, unseen words)
| arm | mel L1 | pitch RMSE | H | L | F |
|---|---|---|---|---|---|
| fs2_tone_length | 0.400 | 0.583 | 0.621 | 0.160 | 0.810 |
| + K=3 contour only (`tone_length_K3`) | 0.399 | 0.486 | 0.614 | 0.164 | **0.366** |
| full, **no hierarchy**, no tone dropout | 0.385 | **0.357** | 0.332 | **0.157** | 0.200 |
| full, **no tone dropout** | **0.370** | 0.362 | 0.336 | 0.171 | **0.190** |
| full, tone dropout 0.05 | 0.392 | 0.398 | 0.401 | 0.212 | 0.246 |
| full, tone dropout 0.25 + 0.1 utt (old) | 0.438 | 0.606 | 0.719 | 0.353 | 0.461 |

* Pitch error rises monotonically with tone dropout (0.362 → 0.398 → 0.606).
* K=3 contours fix **falling tones** (F 0.81 → 0.37); the downstep counter / tone→pitch skip / weight and position features
  fix **H levels** (0.61 → 0.33).
* The hierarchical encoder shows **no pitch gain here** and at most a small mel gain (0.370 vs 0.385, one seed, inconclusive).
* Without dropout the full model is ~36 % better in pitch RMSE and ~8 % in mel L1 than the fair baseline. The ablation arms
  are **single-seed**; the effect is much larger than the ~0.03 seed spread, but it should be confirmed with more seeds.
* Parameter count is not matched (1.79M vs 1.42M); the no-hierarchy arm (1.42M) reaches the same pitch error, so the gain is not just size.

**Change made:** default `tone.dropout_word/dropout_utt` is now 0 in the model and production configs. The smoke config keeps 0.05 so
the tone-head path stays exercised.

## Plain-orthography text (tones hidden, filled by the tone head), unseen words
| arm | pitch RMSE | mel L1 |
|---|---|---|
| fs2_tone_length | 1.063 | 0.502 |
| full, no dropout | 1.044 | 0.497 |
| full, dropout 0.05 | 1.129 | 0.515 |
| full, dropout 0.25 (old) | 1.035 | 0.498 |
(for reference, the no-tone baseline scores 1.03)

When tone cannot be read from the text and the word is unseen, **every arm is about as bad as the model that never had tone
information** -- dropout and the tone head do not rescue it, because the tone of an unseen word is not recoverable from its
spelling. (Seen words are recoverable from memory; a lexicon does that better.) Consequence for real data written in plain
orthography, such as `suleiman2003/unified-hausa-speech`: tone knowledge is the bottleneck. Expect the tone-aware inputs to help
only after you add a tone lexicon or tone-annotated transcripts; the pitch-contour, weight and hierarchy parts are what remain.

## Limits of this evidence
Synthetic data built with the same rules the model is designed to capture; one language-like phonotactics; 2 speakers; tiny
training budget; ablations single-seed; no listening tests; no real Hausa. The honest summary: *the architecture can use tone,
length, downstep and contour information that is present in the data, and (after the dropout fix) does so better than a
tone-aware plain FastSpeech 2.* Whether that carries over to real recordings is untested.
