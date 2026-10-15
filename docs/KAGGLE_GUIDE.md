# Kaggle experiment guide

File to paste into ONE Kaggle cell: `kaggle/hausa_kaggle_experiment.py` (also delivered stand-alone).

## 0. Security first
The data-pipeline script you shared contains a hard-coded Hugging Face token. **Revoke it** (huggingface.co/settings/tokens) and
create a new *write* token. Never commit tokens to GitHub. In the runner either paste the new token into `HF_TOKEN`
(then do not share the notebook publicly) or, safer, add a Kaggle Secret named `HF_TOKEN` (Notebook -> Add-ons -> Secrets) and leave
the placeholder in the cell.

## 1. Put the code on GitHub
```bash
unzip hausa-fastspeech2.zip && cd hausa-fastspeech2
git init -b main
git add .
git commit -m "Hausa FastSpeech 2"
git remote add origin https://github.com/issuleiman/hausa-fastspeech2.git
git push -u origin main
git tag v1.0 && git push origin v1.0        # for the paper: run experiments from a TAG
```
* `vocoder_assets/generator_universal.pth.tar.zip` is 49.5 MB (under GitHub's 100 MB hard limit, over the 50 MB soft warning only on some
  clients). If you prefer, delete it and run `bash scripts/fetch_vocoder.sh` in the notebook instead.
* Private repo: create a fine-grained read-only token for that repo and put it in `GITHUB_TOKEN`.
* The cell clones `GITHUB_URL` at `GITHUB_REF`, records the **exact commit** in the experiment state, and later sessions check out that same
  commit (`PIN_TO_RECORDED_COMMIT=True`) so every session of one experiment runs identical code even if you keep pushing.
  Use a new `EXP_NAME` when you intentionally change the code.

## 2. Kaggle notebook settings
Settings: **Accelerator = GPU (T4 or P100)**, **Internet = On** (needs phone verification), Persistence not required.
Paste the file into a single code cell, fill the config block at the top:
`HF_TOKEN`, `HF_REPO_ID` (e.g. `yourname/hausa-fs2-interspeech`, created automatically as a *private* repo), `GITHUB_URL`, `GITHUB_REF`.

## 3. Run order
1. `MODE = "smoke"` -> 20 train + 6 test clips, tiny model, 2 arms, 60 steps (~15-25 min, mostly model downloads). This checks the
   token, GitHub clone, dataset access, MMS aligner download, alignment, preprocessing, training, evaluation, report, listening samples
   and the Hub uploads. **Do this first; several parts could not be tested where this code was written** (see section 7).
2. Look at the printed log, `report.md` and the Hub repo `exp/exp1/` folder. Fix config problems.
3. Change `EXP_NAME = "main1"`, `MODE = "full"`, optionally `FORCE_SPEAKERS = [...]` (see `speaker_table.csv` on the Hub / in the log),
   and run again.

## 4. How resume works (and what it protects against)
Everything that matters lives in your Hugging Face repo under `exp/<EXP_NAME>/`; Kaggle's disk is treated as disposable.

| Stage | Hub artifact | Checkpoint granularity |
|---|---|---|
| select subset | `subset.tar`, `selection.csv`, `speaker_table.csv` | once |
| alignment | `align.tar` (TextGrids + per-clip `progress.json`) | every 25 clips / 10 min |
| preprocess | `pre.tar` (mel, features, stats; not `raw/`) | once |
| each training run | `runs/<run_id>/last.pt` (model+optimizer+step) | every `ckpt_upload_min` (25 min) and at session end |
| finished run | `runs/<run_id>/result.json` (+`model.pt` for the two main arms) | per run |
| everything | `state.json` (stages done, runs done/running + step, commit, versions) | with each upload |

Start of every session: `state.json` is pulled (a download error aborts instead of being mistaken for "no state", so state is never
overwritten by accident); finished stages are skipped, missing local files are restored from the Hub, an interrupted run restores
`last.pt` and continues from its step. Training runs in segments of `seg` steps (1000), each segment reseeded so the batch order differs.

**Worst case loss:** up to `ckpt_upload_min` minutes of one training run, or up to 10 minutes of alignment. Run ids contain a hash of the
configuration, so changing steps/model settings creates new runs instead of silently reusing old results.

**Session limits:** the runner tracks the session budget (8.5 h interactive, 11 h when run through *Save & Run All*; override with
`SESSION_HOURS`). Before a segment that would not fit, it uploads everything and stops cleanly with
`SESSION BUDGET REACHED ... run this cell again`. If Kaggle kills the session abruptly, just start a new session and run the cell again.

**Laptop off:** interactive sessions can be dropped when the browser disconnects. Use **Save Version -> Save & Run All (Commit)**: it runs
in the background on Kaggle's side (up to 12 h) without your laptop. When it ends (finished, stopped or crashed) press it again; it continues.
Free GPU quota is about 30 h/week.

**Rate limits / failures:** uploads are one commit per upload batch; if the Hub rate-limits, uploads are *deferred* (training keeps going,
no one-hour sleep burning GPU quota) and retried later and at the end. A failing arm is logged and skipped; others continue.

**Storage:** each `last.pt` of the full-size model is ~400 MB. It is deleted from the Hub when a run completes. If you hit your Hub storage
limit, set `SQUASH_HISTORY = True` in the config block: after each finished run the repo history is squashed (old checkpoint versions are dropped; irreversible).

## 5. What the experiment contains
* **Baselines:** `fs2_original` (phone ids, phoneme-level pitch; ming024-style), `fs2_framepitch` (paper's frame-level pitch).
* **Proposed:** `hausa_full` = prosodic position/boundary features + syllable structure (weight, role, length) + K=3 pitch contour + hierarchical prosody encoder.
* **Cumulative ladder:** `ladder_pos` -> `ladder_syl` -> `ladder_contour` -> `hausa_full`.
* **Leave-one-out:** `loo_no_pos`, `loo_no_syl`, `loo_no_contour` (and `ladder_contour` = no hierarchy).
* **Isolated:** `contour_only`, `hier_only`; **sensitivity:** `sens_K5`.
* **Tone arms** (`fs2_tonelength`, `full_tone`, `full_tone_noskip`) are added **only if** >30 % of vowels carry tone information (needs `TONE_LEXICON_PATH` or tone-marked text).
  The dataset text has no tone marks, so by default they are off - tone cannot be claimed as a contribution without tone information.
* Per-speaker F0/energy normalisation is shared by all arms (not ablated).
* Metrics on a **text-disjoint** test set (ground-truth durations, predicted pitch/energy): mel L1, MCD-proxy (DCT cepstra of the log-mel, *not* WORLD-MCD),
  pitch RMSE (speaker-normalised log-F0), pitch correlation, duration MAE (frames), speaking-rate error. Seeds: 2 by default (add a third via `OVERRIDES`).
* Statistics: mean +- sd over seeds; **paired bootstrap** (5000 resamples over test utterances) of every arm vs `fs2_original` and vs `hausa_full`.
  Outputs: `results_by_arm.csv`, `paired_bootstrap_vs_reference.csv`, `results_table.tex`, `report.md`, `results_bars.png`.
* Listening-test material: `samples.tar` (HiFi-GAN renderings with *predicted* durations for `fs2_original` and `hausa_full`, plus ground truth). **MOS/ABX needs human
  listeners** (typically >= 20 listeners, randomised, attention checks); nothing here replaces that.

GPU time: measure it from the smoke run (`train_minutes` in `result.json`) and scale: total = arms x seeds x steps x time/step. With 12 arms x 2 seeds x 12k steps the
total is large; reduce with `OVERRIDES={"arms": [...], "steps": ...}`. Every run is independent, so partial results are valid.

## 6. About the data-pipeline script you attached (read, not used)
Ideas worth keeping that this runner also uses: state stored on the Hub and *merged* with local state at start; progress counters per input file
(`source_rows_consumed`) so a crash resumes mid-file; scanning what already exists on the Hub before doing work; uploads batched into few commits and
verified by listing the repo; retry with back-off and rate-limit detection; a persistent speaker registry.
Differences / things I noticed: (a) the token is hard-coded; (b) on a rate limit it sleeps ~61 min, which burns Kaggle GPU quota (the runner defers instead);
(c) it deduplicates on (text, speaker) only, so the *same sentence read by many speakers* ends up in train and test; the runner makes the test split **text-disjoint**;
(d) the final `sort_dataset_on_hub` reloads and re-pushes the entire dataset, which is heavy.
The dataset mixes sources (studio Bible recordings, crowd-sourced phone recordings): check `speaker_table.csv` and use `FORCE_SPEAKERS` to pick clean speakers.

## 7. Honest status: what is and is not verified
Verified offline (72 tests, plus a three-session simulation with empty disks between sessions: budget stop -> crash mid-training -> resume, asserting that finished
stages/runs are not redone, the checkpoint is restored and training continues from step 20/40): CTC Viterbi alignment, vocabulary fallback, selection (text-disjoint, gender-balanced),
state merging, Hub wrapper error semantics (stubbed network), aligner glue (stub model), git clone + commit pinning, whole pipeline on a synthetic corpus.

**Not verified (no network / GPU where this was written): the real `huggingface.co` calls, the MMS-1B-all download and its Hausa vocabulary, reading the real dataset shards,
GPU speed/memory, Kaggle's environment details (e.g. what `KAGGLE_KERNEL_RUN_TYPE` is in commit mode - hence `SESSION_HOURS`).** The smoke run exists to flush these out.
API names used from `huggingface_hub` were checked against version 2.2.

**Scientific limitations to state in the paper:** CTC alignments are rough (~20 ms, peaky; word gaps shorter than ~250 ms are absorbed into the preceding vowel, so pause
durations are underestimated; the 5 % least confident clips are dropped) - validate on a hand-checked sample or re-align with MFA; no tone/length annotation in the data;
MCD-proxy is not standard MCD; objective metrics measure predictor accuracy, not naturalness; earlier synthetic-data results do not transfer automatically.
