# ClauseGuard — Additive Fine-Tuned Clause Classifier Experiment

**Status: additive experiment only.** Nothing in `backend/classification/`
was modified. This lives entirely in `experiments/clause_classifier_finetune/`.

## 1. Motivation

ClauseGuard's live classifier is a two-stage heuristic: heading-lexicon
pattern matching, then an embedding-centroid fallback. Neither stage
involves a trained model. This experiment asks: does a genuinely
fine-tuned transformer classifier do better, on real data, than that
heuristic — and by how much?

## 2. Data audit (Phase 1) — why this experiment covers 3 classes, not 5

ClauseGuard's live classifier recognizes 5 categories. The only real,
labeled clause text found anywhere in the ClauseGuard repo was 23
examples embedded in `backend/classification/centroid_fallback.py`
(used as that classifier's own centroid-bootstrap set) — an order of
magnitude below the ~50-100/class usually needed for real fine-tuning
signal. No unlabeled contracts existed in the repo to hand-label further.

Per an explicit decision, this experiment supplements with **CUAD**
(Contract Understanding Atticus Dataset v1, Hendrycks et al. 2021, CC BY
4.0, 510 real contracts). CUAD's 41 official categories, however, only
map cleanly onto **3 of ClauseGuard's 5** categories:

| ClauseGuard category | CUAD category used | Real official CUAD label? |
|---|---|---|
| Limitation of Liability | Cap On Liability | Yes |
| Governing Law / Jurisdiction | Governing Law | Yes |
| Termination | Termination For Convenience | Yes, but **narrower** — CUAD's category excludes general termination-for-cause/notice clauses that ClauseGuard's own "Termination" covers |
| Indemnification | — | **No CUAD category exists at all** |
| Confidentiality | — | **No CUAD category exists at all** |

Per an explicit decision ("b1"), **Indemnification and Confidentiality
were dropped from this experiment** rather than force a mapping or
fabricate labels for them. This is a real, disclosed scope reduction:
this experiment's results say nothing about those 2 categories.

## 3. Data preparation

- Source: `data/CUADv1_raw.json` (real CUAD v1 SQuAD-style answer spans).
- Extraction: all non-impossible answer spans for the 3 mapped categories,
  exact-duplicate-collapsed, spans under 40 characters dropped (too short
  to be a meaningful classification example on their own).
- Split **by contract** (not by row), so no contract's text leaks across
  train/val/test:

| Split | Limitation of Liability | Governing Law / Jurisdiction | Termination | Total | Contracts |
|---|---|---|---|---|---|
| Train | 484 | 310 | 161 | 955 | 310 |
| Val | 92 | 67 | 47 | 206 | 66 |
| Test | 88 | 69 | 32 | 189 | 68 |

Script: `scripts/prepare_data.py`.

## 4. Fine-tuning

- Model: `distilbert-base-uncased`, standard `AutoModelForSequenceClassification`
  head, 3 classes.
- Run in Google Colab (free T4 GPU) — this dev sandbox's network allowlist
  blocks `huggingface.co`, so the actual training run happened where real
  internet access exists. Notebook: `scripts/clauseguard_finetune_experiment.ipynb`.
- 4 epochs, batch size 16, lr 2e-5, max seq length 256, best checkpoint by
  macro-F1 on val.
- All numbers below are the **real output** of that run
  (`data/finetuned_test_metrics.json`, `data/finetuned_test_predictions.jsonl`),
  not estimated.

### Fine-tuned model — held-out TEST results (n=189)

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Limitation of Liability | 0.967 | 1.000 | 0.983 | 88 |
| Governing Law / Jurisdiction | 1.000 | 0.971 | 0.985 | 69 |
| Termination | 0.968 | 0.938 | 0.952 | 32 |
| **Accuracy** | | | **0.979** | 189 |

Confusion matrix (rows=true, cols=predicted; order LoL / GovLaw / Termination):
```
[88,  0,  0]
[ 1, 67,  1]
[ 2,  0, 30]
```
4 total errors, all near category boundaries (Termination↔Liability), no
systematic collapse into one class.

## 5. Baseline: ClauseGuard's real stage-1 classifier on the same test set

Script: `scripts/eval_baseline_stage1.py`. Imports
`backend/classification/heading_match.py`'s real, **unmodified**
`classify_heading()` directly.

**Only stage 1 was run, not the full two-stage cascade** — stage 2
(`classification/centroid_fallback.py`) requires a live Voyage AI API
call (`VOYAGE_API_KEY`), unavailable in this environment. Per explicit
user decision, this is reported as a **partial, stage-1-only baseline**,
not the full live system's real accuracy. This is the single most
important caveat in this report: **the comparison below is fine-tuned
DistilBERT vs. half of the live system**, not vs. the complete live
system.

**Second disclosed adaptation:** `classify_heading()` needs a
`heading_path` string that ClauseGuard's real chunker derives from
document structure. CUAD's test spans have no such metadata, so
`heading_path` was approximated as the real CUAD contract text
immediately preceding each answer span (e.g. `"6.9   Governing Law."`).
This is a reasonable proxy, not identical to ClauseGuard's real chunker
output.

### Stage-1-only baseline — held-out TEST results (same 189 examples)

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Limitation of Liability | 1.000 | 0.227 | 0.370 | 88 |
| Governing Law / Jurisdiction | 0.979 | 0.667 | 0.793 | 69 |
| Termination | 0.857 | 0.375 | 0.522 | 32 |
| **Accuracy (Unclassified scored as wrong)** | | | **0.413** | 189 |

**Coverage:** stage 1 left **108/189 (57.1%)** of clauses `Unclassified`
— i.e. it never guessed for most of the test set at all.

Confusion matrix (rows=true; cols = Unclassified / LoL / GovLaw / Termination):
```
[65, 20,  1,  2]
[23,  0, 46,  0]
[20,  0,  0, 12]
```

**What this actually shows:** stage 1's precision is near-perfect (1.00 /
0.98 / 0.86) — when it does commit to a category, it's almost never
wrong. Its problem is **recall/coverage**, not accuracy-when-confident:
by design, it returns `None` rather than guess on any heading it doesn't
recognize, and CUAD's real contract headings ("21. Law application")
often don't match the lexicon's hand-built phrase list. This is exactly
the gap ClauseGuard's own stage 2 (embedding centroid) exists to close —
which is precisely the part this evaluation could not run.

## 6. Honest comparison and conclusion

| | Fine-tuned DistilBERT | ClauseGuard stage-1-only |
|---|---|---|
| Accuracy | 0.979 | 0.413 |
| Coverage (non-"Unclassified") | 100% | 42.9% |

The fine-tuned model substantially outperforms **stage 1 alone**. It
does **not** have a measured comparison against ClauseGuard's real,
complete two-stage live system, because stage 2 could not be executed
here (missing `VOYAGE_API_KEY`). It would be overstating the evidence to
claim the fine-tuned model "beats ClauseGuard" — it beats the specific,
disclosed, partial baseline that was actually measurable in this
environment.

**Limitations, stated plainly:**
- Only 3 of ClauseGuard's 5 live categories are covered (no Indemnification
  or Confidentiality data existed to fine-tune on honestly).
- "Termination" here is CUAD's narrower "Termination For Convenience"
  subtype, not ClauseGuard's full termination-clause definition.
- The baseline is stage-1 only; stage 2's real contribution is unmeasured.
- `heading_path` for the baseline was approximated from raw contract text,
  not produced by ClauseGuard's actual chunker.
- Test set is CUAD contracts (SEC filings, mostly historical, English-only,
  M&A-context wording) — may not represent ClauseGuard's real target
  contract population.
- n=189 test examples across 3 classes is a real, usable size for this
  binary/multi-class task, but still modest, especially for Termination
  (support=32).

## 7. Reproducing this experiment

```
scripts/prepare_data.py                  # rebuilds train/val/test.jsonl from CUADv1_raw.json
scripts/clauseguard_finetune_experiment.ipynb   # run in Colab (GPU) to fine-tune + get real test metrics
scripts/eval_baseline_stage1.py           # runs the real stage-1 classifier on the same test set
```
