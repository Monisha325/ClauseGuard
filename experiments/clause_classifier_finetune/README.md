# Clause Classifier Fine-Tuning Experiment (Additive)

A genuinely fine-tuned transformer classifier, added **alongside**
ClauseGuard's live heading-lexicon + embedding-centroid classifier —
not a replacement. Nothing in `backend/classification/` is touched by
anything in this folder.

**Start here:** [`REPORT.md`](./REPORT.md) — full methodology, real
results, and honestly-stated limitations. [`RESUME_BULLETS.md`](./RESUME_BULLETS.md)
has portfolio-ready summary bullets using the real numbers below.

## TL;DR results

| | Fine-tuned DistilBERT | ClauseGuard stage-1-only baseline |
|---|---|---|
| Test accuracy | **97.9%** | 41.3% |
| Coverage (non-"Unclassified") | 100% | 42.9% |

3 classes only (Limitation of Liability, Governing Law/Jurisdiction,
Termination) — see `REPORT.md` §2 for why Indemnification and
Confidentiality were honestly excluded (no CUAD source data existed for
either). The baseline above is **stage 1 only**; stage 2 (embedding
centroid) requires a Voyage AI API key not available when this baseline
was run — see `REPORT.md` §5 for the full caveat.

## Folder contents

```
REPORT.md                 Full model-card-style writeup (read this first)
RESUME_BULLETS.md          Portfolio-ready summary bullets
scripts/
  prepare_data.py          Extracts + splits the CUAD-derived dataset
  clauseguard_finetune_experiment.ipynb
                            Colab notebook — the actual executed run
                            (real cell outputs included) that produced
                            the fine-tuned model and its test metrics
  eval_baseline_stage1.py  Runs ClauseGuard's real, unmodified stage-1
                            classifier on the same held-out test set
data/
  data_report.json          Real dataset statistics (per-category counts,
                            train/val/test split sizes)
  train.jsonl / val.jsonl / test.jsonl
                            The actual train/val/test splits used
  finetuned_test_metrics.json / finetuned_test_predictions.jsonl
                            Real output of the Colab fine-tuning run
  baseline_stage1_metrics.json
                            Real output of the stage-1 baseline eval
```

**Not included in this repo (regenerate locally, see below):**
- `data/CUADv1_raw.json` — the raw ~40MB CUAD v1 label file. Too large
  and license-encumbered to vendor into this repo; download it yourself
  (see below).
- The fine-tuned model checkpoint (`model.safetensors`, ~260MB) — not
  committed for size reasons. Re-run the notebook to regenerate it, or
  ask for the checkpoint directly.

## Reproducing this from scratch

1. Download CUAD v1's label file:
   ```bash
   curl -L https://github.com/TheAtticusProject/cuad/raw/main/data.zip -o cuad_data.zip
   unzip cuad_data.zip CUADv1.json
   mv CUADv1.json experiments/clause_classifier_finetune/data/CUADv1_raw.json
   ```
2. `python3 experiments/clause_classifier_finetune/scripts/prepare_data.py`
   — rebuilds `train.jsonl` / `val.jsonl` / `test.jsonl` / `data_report.json`.
3. Open `scripts/clauseguard_finetune_experiment.ipynb` in Google Colab
   (Runtime → GPU), run all cells, upload the 3 `.jsonl` files when
   prompted, download `outputs_bundle.zip` at the end.
4. `python3 experiments/clause_classifier_finetune/scripts/eval_baseline_stage1.py`
   — runs the real stage-1 baseline against the same test set (needs the
   repo's `backend/` on `sys.path`, handled automatically by the script).

## Data source & license

[CUAD v1](https://www.atticusprojectai.org/cuad) (Hendrycks, Burns, Chen,
Ball — *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review*,
NeurIPS 2021), licensed CC BY 4.0.
