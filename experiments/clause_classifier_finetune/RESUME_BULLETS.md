# Resume Bullets — Fine-Tuning Experiment

Real numbers only, from `data/finetuned_test_metrics.json` and
`data/baseline_stage1_metrics.json` (see `REPORT.md` for full methodology
and caveats — especially that the baseline below is stage-1-only, not
ClauseGuard's complete live classifier).

- Fine-tuned DistilBERT on 955 real labeled contract clauses (sourced and
  filtered from the CUAD legal-NLP benchmark) for 3-class clause
  classification, achieving **97.9% test accuracy** (macro F1 0.97) on a
  189-example held-out test set split at the document level to prevent
  leakage.

- Built an end-to-end, reproducible fine-tuning pipeline (data
  extraction/filtering/contract-level splitting, HuggingFace
  `transformers`/`datasets` training, held-out evaluation) as an additive
  experiment layered on an existing production heuristic classifier,
  without modifying any live code.

- Ran an honest before/after evaluation against the existing system's
  real (unmodified) rule-based classifier on the same test set, correctly
  identifying and reporting the rule-based approach's actual
  precision/recall trade-off (near-100% precision, 43% coverage) rather
  than overstating the fine-tuned model's improvement over a fully
  equivalent baseline.
