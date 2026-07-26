# LLM Distillation Experiment (In Progress)

An **additive** experiment exploring whether ClauseGuard's clause-risk-flagging
task — currently served by a 70B teacher model (`llama-3.3-70b-versatile` via
Groq) — could be distilled into a small, open-weight, self-hostable student
(`Qwen2-0.5B`) via PEFT/LoRA. Nothing here touches or replaces
`backend/agent/flag_clause.py`; the live system still calls the real 70B
model.

This folder is a **snapshot of a real, in-progress experiment**, not a
finished result. In keeping with this project's stated preference for
honest, reproducible reporting over inflated claims, here's exactly what's
done and what isn't.

## Status

| Phase | Status |
|---|---|
| Select a stratified clause subset for distillation | ✅ Done (`select_distillation_subset.py`) — 450 clauses sampled proportionally from the classifier experiment's existing train/val/test splits, preserving category balance and contract-level grouping |
| Generate real teacher labels (severity/explanation/citation) | ✅ Done, partially — `generate_teacher_labels.py` calls the actual unmodified `flag_clause()` → Groq 70B pipeline; a real run is checkpointed in `data/teacher_labels.jsonl`-derived splits, constrained by Groq's free-tier daily token budget (see script docstring for the exact math) |
| Build the final train/val/test split from surviving labeled rows | ✅ Done (`finalize_distillation_set.py`) — 107 train / 24 val / 22 test rows (153 total), grouped by contract, re-split from whatever rows actually finished labeling before the daily quota was hit |
| Verify the LoRA/PEFT training pipeline mechanics | ✅ Done, on synthetic data only — `smoke_test_lora_pipeline.py` runs tokenization, adapter attachment, a forward/backward pass, and a save/load round-trip against a **tiny random-weight** Qwen2 architecture, entirely offline. This proves the plumbing doesn't break; it proves nothing about model quality |
| Fine-tune the real Qwen2-0.5B student on the real 153-row set | ⬜ Not yet run in this repo snapshot (needs a GPU — the classifier experiment's notebook used Colab for the same reason; this experiment would follow the same pattern) |
| Evaluation harness: benchmark student vs. teacher outputs | ⬜ Not yet built |

## Why the dataset is small (153 rows, not 450)

Groq's free tier caps daily token throughput well below what 450 clauses at
~400-900 tokens/call would need in one day (see `select_distillation_subset.py`'s
docstring for the exact rate-limit math). The real labeling run is
checkpointed and resumable, but only a subset of the planned 450 clauses had
finished labeling — with `label_status="ok"` — by the time this snapshot was
taken. `finalize_distillation_set.py` builds a clean, contract-grouped
70/15/15 split from exactly those survivors, rather than reusing the
original (now-inaccurate) split assignments.

## Folder contents

```
scripts/
  select_distillation_subset.py   Stratified 450-clause subset selection
  generate_teacher_labels.py       Calls the real flag_clause()/Groq pipeline (run locally, not in CI)
  finalize_distillation_set.py     Builds the final contract-grouped train/val/test split
  smoke_test_lora_pipeline.py      Offline plumbing test (tiny synthetic Qwen2, no real weights)
data/
  distillation_input_clauses.jsonl The 450-clause subset sent to the teacher
  final_train.jsonl / final_val.jsonl / final_test.jsonl
                                    The 153 real teacher-labeled rows, split
```

## Reproducing / continuing this experiment

1. From the project root, with the backend's dependencies installed and a
   real `.env` (containing a valid `GROQ_API_KEY`) at the project root:
   ```bash
   python experiments/llm_distillation/scripts/select_distillation_subset.py
   python experiments/llm_distillation/scripts/generate_teacher_labels.py
   python experiments/llm_distillation/scripts/finalize_distillation_set.py
   ```
   `generate_teacher_labels.py` paces itself against Groq's free-tier daily
   quota and checkpoints progress — see its docstring before running.
2. Run `scripts/smoke_test_lora_pipeline.py` locally to confirm the
   training plumbing works in your environment before spending real GPU
   time.
3. The remaining work — fine-tuning the real Qwen2-0.5B student on
   `data/final_train.jsonl` and building the teacher-vs-student evaluation
   harness — needs a GPU (a free Colab T4 is sufficient, matching the
   approach used in `experiments/clause_classifier_finetune/`). PRs welcome.
