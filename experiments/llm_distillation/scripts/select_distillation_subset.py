"""Phase 2(A) — select a ~450-clause stratified subset of the CUAD-derived
clause dataset (experiments/clause_classifier_finetune/data/{train,val,test}
.jsonl) to send through the REAL flag_clause()/Groq pipeline as distillation
input.

WHY 450, NOT ALL 1,350 (user's explicit choice, "Medium" scale option):
Groq's free tier for llama-3.3-70b-versatile is 30 RPM / 1,000 RPD / 12K TPM
/ 100K TPD (see backend/agent/flag_clause.py's own module docstring). The
BINDING constraint is TPD, not RPD: each flag_clause() call is roughly
400-900 total tokens (prompt + completion), so the realistic daily ceiling
is ~120-220 calls/day, not 1,000. 450 clauses is expected to take ~2-3 days
of free-tier quota to fully label -- see generate_teacher_labels.py's own
docstring for how that script paces itself and checkpoints across days.

SAMPLING METHOD (proportional stratified, by split AND category):
Each of train/val/test.jsonl is sampled independently at a fixed ~1/3 rate,
stratified by category so each split's original category balance is
preserved in the subset. Splits are NOT touched or reshuffled -- this
script only takes a sub-sample WITHIN each existing split file, so the
prior experiment's contract-level train/val/test leakage discipline
(clauses split by contract, never by row) is automatically inherited
unchanged; sampling rows out of an already-contract-clean split cannot
introduce contract leakage.

Target counts below were computed by taking round(category_count / 3) per
split (see this repo's own data_report.json for the source category
counts: train 955 = 310/484/161, val 206 = 67/92/47, test 189 = 69/88/32),
giving 318/69/63 = 450 total.

Fixed random seed (42) for reproducibility -- rerunning this script
produces the exact same subset.
"""

import json
import random
from pathlib import Path

SEED = 42
DATA_DIR = Path(__file__).resolve().parents[2] / "clause_classifier_finetune" / "data"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "distillation_input_clauses.jsonl"

# round(category_count / 3) per split, derived from data_report.json
TARGET_PER_SPLIT_CATEGORY = {
    "train": {
        "Governing Law / Jurisdiction": 103,
        "Limitation of Liability": 161,
        "Termination": 54,
    },
    "val": {
        "Governing Law / Jurisdiction": 22,
        "Limitation of Liability": 31,
        "Termination": 16,
    },
    "test": {
        "Governing Law / Jurisdiction": 23,
        "Limitation of Liability": 29,
        "Termination": 11,
    },
}


def load_split(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_split(split: str, rows: list[dict], rng: random.Random) -> list[dict]:
    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)

    targets = TARGET_PER_SPLIT_CATEGORY[split]
    sampled = []
    for category, target_n in targets.items():
        pool = by_category.get(category, [])
        if len(pool) < target_n:
            raise ValueError(
                f"split={split} category={category!r} has only {len(pool)} "
                f"rows, need {target_n}"
            )
        sampled.extend(rng.sample(pool, target_n))
    return sampled


def main() -> None:
    rng = random.Random(SEED)
    all_selected = []

    for split in ("train", "val", "test"):
        rows = load_split(split)
        selected = sample_split(split, rows, rng)
        for i, row in enumerate(selected):
            all_selected.append(
                {
                    "distill_id": f"{split}_{i:04d}",
                    "text": row["text"],
                    "category": row["category"],
                    "contract": row["contract"],
                    "split": split,
                    "source_label_id": row["label_id"],
                }
            )
        print(f"{split}: sampled {len(selected)} / {len(rows)} rows")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in all_selected:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(all_selected)} rows to {OUT_PATH}")
    print(
        "NOTE: these are INPUT CLAUSES ONLY, no severity/explanation/"
        "citation yet -- run generate_teacher_labels.py against this file "
        "next to actually call flag_clause()/Groq and produce teacher labels."
    )


if __name__ == "__main__":
    main()
