"""
Data prep for the clause-classifier fine-tuning experiment.

SCOPE (see Phase 1 data audit + user decision "b1"):
ClauseGuard's live classifier recognizes 5 categories, but only 3 of them
have a real, officially-labeled counterpart in CUAD v1's fixed 41-category
schema:

  ClauseGuard category              -> CUAD category
  ---------------------------------------------------------------
  "Limitation of Liability"         -> "Cap On Liability"
  "Governing Law / Jurisdiction"     -> "Governing Law"
  "Termination"                      -> "Termination For Convenience"
                                         (NOTE: this is a narrower subtype
                                         of ClauseGuard's own "Termination"
                                         category, which also covers
                                         termination-for-cause/notice
                                         clauses. This is a real scope
                                         narrowing, not a clean 1:1 match --
                                         documented here and again in the
                                         model card.)

"Indemnification" and "Confidentiality" have NO officially-labeled CUAD
category (confirmed by enumerating all 41 category names in CUADv1.json --
see chat history / model card for the full list). Per the user's explicit
"b1" decision, this experiment is SCOPED TO THESE 3 CLASSES ONLY. This is
a real, honest reduction in scope from ClauseGuard's live 5-category
classifier, not a simplification hidden for convenience.

SOURCE: CUADv1.json (SQuAD-style answer spans over 510 real commercial
contracts), CC BY 4.0, Atticus Project / Hendrycks et al. 2021.
Downloaded from https://github.com/TheAtticusProject/cuad (raw/main/data.zip)
and saved verbatim as data/CUADv1_raw.json for reproducibility.

WHAT THIS SCRIPT DOES (no fabrication -- every row below is a real CUAD
answer span, not synthetic text):
  1. Loads CUADv1_raw.json.
  2. For each of the 3 target CUAD categories, pulls every non-impossible
     answer span (qa['answers'][i]['text']) across all 510 contracts.
  3. Deduplicates exact-duplicate spans (some contracts have boilerplate
     reused near-verbatim; exact dupes are collapsed to avoid double-
     counting the same sentence as if it were independent signal).
  4. Drops spans under MIN_CHARS -- CUAD's span-level annotation means many
     "answers" are sub-sentence fragments (e.g. a single defined term),
     which are real, correctly-labeled spans but too short to be a
     meaningful classification example on their own.
  5. Splits into train/val/test with a document-level-aware stratified
     split (see NOTE below on why document-level matters).
  6. Writes train.jsonl / val.jsonl / test.jsonl + a data_report.json
     with the real resulting counts (no estimates).

NOTE on document-level leakage: CUAD contracts often contain near-duplicate
boilerplate reused within the SAME contract (e.g. a clause repeated in an
amendment section). To avoid a split where train and test share
near-identical text from the same contract (which would inflate test
accuracy), the split is done by CONTRACT (title), not by row: all spans
from a given contract go entirely into train, val, or test.
"""
import json
import re
import random
import hashlib
from collections import defaultdict, Counter
from pathlib import Path

RAW_PATH = Path(__file__).parent.parent / "data" / "CUADv1_raw.json"
OUT_DIR = Path(__file__).parent.parent / "data"

# ClauseGuard label -> CUAD official category name
CATEGORY_MAP = {
    "Limitation of Liability": "Cap On Liability",
    "Governing Law / Jurisdiction": "Governing Law",
    "Termination": "Termination For Convenience",
}
LABELS = list(CATEGORY_MAP.keys())
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

MIN_CHARS = 40  # drop sub-sentence fragments (see docstring)
SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def clean_text(t: str) -> str:
    t = t.strip()
    t = re.sub(r"\s+", " ", t)
    return t


def load_raw_examples():
    """Returns list of dicts: {contract, category, text}. Real spans only."""
    data = json.loads(RAW_PATH.read_text())["data"]
    cuad_to_cg = {v: k for k, v in CATEGORY_MAP.items()}
    rows = []
    for doc in data:
        title = doc["title"]
        for para in doc["paragraphs"]:
            for qa in para["qas"]:
                cuad_cat = qa["id"].split("__")[-1]
                if cuad_cat not in cuad_to_cg:
                    continue
                if qa.get("is_impossible", True):
                    continue
                for ans in qa.get("answers", []):
                    text = clean_text(ans["text"])
                    if len(text) < MIN_CHARS:
                        continue
                    rows.append({
                        "contract": title,
                        "category": cuad_to_cg[cuad_cat],
                        "text": text,
                    })
    return rows


def dedup_exact(rows):
    seen = set()
    out = []
    for r in rows:
        h = hashlib.sha256((r["category"] + "||" + r["text"]).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(r)
    return out


def split_by_contract(rows, seed=SEED):
    """Group rows by contract; assign whole contracts to train/val/test so
    no contract's text appears in more than one split."""
    by_contract = defaultdict(list)
    for r in rows:
        by_contract[r["contract"]].append(r)

    contracts = list(by_contract.keys())
    rng = random.Random(seed)
    rng.shuffle(contracts)

    n = len(contracts)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    train_contracts = set(contracts[:n_train])
    val_contracts = set(contracts[n_train:n_train + n_val])
    test_contracts = set(contracts[n_train + n_val:])

    splits = {"train": [], "val": [], "test": []}
    for c in train_contracts:
        splits["train"].extend(by_contract[c])
    for c in val_contracts:
        splits["val"].extend(by_contract[c])
    for c in test_contracts:
        splits["test"].extend(by_contract[c])
    return splits


def main():
    raw_rows = load_raw_examples()
    rows = dedup_exact(raw_rows)

    report = {
        "labels": LABELS,
        "cuad_category_map": CATEGORY_MAP,
        "min_chars_filter": MIN_CHARS,
        "raw_span_count_per_category": dict(Counter(r["category"] for r in raw_rows)),
        "after_dedup_count_per_category": dict(Counter(r["category"] for r in rows)),
    }

    splits = split_by_contract(rows)

    for split_name, split_rows in splits.items():
        out_path = OUT_DIR / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for r in split_rows:
                f.write(json.dumps({
                    "text": r["text"],
                    "category": r["category"],
                    "label_id": LABEL2ID[r["category"]],
                    "contract": r["contract"],
                }) + "\n")
        report[f"{split_name}_count_per_category"] = dict(Counter(r["category"] for r in split_rows))
        report[f"{split_name}_total"] = len(split_rows)
        report[f"{split_name}_num_contracts"] = len({r["contract"] for r in split_rows})

    report["total_examples_after_filtering"] = len(rows)
    report["total_contracts_used"] = len({r["contract"] for r in rows})

    (OUT_DIR / "data_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
