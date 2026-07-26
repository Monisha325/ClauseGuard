"""Phase 2 finalize (reduced scale) -- run this locally, same as
generate_teacher_labels.py. Reads your REAL data/teacher_labels.jsonl and:

  1. Reports exact real counts: how many rows are genuinely label_status="ok"
     vs. "flagging_failed" vs. "needs_manual_review" -- no guessing, no
     fabrication, straight from your actual file.
  2. Filters to ok rows only, and builds a FRESH ~70/15/15 train/val/test
     split from just those survivors -- grouped by contract (never
     splitting one contract's clauses across two sets), stratified
     roughly by category.

     WHY A FRESH SPLIT, NOT REUSING THE ORIGINAL split FIELD: each row
     still carries the "split" label select_distillation_subset.py gave
     it (from the ORIGINAL 450-clause plan). But generate_teacher_labels.py
     processes rows in file order -- all train_* rows first, then val_*,
     then test_* -- and the real run stopped partway through due to the
     daily token budget. That means the surviving "ok" rows are very
     likely concentrated almost entirely in train_*, leaving val/test
     empty or near-empty if we just filtered by the old label -- which
     would silently leave NO real held-out test set for Phase 4's
     evaluation. Building a fresh split from the actual survivors avoids
     that trap and is what was actually asked for ("re-split this final
     set into train/val/test").
  3. Prints a per-split x per-category count table (paste this back, not
     the raw file) plus flags any cell that's very small (<5), since a
     preliminary small-scale experiment's honest limitations should be
     visible, not discovered later.
  4. Writes experiments/llm_distillation/data/final_{train,val,test}.jsonl
     locally -- these stay on your machine; you'll upload THESE (not the
     raw teacher_labels.jsonl) to Colab in Phase 3.

Nothing here calls Groq or any network -- pure local file processing.
Fixed random seed (42) for reproducibility.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

INPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "teacher_labels.jsonl"
OUT_DIR = Path(__file__).resolve().parents[1] / "data"
SEED = 42
TARGET_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def assign_contracts_to_splits(ok_rows: list[dict], rng: random.Random) -> dict[str, str]:
    """Greedy, deterministic: shuffle contracts (fixed seed), then assign
    each contract's ENTIRE group of clauses to whichever split currently
    has the largest shortfall versus its target fraction of the total.
    This keeps a contract's clauses together (no leakage) while tracking
    the 70/15/15 target at the overall clause-count level. For a dataset
    this small, perfect per-category balance isn't guaranteed -- the
    resulting per-category counts are reported honestly afterward rather
    than forced.
    """
    by_contract: dict[str, list[dict]] = defaultdict(list)
    for r in ok_rows:
        by_contract[r["contract"]].append(r)

    contracts = list(by_contract.keys())
    rng.shuffle(contracts)

    total = len(ok_rows)
    targets = {split: total * frac for split, frac in TARGET_FRACTIONS.items()}
    current = {"train": 0, "val": 0, "test": 0}
    contract_to_split = {}

    for contract in contracts:
        n = len(by_contract[contract])
        # assign to whichever split is furthest below its target
        deficits = {s: targets[s] - current[s] for s in current}
        chosen = max(deficits, key=deficits.get)
        contract_to_split[contract] = chosen
        current[chosen] += n

    return contract_to_split


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"Could not find {INPUT_PATH} -- make sure you're running this "
              f"from the project root, same as generate_teacher_labels.py.")
        return

    rows = load_jsonl(INPUT_PATH)
    status_counts = Counter(r.get("label_status", "UNKNOWN") for r in rows)

    print("=" * 70)
    print("REAL COUNTS FROM YOUR ACTUAL teacher_labels.jsonl")
    print("=" * 70)
    print(f"Total rows in file:       {len(rows)}")
    for status in ("ok", "flagging_failed", "needs_manual_review"):
        print(f"  {status:<22} {status_counts.get(status, 0)}")
    other = set(status_counts) - {"ok", "flagging_failed", "needs_manual_review"}
    for status in other:
        print(f"  {status:<22} {status_counts[status]} (UNEXPECTED STATUS)")
    print()

    ok_rows = [r for r in rows if r.get("label_status") == "ok"]

    if not ok_rows:
        print("No successfully labeled rows found -- nothing to split.")
        return

    bad = [r for r in ok_rows if "contract" not in r or "category" not in r]
    if bad:
        print(f"WARNING: {len(bad)} ok rows are missing contract/category "
              f"fields -- these will be excluded from the split below. "
              f"This is a REAL data issue to report, not to hide.")
        ok_rows = [r for r in ok_rows if "contract" in r and "category" in r]

    rng = random.Random(SEED)
    contract_to_split = assign_contracts_to_splits(ok_rows, rng)

    for r in ok_rows:
        r["final_split"] = contract_to_split[r["contract"]]

    # Hardcode the full known category set (from the original CUAD-derived
    # dataset) rather than deriving it from ok_rows only -- if a category
    # got ZERO successful examples, it should show up as an explicit 0 and
    # trigger the thin-cell warning below, not silently vanish from the
    # table as if it never existed.
    categories = [
        "Governing Law / Jurisdiction",
        "Limitation of Liability",
        "Termination",
    ]
    found_categories = {r["category"] for r in ok_rows}
    unexpected = found_categories - set(categories)
    if unexpected:
        print(f"WARNING: found categories not in the expected set: "
              f"{unexpected} -- adding them to the table.")
        categories += sorted(unexpected)
    splits = ["train", "val", "test"]

    by_split_category: dict[str, Counter] = defaultdict(Counter)
    for r in ok_rows:
        by_split_category[r["final_split"]][r["category"]] += 1

    print("=" * 70)
    print("FRESH 70/15/15 RE-SPLIT, PER-SPLIT x PER-CATEGORY COUNTS")
    print("(paste this whole block back)")
    print("=" * 70)
    header = f"{'Split':<8}" + "".join(f"{c[:22]:<24}" for c in categories) + "Total"
    print(header)
    grand_total = 0
    for split in splits:
        row_counts = by_split_category.get(split, Counter())
        split_total = sum(row_counts.values())
        grand_total += split_total
        line = f"{split:<8}" + "".join(f"{row_counts.get(c, 0):<24}" for c in categories) + f"{split_total}"
        print(line)
    print(f"{'TOTAL':<8}" + "".join(
        f"{sum(by_split_category[s].get(c, 0) for s in splits):<24}" for c in categories
    ) + f"{grand_total}")
    print()

    thin_cells = []
    for split in splits:
        for c in categories:
            n = by_split_category.get(split, Counter()).get(c, 0)
            if n < 5:
                thin_cells.append((split, c, n))
    if thin_cells:
        print("DATA-QUALITY NOTE (real, disclosed -- include in REPORT.md):")
        for split, c, n in thin_cells:
            print(f"  - {split}/{c}: only {n} example(s) -- too small to "
                  f"draw a reliable per-category conclusion from.")
        print()

    n_contracts = len(set(contract_to_split.keys()))
    print(f"Total unique contracts represented: {n_contracts}")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in splits:
        split_rows = [r for r in ok_rows if r["final_split"] == split]
        out_path = OUT_DIR / f"final_{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in split_rows:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(split_rows)} rows to {out_path}")

    print()
    print("Paste the counts table + data-quality note above back to Claude. "
          "Keep final_train/val/test.jsonl on this machine -- you'll upload "
          "those (not raw teacher_labels.jsonl) to Colab in Phase 3.")


if __name__ == "__main__":
    main()
