"""
Phase 3(B) — honest baseline evaluation using ClauseGuard's REAL, unmodified
stage-1 heading-lexicon classifier (backend/classification/heading_match.py's
classify_heading()), imported directly, on the SAME 189-example held-out
test set the fine-tuned DistilBERT model was scored on.

WHY STAGE 1 ONLY, NOT THE FULL TWO-STAGE CASCADE (disclosed, not hidden):
classify_clause()'s stage 2 (classification/centroid_fallback.py's
classify_by_centroid()) makes a live Voyage AI API call requiring
VOYAGE_API_KEY. Per the user's explicit choice ("B" in chat), this
environment does not have that credential, so stage 2 cannot be executed.
This script therefore measures stage 1 ALONE -- a real, meaningful, but
partial baseline. Any clause stage 1 can't confidently match is correctly
left "Unclassified" (None) here, exactly as the real code behaves; this
script does NOT invent a stage-2 substitute or guess in its place.

TWO DISCLOSED, NECESSARY ADAPTATIONS (neither changes classify_heading()'s
own code or logic -- both are documented here, not silently done):

1. HEADING_PATH PROXY: classify_heading() takes a `heading_path` string.
   Real ClauseGuard gets this from its own document chunker parsing a
   PDF/DOCX's actual heading structure. CUAD's test examples are raw
   SQuAD-style answer SPANS (clause body text only) -- there is no
   chunker-produced heading_path for them. This script approximates one
   by taking the line of the ORIGINAL CUAD contract text immediately
   preceding the answer span (real contract text, not fabricated) --
   e.g. "6.9      Governing Law." often appears right before a
   Governing Law answer span in the raw CUAD contracts. This is a
   reasonable proxy, not an exact substitute for ClauseGuard's real
   chunker output, and is reported as such.

2. IMPORT SHIM (tiktoken only, NOT classify_heading's logic):
   heading_match.py imports centroid_fallback.py at module level, which
   imports embeddings/rate_limit.py, which calls
   tiktoken.get_encoding("cl100k_base") at IMPORT TIME to build an
   unrelated token-count estimate used only for Voyage-API rate pacing
   (irrelevant to classify_heading()). That call tries to download
   encoding files from openaipublic.blob.core.windows.net, blocked by
   this sandbox's network allowlist. We stub ONLY tiktoken.get_encoding
   with a no-op encoder so the import chain resolves -- classify_heading()
   itself is imported and called completely unmodified, verbatim from
   the repo.
"""
import json
import re
import sys
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report

# --- disclosed shim (see docstring point 2) ---
import tiktoken


class _FakeEncoding:
    def encode(self, text):
        return text.split()


tiktoken.get_encoding = lambda name: _FakeEncoding()
# --- end shim ---

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from classification.heading_match import classify_heading  # noqa: E402  (real, unmodified import)

EXP_DIR = Path(__file__).resolve().parents[1]
RAW_CUAD_PATH = EXP_DIR / "data" / "CUADv1_raw.json"
TEST_PATH = EXP_DIR / "data" / "test.jsonl"
FT_PREDICTIONS_PATH = EXP_DIR / "data" / "finetuned_test_predictions.jsonl"  # copied from user's outputs_bundle.zip

LABELS = ["Limitation of Liability", "Governing Law / Jurisdiction", "Termination"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

CUAD_TO_CG = {
    "Cap On Liability": "Limitation of Liability",
    "Governing Law": "Governing Law / Jurisdiction",
    "Termination For Convenience": "Termination",
}

HEADING_WINDOW_CHARS = 200


def load_test_rows():
    rows = []
    with open(TEST_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_text(t: str) -> str:
    """MUST match prepare_data.py's clean_text() exactly -- test.jsonl's
    "text" field was written through that normalization, so raw CUAD answer
    spans need the identical normalization applied before string-matching
    against it, or matching spuriously fails on whitespace differences
    alone (confirmed: this was the actual cause of an inflated
    unmatched-lookup count in an earlier version of this script)."""
    t = t.strip()
    t = re.sub(r"\s+", " ", t)
    return t


def build_span_lookup(raw_data):
    """contract title -> list of (cuad_category, normalized_answer_text, answer_start, context)"""
    lookup = {}
    for doc in raw_data["data"]:
        title = doc["title"]
        for para in doc["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                cat = qa["id"].split("__")[-1]
                if cat not in CUAD_TO_CG or qa.get("is_impossible", True):
                    continue
                for ans in qa["answers"]:
                    lookup.setdefault(title, []).append(
                        (cat, clean_text(ans["text"]), ans["answer_start"], ctx)
                    )
    return lookup


def extract_heading_proxy(context: str, answer_start: int) -> str | None:
    """Take the last non-empty line of raw contract text immediately
    preceding the answer span, as a heading_path proxy. Real, unmodified
    contract text -- not fabricated. Returns None if nothing usable found."""
    window_start = max(0, answer_start - HEADING_WINDOW_CHARS)
    window = context[window_start:answer_start]
    lines = [ln.strip() for ln in window.split("\n") if ln.strip()]
    if not lines:
        return None
    candidate = lines[-1]
    # Guard against grabbing a long body-text line that isn't heading-shaped;
    # real contract headings are short. If the immediate-preceding line is
    # too long to plausibly be a heading, fall back one more line.
    if len(candidate) > 80 and len(lines) >= 2:
        candidate = lines[-2]
    return candidate


def main():
    test_rows = load_test_rows()
    raw_data = json.loads(RAW_CUAD_PATH.read_text())
    span_lookup = build_span_lookup(raw_data)

    y_true, y_pred, unmatched = [], [], 0
    heading_proxies_used = []

    for row in test_rows:
        contract = row["contract"]
        text = row["text"]
        true_cat = row["category"]

        candidates = span_lookup.get(contract, [])
        match = None
        for cat, ans_text, ans_start, ctx in candidates:
            if ans_text == text:
                match = (ans_start, ctx)
                break

        if match is None:
            unmatched += 1
            y_true.append(LABEL2ID[true_cat])
            y_pred.append(-1)  # can't even build a heading proxy -> Unclassified
            continue

        ans_start, ctx = match
        heading_proxy = extract_heading_proxy(ctx, ans_start)
        heading_proxies_used.append(heading_proxy)

        predicted = classify_heading(heading_proxy) if heading_proxy else None
        y_true.append(LABEL2ID[true_cat])
        y_pred.append(LABEL2ID[predicted] if predicted in LABEL2ID else -1)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n_total = len(y_true)
    n_unclassified = int((y_pred == -1).sum())
    coverage = 1 - (n_unclassified / n_total)

    # For accuracy/P/R/F1: an "Unclassified" (-1) prediction is scored as
    # simply wrong for whatever the true label was (a real classifier output
    # of "no confident match" is not credited as correct), consistent with
    # how a live system's silence would be judged in production.
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(LABELS))), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[-1] + list(range(len(LABELS))))

    report = {
        "labels": LABELS,
        "n_test_examples": n_total,
        "n_unmatched_span_lookup_failures": unmatched,
        "n_unclassified_by_stage1": n_unclassified,
        "stage1_coverage_fraction": coverage,
        "accuracy_treating_unclassified_as_wrong": float(accuracy),
        "per_class": {
            LABELS[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(LABELS))
        },
        "confusion_matrix_label_order": ["Unclassified"] + LABELS,
        "confusion_matrix": cm.tolist(),
        "note": (
            "Baseline = ClauseGuard's real, unmodified stage-1 heading-lexicon "
            "classifier only. Stage 2 (embedding centroid) was NOT run: it "
            "requires a live Voyage AI API call (VOYAGE_API_KEY), unavailable "
            "in this environment, per explicit user decision to proceed with "
            "stage-1-only baseline rather than fabricate/skip this constraint. "
            "heading_path was approximated from real preceding CUAD contract "
            "text (see script docstring) since CUAD spans have no chunker-"
            "produced heading_path of their own."
        ),
    }

    out_path = EXP_DIR / "data" / "baseline_stage1_metrics.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
