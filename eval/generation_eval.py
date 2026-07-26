"""M27: the first end-to-end PRODUCT quality check in this project's eval
suite -- does the REAL flag_clause() output (backend/agent/flag_clause.py,
M12/M13, unchanged, a real live Gemini call) match the human label on
each of the 5 real rows in eval/labeled_set/m19_fixture.json? Every eval
since M19 measured retrieval/classification MECHANICS (Hit@K, reranking,
fusion, threshold placement) using local, cached, or CPU-only
computation; this is the first one that measures actual generated
output against a human judgment call, and the first that spends a real,
metered third-party quota to do it.

REUSE, NOT REIMPLEMENTATION: eval/loader.py's LabeledClause /
load_labeled_clauses (M18, unchanged) for the labeled rows;
eval/retrieval_eval.py's _fetch_contract_clauses (M19, unchanged) for
real Postgres clause text; backend/pipeline/run_contract.py's own
already-established, real-429-tested Gemini pacing/retry logic
(_flag_with_retry, MIN_SECONDS_BETWEEN_FLAG_CALLS=7,
RATE_LIMIT_RETRY_WAIT_SECONDS=45, MAX_RATE_LIMIT_RETRIES=2 -- all M13,
confirmed against real rate-limit responses during that milestone, not
reimplemented here). backend/agent/flag_clause.py's flag_clause() itself
is called directly and is NOT modified, wrapped, or reimplemented --
this milestone measures it, per spec it does not fix it.

QUOTA DISCIPLINE (read before running this file): a full run costs 5 real
Gemini calls (one per fixture row), paced 7s apart (M13's own established
floor for gemini-2.5-flash-lite's third-party-reported 15 RPM limit),
with up to 2 retries per row on an actual rate-limit hit -- worst case
~15 calls for one complete run, still well within a typical free-tier
daily quota, but NOT something to re-run casually. The four functions
that do NOT touch Gemini at all -- _is_flagged(), _severity_closeness(),
_citation_resolved(), _precision_recall() -- are plain, pure,
synchronously-testable functions; verify those against synthetic/stubbed
inputs first (no API calls) before spending real quota running evaluate()
against the live pipeline.

DETERMINISM, EXPLICITLY DIFFERENT FROM EVERY PRIOR EVAL: M19 through M26
were fully deterministic by construction (disk-cached Voyage embeddings,
local CPU-only cross-encoder/BM25 inference, no live calls once a cache
was warm). THIS milestone's core input -- Gemini's real severity/
explanation/citation for a given clause -- is NOT expected to be
bit-identical across separate live calls. flag_clause.py does not pin a
deterministic sampling temperature, and even near-zero-temperature hosted
LLM inference is not contractually guaranteed reproducible run-to-run
(the same class of caveat M19's own DETERMINISM NOTE raised for Voyage's
embeddings -- "undocumented either way" -- now applies to GENERATION
rather than embedding). Running this script twice may legitimately
produce a different severity or citation for the same clause; that is
NOT a bug in this script to chase. What IS deterministic, and IS the
thing worth verifying repeatably: the citation-resolution substring
check, the severity-closeness classification, and the precision/recall
arithmetic -- pure functions over whatever real output a given run
happens to produce.

WHAT "FLAGGED" MEANS HERE (an explicit definition, not left implicit):
flag_clause()'s tool schema (agent/tools.py's FlagClauseOutput) has no
native yes/no "is this risky" field -- Gemini is FORCED to call the tool
every time (tool_config mode="ANY") and always returns a severity
(low/medium/high). This script defines predicted_flagged as
severity != "low" -- and this is not an arbitrary choice invented for
this script: every is_risky=true row in the real fixture has
severity medium or high, and every is_risky=false row has severity low
(confirmed directly by reading m19_fixture.json), so this definition
exactly matches how the human labels were themselves constructed.

CITATION RESOLUTION: "genuinely traceable", not merely "non-empty".
_citation_resolved() checks whether Gemini's returned citation string
appears as a SUBSTRING of the real persisted clause text for that row's
clause_id, after normalizing whitespace only (collapsing runs of
whitespace to a single space, stripping ends) on both sides -- tolerating
trivial formatting differences, NOT fuzzy/approximate text matching. A
paraphrased or fabricated-sounding citation will still fail this check,
since its actual wording would differ from the real source text; only
whitespace/newline placement differences are forgiven.

OUT OF SCOPE (per this milestone's own spec): no recalibration of
flag_clause's prompt or logic based on what this run finds -- a real
quality problem discovered here is a FINDING to report, not something
this file fixes. No new labeled data. No negotiation-suggestion tool
(M28+).
"""

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `from loader import ...` / `from retrieval_eval import ...` work regardless of caller's cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so backend/'s top-level modules (db, models, embeddings, agent, pipeline) resolve

from loader import LabeledClause, load_labeled_clauses  # noqa: E402
from retrieval_eval import _fetch_contract_clauses  # noqa: E402

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}
_WHITESPACE_RE = re.compile(r"\s+")


def _is_flagged(severity: str) -> bool:
    """Predicted 'flagged' = severity != 'low'. See module docstring's
    WHAT "FLAGGED" MEANS HERE section for why this exact definition
    (not an arbitrary threshold) matches how the real labeled fixture's
    is_risky values were themselves derived from severity.
    """
    return severity != "low"


def _severity_closeness(predicted: str, actual: str) -> str:
    """EXACT (identical) / CLOSE (one step apart on the low<medium<high
    ordinal scale, e.g. medium vs high) / FAR (opposite ends, low vs
    high) -- a pure, deterministic function of two severity strings, no
    Gemini call involved.
    """
    if predicted == actual:
        return "EXACT"
    distance = abs(SEVERITY_ORDER[predicted] - SEVERITY_ORDER[actual])
    return "CLOSE" if distance == 1 else "FAR"


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _citation_resolved(citation: str, real_clause_text: str) -> bool:
    """True if `citation` genuinely appears as a substring of
    `real_clause_text` after whitespace-only normalization on both sides
    -- see module docstring's CITATION RESOLUTION section for why this
    is real traceability, not just "a citation was returned."
    """
    if not citation:
        return False
    return _normalize_whitespace(citation) in _normalize_whitespace(real_clause_text)


def _precision_recall(rows: list[dict]) -> dict:
    """rows: list of {"actual_flagged": bool, "predicted_flagged": bool | None}.
    Rows with predicted_flagged=None (a failed flag_clause() call) are
    excluded from N entirely -- a failure is not a negative prediction,
    it's an absence of a prediction. Pure arithmetic, deterministic given
    its inputs, no Gemini call.
    """
    evaluable = [r for r in rows if r["predicted_flagged"] is not None]
    tp = sum(1 for r in evaluable if r["predicted_flagged"] and r["actual_flagged"])
    fp = sum(1 for r in evaluable if r["predicted_flagged"] and not r["actual_flagged"])
    fn = sum(1 for r in evaluable if not r["predicted_flagged"] and r["actual_flagged"])
    tn = sum(1 for r in evaluable if not r["predicted_flagged"] and not r["actual_flagged"])
    n = len(evaluable)
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall}


# --- evaluation orchestration (makes REAL Gemini calls) ----------------------


def evaluate(labeled_rows: list[LabeledClause], db_session) -> dict:
    """Runs the REAL flag_clause() against every labeled row's real
    clause text, paced/retried via pipeline.run_contract's own
    already-established Gemini rate-limit handling (M13, not
    reimplemented here). Rows whose clause_id has no real backing data
    are skipped, not silently scored. Rows whose flag_clause() call fails
    even after retries are recorded with predicted_flagged=None -- an
    absence of a prediction, excluded from precision/recall's N, never
    silently treated as a negative prediction.
    """
    from pipeline.run_contract import MIN_SECONDS_BETWEEN_FLAG_CALLS, _flag_with_retry

    text_by_clause_id: dict[str, str] = {}
    fetched_contract_ids: set = set()
    for row in labeled_rows:
        if row.contract_id not in fetched_contract_ids:
            fetched_contract_ids.add(row.contract_id)
            candidates = _fetch_contract_clauses(row.contract_id, db_session)
            text_by_clause_id.update(dict(candidates))

    results: list[dict] = []
    skipped_rows: list[LabeledClause] = []
    last_call_time: float | None = None
    total = len(labeled_rows)

    for i, row in enumerate(labeled_rows, start=1):
        clause_id = str(row.clause_id)
        if clause_id not in text_by_clause_id:
            skipped_rows.append(row)
            continue

        if last_call_time is not None:
            wait = MIN_SECONDS_BETWEEN_FLAG_CALLS - (time.monotonic() - last_call_time)
            if wait > 0:
                time.sleep(wait)

        clause_text = text_by_clause_id[clause_id]
        output = _flag_with_retry(clause_text, i, total)
        last_call_time = time.monotonic()

        if output is None:
            results.append({
                "row": row,
                "clause_text": clause_text,
                "predicted_severity": None,
                "predicted_flagged": None,
                "citation": None,
                "citation_resolved": None,
                "explanation": None,
                "failed": True,
            })
            continue

        results.append({
            "row": row,
            "clause_text": clause_text,
            "predicted_severity": output.severity,
            "predicted_flagged": _is_flagged(output.severity),
            "citation": output.citation,
            "citation_resolved": _citation_resolved(output.citation, clause_text),
            "explanation": output.explanation,
            "failed": False,
        })

    pr_rows = [
        {"actual_flagged": r["row"].is_risky, "predicted_flagged": r["predicted_flagged"]}
        for r in results
    ]
    precision_recall = _precision_recall(pr_rows)

    return {"results": results, "skipped": skipped_rows, "precision_recall": precision_recall}


def _fmt_rate(value: float | None, numerator: int, denominator: int, n: int) -> str:
    if value is None:
        return f"N/A (denominator is 0)  [N={n}]"
    return f"{value:.4f} ({numerator}/{denominator})  [N={n}]"


def print_report(result: dict) -> None:
    print("=" * 72)
    print("ClauseGuard M27 Generation (flag_clause) Quality Eval")
    print("=" * 72)
    print(
        "HONEST FRAMING: N=5 -- illustrative only, NOT statistically meaningful. "
        "One wrong prediction swings precision or recall by 20 percentage points. "
        "A real, confident quality verdict requires M18's still-outstanding real "
        "hand-labeling work (15-20+ real contracts). This eval MEASURES flag_clause's "
        "real output against the label; it does not tune or fix flag_clause based on "
        "what it finds here, regardless of the result."
    )
    print(
        "DETERMINISM NOTE: unlike M19-M26's fully local/cached pipelines, Gemini's "
        "real severity/citation output is NOT expected to be bit-identical across "
        "separate live runs -- expected, not a bug. The citation-resolution check, "
        "severity-closeness classification, and precision/recall arithmetic below ARE "
        "pure, deterministic functions of whatever real output this specific run produced."
    )
    print("=" * 72)

    for row in result["skipped"]:
        print(
            f"SKIPPED contract_id={row.contract_id} clause_id={row.clause_id}: "
            f"no real Clause row found for this clause_id -- excluded, no Gemini call made."
        )

    print()
    print("Per-row detail:")
    for r in result["results"]:
        row = r["row"]
        print(f"  clause_id={row.clause_id}  category={row.category!r}")
        print(f"    query={row.query!r}")
        if r["failed"]:
            print("    *** flag_clause() call FAILED for this row (see logged warning/error above) -- excluded from precision/recall N ***")
            print()
            continue

        actual_flagged = row.is_risky
        pred_flagged = r["predicted_flagged"]
        flag_result = "MATCH" if pred_flagged == actual_flagged else "MISMATCH"
        print(f"    flag:     predicted={pred_flagged}  actual={actual_flagged}  [{flag_result}]")

        closeness = _severity_closeness(r["predicted_severity"], row.severity)
        print(f"    severity: predicted={r['predicted_severity']!r}  actual={row.severity!r}  [{closeness}]")

        print(f"    citation_resolved={r['citation_resolved']}")
        print(f"    citation={r['citation']!r}")
        print(f"    explanation={r['explanation']!r}")
        print()

    pr = result["precision_recall"]
    print("-" * 72)
    print(f"PRECISION/RECALL for is_risky flag prediction -- SEE HONEST FRAMING ABOVE:")
    print(f"  TP={pr['tp']}  FP={pr['fp']}  FN={pr['fn']}  TN={pr['tn']}   [N={pr['n']}]")
    print(f"  Precision: {_fmt_rate(pr['precision'], pr['tp'], pr['tp'] + pr['fp'], pr['n'])}")
    print(f"  Recall:    {_fmt_rate(pr['recall'], pr['tp'], pr['tp'] + pr['fn'], pr['n'])}")
    print("-" * 72)
    print(f"REMINDER: N={pr['n']} is far too small for these numbers to be a meaningful quality verdict. Illustrative only -- see HONEST FRAMING above.")
    print("=" * 72)


if __name__ == "__main__":
    default_path = str(Path(__file__).resolve().parent / "labeled_set" / "m19_fixture.json")
    target = sys.argv[1] if len(sys.argv) > 1 else default_path

    labeled_rows = load_labeled_clauses(target)

    from db import SessionLocal

    db = SessionLocal()
    try:
        result = evaluate(labeled_rows, db)
    finally:
        db.close()

    print_report(result)
