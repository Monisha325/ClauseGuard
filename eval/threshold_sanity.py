"""M26: lightweight sanity check on M20's per-category confidence
threshold table (backend/retrieval/thresholds.py's CATEGORY_THRESHOLDS),
NOT a full recalibration -- that requires far more labeled data than
currently exists (5 rows total, ~1 per category, in
eval/labeled_set/m19_fixture.json). This script's whole job is printing
what the current handful of real labeled examples' actual post-sigmoid
rerank scores look like against each category's threshold, honestly
flagging INSUFFICIENT DATA wherever (almost certainly everywhere, given
today's fixture) there isn't enough to say anything meaningful, rather
than manufacturing a confident-looking verdict from N=1.

REUSE, NOT REIMPLEMENTATION: eval/loader.py's LabeledClause /
load_labeled_clauses (M18, unchanged) for the labeled rows themselves;
eval/retrieval_eval.py's _fetch_contract_clauses (M19, unchanged) for
real Postgres clause text, and _assert_rerank_path_is_local_only (M22/M25,
unchanged) for the same concrete determinism guard those scripts already
established; backend/retrieval/reranker.py's rerank() (M21, unchanged)
for the real post-sigmoid score; backend/retrieval/thresholds.py's
CATEGORY_THRESHOLDS (M20, unchanged) for the threshold values themselves.
None of these are reimplemented here.

WHY EACH ROW'S SCORE COMES FROM A SINGLE-CANDIDATE rerank() CALL:
CrossEncoder.predict() (reranker.py's own underlying model call) scores
each (query, candidate_text) PAIR independently -- it is a pointwise
cross-encoder, not a listwise ranker, so a candidate's sigmoid score does
not depend on what OTHER candidates happen to be in the same rerank()
call (confirmed by reading reranker.py's own rerank(): it maps
model.predict() over the given pairs, sigmoids each independently, then
sorts -- no cross-candidate interaction in the scoring step, only in the
final sort order). This means calling
rerank(row.query, [(row.clause_id, row's own real persisted text)]) --
a single-candidate call -- produces EXACTLY the same score for that
clause as if it were reranked alongside a full candidate pool, the way
retrieval.py's real endpoint does it. This is the real, correct score
for "how does this clause's own query score against its own text",
independent of pool composition -- not an approximation of the real
production score, the actual same computation.

MINIMUM COUNT: this milestone's spec asks for "~5" labeled examples per
category as a reasonable floor before attempting any sanity judgment at
all -- MIN_EXAMPLES_PER_CATEGORY below. Given the current fixture has
exactly 1 example per category (confirmed directly by reading
m19_fixture.json, not assumed), EVERY category is expected to trip this
flag right now -- that is the correct, honest, EXPECTED output at this
stage of the project, not a bug in this script. The per-category gap
logic below is still written to work correctly if the fixture ever grows
past the floor (so this script doesn't need rewriting once M18's real
hand-labeling happens), it just won't have anything to say yet.

QUALITATIVE FINDINGS below are PRIOR CONTEXT, NOT DERIVED HERE -- printed
verbatim as attributed context from earlier milestones' own independent
reviews, never as if this script computed them itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `from loader import ...` / `from retrieval_eval import ...` work regardless of caller's cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so backend/'s top-level modules (db, models, embeddings, retrieval) resolve

from loader import LabeledClause, load_labeled_clauses  # noqa: E402
from retrieval_eval import _assert_rerank_path_is_local_only, _fetch_contract_clauses  # noqa: E402

MIN_EXAMPLES_PER_CATEGORY = 5  # per this milestone's own spec: "~5" as a reasonable floor before any sanity judgment is attempted

# Attributed prior context from earlier milestones' own independent
# reviews -- NOT derived by this script. Printed as-is, clearly labeled,
# alongside (not folded into) the real quantitative table below.
QUALITATIVE_FINDINGS: list[tuple[str, str]] = [
    ("M11", "tight score clustering observed among non-top-1 vector search results."),
    ("M21", "a steep, highly-peaked post-rerank (sigmoid) score distribution -- real scores observed clustering near 0 or near 1, with little middle ground, on the queries tested."),
    ("M22", "a ceiling effect in the current fixture (every row already an unambiguous, 'easy' match for every retrieval method tried), plus a real jargon/abbreviation sensitivity -- an NDA-phrased query ('penalty for breaking an NDA') dropped the true Confidentiality clause to LAST place (position 7 of 7) under vector-only search, though reranking partially recovered it (position 5 of 7)."),
    ("M23", "a narrow (~0.085 similarity units) real margin observed around the embedding-centroid fallback classifier's 0.5 threshold, between the closest confirmed-positive and confirmed-negative real examples found."),
    ("M24", "vector search and BM25 tend to AGREE on rare-phrase overlap for this domain/embedding-model pairing -- a real, recurring signal, not a one-off."),
    ("M25", "fusion (BM25 + RRF) alone can REGRESS Hit@1 on keyword-trap-style queries (a real, measured drop from 1.0 to 0.8 on this fixture's deliberate keyword-vs-semantic contrast row), recovered by the reranker in the one case observed."),
]


def _score_row(row: LabeledClause, text_by_clause_id: dict[str, str]) -> float:
    """Real post-sigmoid rerank score for this row's own clause against
    its own labeled query -- see module docstring for why a
    single-candidate rerank() call gives the exact same score as a full
    candidate pool would.
    """
    from retrieval.reranker import rerank

    clause_id = str(row.clause_id)
    text = text_by_clause_id[clause_id]
    reranked = rerank(row.query, [(clause_id, text)])
    return reranked[0][1]


def evaluate(labeled_rows: list[LabeledClause], db_session) -> dict:
    """Fetches real clause text (grouped by contract_id, reusing
    _fetch_contract_clauses per distinct contract_id so a shared
    contract's clauses aren't re-fetched once per row), scores every row
    via the real rerank(), and groups results by category. Rows whose
    clause_id has no real backing data are skipped, not silently scored
    as 0 or omitted without a trace.
    """
    _assert_rerank_path_is_local_only()

    text_by_clause_id: dict[str, str] = {}
    fetched_contract_ids: set = set()
    for row in labeled_rows:
        if row.contract_id not in fetched_contract_ids:
            fetched_contract_ids.add(row.contract_id)
            candidates = _fetch_contract_clauses(row.contract_id, db_session)
            text_by_clause_id.update(dict(candidates))

    by_category: dict[str, list[dict]] = {}
    skipped_rows: list[LabeledClause] = []
    for row in labeled_rows:
        clause_id = str(row.clause_id)
        if clause_id not in text_by_clause_id:
            skipped_rows.append(row)
            continue
        score = _score_row(row, text_by_clause_id)
        by_category.setdefault(row.category, []).append({"row": row, "score": score})

    return {"by_category": by_category, "skipped": skipped_rows}


def _category_report(category: str, threshold: float, entries: list[dict]) -> list[str]:
    """Builds the printed lines for one category. `entries` is this
    category's list of {"row": LabeledClause, "score": float} dicts
    (possibly empty). Returns a list of lines rather than printing
    directly, so print_report() controls overall spacing/structure.
    """
    lines = []
    lines.append(f"--- {category} ---")
    lines.append(f"Threshold (backend/retrieval/thresholds.py CATEGORY_THRESHOLDS): {threshold}")

    n = len(entries)
    lines.append(f"Total labeled examples in this category: {n}")

    if n < MIN_EXAMPLES_PER_CATEGORY:
        lines.append(
            f"*** INSUFFICIENT DATA -- {n} example(s) is far below the "
            f"minimum of {MIN_EXAMPLES_PER_CATEGORY} this script requires before attempting "
            f"ANY judgment about where {threshold} sits. Printing the real score(s) observed "
            f"below for reference only -- NOT as the basis for a verdict. A single example "
            f"(or a small handful) is not a distribution; treating it as one would manufacture "
            f"false confidence this milestone explicitly exists to avoid. ***"
        )
    else:
        tp_scores = [e["score"] for e in entries if e["row"].is_risky]
        fp_scores = [e["score"] for e in entries if not e["row"].is_risky]
        if tp_scores and fp_scores and min(tp_scores) > threshold > max(fp_scores):
            lines.append(
                f"With {n} examples (>= {MIN_EXAMPLES_PER_CATEGORY}): {threshold} appears to sit in a "
                f"defensible gap -- every true-positive score ({min(tp_scores):.4f}-{max(tp_scores):.4f}) "
                f"is above it, and every false-positive score ({min(fp_scores):.4f}-{max(fp_scores):.4f}) "
                f"is below it, for the examples observed."
            )
        elif tp_scores and fp_scores:
            lines.append(
                f"With {n} examples (>= {MIN_EXAMPLES_PER_CATEGORY}): {threshold} does NOT cleanly "
                f"separate the observed groups -- true-positive scores range "
                f"{min(tp_scores):.4f}-{max(tp_scores):.4f}, false-positive scores range "
                f"{min(fp_scores):.4f}-{max(fp_scores):.4f}; these overlap around the threshold. "
                f"Worth a closer look, though still just {n} examples, not a full calibration fit."
            )
        else:
            lines.append(
                f"With {n} examples (>= {MIN_EXAMPLES_PER_CATEGORY}) but only one of the two "
                f"true-positive/false-positive groups present, this script cannot judge whether "
                f"{threshold} separates them -- a gap needs both sides represented."
            )

    tp_entries = [e for e in entries if e["row"].is_risky]
    fp_entries = [e for e in entries if not e["row"].is_risky]

    lines.append(f"  True-positive (is_risky=true) examples: {len(tp_entries)}")
    for e in tp_entries:
        lines.append(f"    score={e['score']:.4f}  clause_id={e['row'].clause_id}  query={e['row'].query!r}")

    lines.append(f"  False-positive (is_risky=false) examples: {len(fp_entries)}")
    if not fp_entries:
        lines.append("    (no genuine false-positive example exists in the current fixture for this category -- not fabricated here)")
    for e in fp_entries:
        lines.append(f"    score={e['score']:.4f}  clause_id={e['row'].clause_id}  query={e['row'].query!r}")

    return lines


def print_report(result: dict) -> None:
    from retrieval.thresholds import CATEGORY_THRESHOLDS

    print("=" * 72)
    print("ClauseGuard M26 Threshold Sanity Check")
    print("=" * 72)
    print(
        "HONEST FRAMING: this is a SANITY CHECK on M20's threshold table, NOT a "
        "calibration. Real calibration requires far more labeled data than the "
        "handful of rows currently in eval/labeled_set/m19_fixture.json. Expect "
        "INSUFFICIENT DATA flags below for most/all categories -- that is the "
        "correct, honest outcome at this stage, not a failure of this script."
    )
    print(
        "ADDITIONAL CAVEAT: this fixture's queries were built for M19's own Hit@K "
        "retrieval testing (each query deliberately targets its own clause), not for "
        "testing whether is_risky correlates with rerank score -- so a high score on a "
        "false-positive (is_risky=false) row below is expected under THAT original "
        "design goal, not necessarily evidence about risk-scoring behavior. Worth "
        "keeping in mind even once more labeled rows exist, unless future labeling work "
        "deliberately varies query-to-clause match strength independently of is_risky."
    )
    print("=" * 72)

    for row in result["skipped"]:
        print(
            f"SKIPPED contract_id={row.contract_id} clause_id={row.clause_id}: "
            f"no real Clause row found for this clause_id -- excluded, not scored as 0."
        )

    total_scored = sum(len(v) for v in result["by_category"].values())
    print()
    print(f"N = {total_scored} labeled row(s) scored (of {total_scored + len(result['skipped'])} total in the input file)")
    print()

    sufficient_count = 0
    for category, threshold in CATEGORY_THRESHOLDS.items():
        entries = result["by_category"].get(category, [])
        if len(entries) >= MIN_EXAMPLES_PER_CATEGORY:
            sufficient_count += 1
        for line in _category_report(category, threshold, entries):
            print(line)
        print()

    other_categories = sorted(set(result["by_category"]) - set(CATEGORY_THRESHOLDS))
    for category in other_categories:
        entries = result["by_category"][category]
        print(f"--- {category} (no per-category threshold applies -- excluded from the sanity table above) ---")
        print(f"  {len(entries)} labeled example(s) present, not assessed against any threshold.")
        print()

    print("=" * 72)
    print("PRIOR QUALITATIVE FINDINGS FROM OTHER MILESTONES' REVIEWS")
    print("(context only -- NOT derived by this script; attributed to their real")
    print("original source milestone; see each milestone's own review for full detail)")
    print("=" * 72)
    for milestone, finding in QUALITATIVE_FINDINGS:
        print(f"  {milestone}: {finding}")
    print()

    print("=" * 72)
    print(
        f"OVERALL: {sufficient_count} of {len(CATEGORY_THRESHOLDS)} categories had >= "
        f"{MIN_EXAMPLES_PER_CATEGORY} examples (enough for this script to attempt any "
        f"sanity judgment); {len(CATEGORY_THRESHOLDS) - sufficient_count} of "
        f"{len(CATEGORY_THRESHOLDS)} flagged INSUFFICIENT DATA. This is a sanity check, "
        f"not a calibration -- see HONEST FRAMING above."
    )
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
