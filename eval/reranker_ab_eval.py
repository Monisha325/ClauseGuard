"""M22: A/B eval answering the actual question M21's reranker exists to
resolve -- does cross-encoder reranking help retrieval quality on real
clause data, versus M11/M19's plain vector search? Reuses M19's Hit@K
methodology and labeled fixture (eval/retrieval_eval.py,
eval/labeled_set/m19_fixture.json) unchanged; this file only adds a
second ranking method (M21's rerank()) computed side by side with the
first, plus explicit per-row disagreement reporting.

REUSE, NOT REIMPLEMENTATION: hit_at_k(), _fetch_contract_clauses(),
_load_embedding_cache()/_save_embedding_cache()/_get_query_embedding()
(the Voyage-jitter fix), _vector_rank() (the Chroma-ANN-omission fix),
and the HIT_KS/TOP_K/SMOKE_TEST_N_THRESHOLD/EMBEDDING_CACHE_PATH/
MIN_SECONDS_BETWEEN_VOYAGE_CALLS constants are all imported directly from
retrieval_eval.py, not copied or rewritten. Both of M19's real,
hard-won determinism fixes (see that module's own DETERMINISM NOTE) are
therefore inherited automatically rather than re-earned or risked by a
second, subtly-different implementation.

HONESTY LABEL (same caveat M19 already established, still true here):
eval/labeled_set/m19_fixture.json is 5 rows against one small synthetic-
but-really-indexed contract -- nowhere near a statistically meaningful
sample. Every number this script prints is DIRECTIONAL evidence about
this reranker on this handful of rows, not a confident verdict on
whether reranking helps ClauseGuard's retrieval in general. Real
confidence requires M18's future hand-labeling work (15-20+ real
contracts).

CEILING EFFECT (a SEPARATE, independent limitation from the N-size one
above -- confirmed via this milestone's own independent review, not
merely suspected): every one of this fixture's 5 rows places the true
clause at position 1 of its candidate pool for BOTH vector-only and
reranked retrieval -- every query here IS a clean, unambiguous match
(see each row's own "easy case"/"moderate case" notes in
m19_fixture.json). Neither method has any room to outperform the other
on any row currently in this file. See _ceiling_effect_detected() below:
when this holds, print_report()/_conclusion() print an explicit caveat
that a tie here reflects an ABSENCE OF DISCRIMINATING POWER in the test
data, NOT evidence the two methods perform equivalently -- adding more
rows of this same easy character would not resolve that question;
meaningfully harder/more ambiguous labeled queries would.

KNOWN GAP IN CURRENT FIXTURE (documented here, not fabricated as a new
row -- see OUT OF SCOPE below): the same independent review found that a
jargon/abbreviation-heavy phrasing targeting the Confidentiality clause
("penalty for breaking an NDA" -- "NDA" never appears anywhere in the
clause's own text) made vector-only search fail badly (true clause
ranked LAST, position 7 of 7) while reranking partially recovered it
(position 5 of 7 -- still only just inside a top-5 cutoff). None of this
file's 5 real rows exercise jargon/abbreviation robustness at all; every
query here uses vocabulary reasonably close to its target clause's own
wording. Worth deliberately covering when M18's future hand-labeling
work expands this fixture -- not something this milestone adds a row
for on its own initiative.

WHY FULL-LENGTH RANKING (not just top TOP_K) FOR BOTH METHODS: both
_vector_rank() and _rerank_rank() below are called with
n_results=len(candidates) (every candidate for the row's contract, not
just 5) so that when a method's ranking DISAGREES with the other on
whether the true clause lands in the top-K, this script can report
exactly where the true clause actually landed (e.g. "reranked to
position 6 of 7") instead of just "not in top 5". Hit@K itself is
unaffected by this choice -- checking membership in the first K entries
of a full ranking is identical to checking a ranking that was already
truncated to K, for any K <= the full length.

DETERMINISM, WHY THE RERANK PATH NEEDS NO PACING/CACHING (verified
concretely, not assumed -- see this milestone's own testing step 5):
M21's rerank() is pure local CPU model inference (sentence-transformers'
CrossEncoder.predict()) -- no HTTP call of any kind once the model is
loaded. backend/Dockerfile sets HF_HUB_OFFLINE=1 on the running
container specifically so that even the ONE-TIME model-load step inside
_get_model() cannot make a live network call (it would hard-fail, not
silently succeed differently, if the local HF cache were somehow
missing) -- see that Dockerfile's own comment for the ~62s-per-request
regression this fixed in M21. This script inherits M19's Voyage-call
pacing/caching (the query embedding used to rank vector-only) but adds
NO equivalent pacing for the rerank path, because there is no live call
there to pace against -- confirmed at runtime below by asserting
HF_HUB_OFFLINE is actually set to "1" in this process's environment
before any row is evaluated, so this assumption fails loudly instead of
silently if the container config ever drifts.

RUNTIME DEPENDENCY NOTE: same as retrieval_eval.py -- this script imports
backend/ modules directly (via retrieval_eval.py and backend.retrieval.reranker)
and Chroma's hardcoded /app/chroma_data path, so it can only be run
inside the running `app` (or `worker`) container, e.g.:
    docker cp eval clauseguard-app-1:/app/eval
    docker compose exec app python /app/eval/reranker_ab_eval.py /app/eval/labeled_set/m19_fixture.json

OUT OF SCOPE (per M22 spec): no BM25/RRF fusion (M24), no threshold
sanity/calibration table (M26), no change to reranker.py itself (M21 is
done and independently verified -- a real quality problem found here is
a FINDING to report, not a bug to fix in this file), no new labeled rows
beyond M19's existing fixture.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `from loader import ...` / `from retrieval_eval import ...` work regardless of caller's cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so backend/'s top-level modules (db, models, embeddings, retrieval) resolve

from loader import LabeledClause, load_labeled_clauses  # noqa: E402
from retrieval_eval import (  # noqa: E402
    EMBEDDING_CACHE_PATH,
    HIT_KS,
    MIN_SECONDS_BETWEEN_VOYAGE_CALLS,
    SMOKE_TEST_N_THRESHOLD,
    _fetch_contract_clauses,
    _get_query_embedding,
    _load_embedding_cache,
    _save_embedding_cache,
    _vector_rank,
    hit_at_k,
)

TOP_K = 5  # same real endpoint TOP_K as retrieval_eval.py -- kept as its own name here (not imported) only because it's used purely for the printed "position N of M" framing below, not for any ranking computation (both methods rank the FULL candidate pool, see module docstring)


def _assert_rerank_path_is_local_only() -> None:
    """Concrete, not assumed, determinism guard: fails loudly if this
    process's environment doesn't actually have HF_HUB_OFFLINE=1 set --
    see module docstring for why that env var is what makes the rerank
    path's zero-live-call guarantee real rather than merely believed.
    """
    value = os.environ.get("HF_HUB_OFFLINE")
    if value != "1":
        raise RuntimeError(
            f"HF_HUB_OFFLINE={value!r} (expected '1') -- the reranking path's "
            f"determinism guarantee (no live HuggingFace network call) depends on "
            f"this being set (see backend/Dockerfile). Refusing to run rather than "
            f"silently produce results that may include live-network jitter."
        )


def _rerank_rank(query: str, candidates: list[tuple[str, str]]) -> list[str]:
    """Real reranked ranking: calls M21's actual reranker.rerank()
    directly (not a reimplementation) against the FULL candidate pool for
    this row's contract, returns clause_ids best-first. Pure local CPU
    inference -- see module docstring's DETERMINISM section.
    """
    from retrieval.reranker import rerank

    if not candidates:
        return []
    reranked = rerank(query, candidates)
    return [clause_id for clause_id, _score in reranked]


def _rank_position(ranked_ids: list[str], true_id: str) -> int | None:
    """1-indexed position of true_id in a FULL ranking, or None if
    (should never happen, since true_id is always one of the candidates
    fetched for its own contract_id) it's genuinely absent.
    """
    return ranked_ids.index(true_id) + 1 if true_id in ranked_ids else None


# --- evaluation orchestration ------------------------------------------------


def evaluate(labeled_rows: list[LabeledClause], db_session) -> dict:
    """Runs vector-only and reranked ranking for every labeled row that
    has real backing data. Both methods rank the FULL candidate pool for
    the row's contract (see module docstring for why) -- Hit@K is then
    computed by truncating each full ranking to K.
    """
    _assert_rerank_path_is_local_only()

    evaluated_rows = []
    skipped_rows = []
    last_voyage_call: float | None = None
    embedding_cache = _load_embedding_cache()

    for row in labeled_rows:
        candidates = _fetch_contract_clauses(row.contract_id, db_session)
        if not candidates:
            skipped_rows.append(row)
            continue

        true_id = str(row.clause_id)

        # Vector path: identical pacing/caching as retrieval_eval.py's own
        # evaluate() -- only a cache MISS counts as a live call worth pacing.
        import time

        from embeddings.voyage_client import MODEL as _MODEL

        cache_key = f"{_MODEL}::{row.query}"
        if cache_key not in embedding_cache and last_voyage_call is not None:
            wait = MIN_SECONDS_BETWEEN_VOYAGE_CALLS - (time.monotonic() - last_voyage_call)
            if wait > 0:
                time.sleep(wait)

        query_vector, was_cached = _get_query_embedding(row.query, embedding_cache)
        if not was_cached:
            last_voyage_call = time.monotonic()
            _save_embedding_cache(embedding_cache)  # persist immediately, same reasoning as retrieval_eval.py

        vector_ranked = _vector_rank(row.contract_id, query_vector, n_results=len(candidates))

        # Reranked path: no pacing needed -- pure local inference, see
        # module docstring's DETERMINISM section and _assert_rerank_path_is_local_only() above.
        reranked_ranked = _rerank_rank(row.query, candidates)

        evaluated_rows.append({
            "row": row,
            "true_id": true_id,
            "n_candidates": len(candidates),
            "vector_ranked": vector_ranked,
            "reranked_ranked": reranked_ranked,
        })

    n = len(evaluated_rows)
    aggregate = {"n": n, "vector": {}, "reranked": {}}
    for k in HIT_KS:
        aggregate["vector"][k] = (
            sum(1 for r in evaluated_rows if hit_at_k(r["vector_ranked"], r["true_id"], k)) / n
            if n else None
        )
        aggregate["reranked"][k] = (
            sum(1 for r in evaluated_rows if hit_at_k(r["reranked_ranked"], r["true_id"], k)) / n
            if n else None
        )

    return {"evaluated": evaluated_rows, "skipped": skipped_rows, "aggregate": aggregate}


def _fmt_rate(rate: float | None) -> str:
    return "N/A" if rate is None else f"{rate:.4f}"


def _disagreements(evaluated_rows: list[dict]) -> list[dict]:
    """Rows where vector-only and reranked disagree on Hit@K at ANY K --
    i.e. one method found the true clause within the top K and the other
    didn't, for at least one K in HIT_KS. Returned in evaluated_rows order.
    """
    out = []
    for r in evaluated_rows:
        disagreeing_ks = [
            k for k in HIT_KS
            if hit_at_k(r["vector_ranked"], r["true_id"], k) != hit_at_k(r["reranked_ranked"], r["true_id"], k)
        ]
        if disagreeing_ks:
            out.append({**r, "disagreeing_ks": disagreeing_ks})
    return out


def _ceiling_effect_detected(evaluated_rows: list[dict]) -> bool:
    """True if EVERY evaluated row already places the true clause at
    position 1 for BOTH methods -- i.e. neither method had any room to
    outperform the other on any row in this run. See module docstring's
    CEILING EFFECT note: this is a real, independently-confirmed
    limitation of the current fixture's rows (each constructed as a
    clean, unambiguous match -- see m19_fixture.json's own per-row
    notes), not a property of the two retrieval methods themselves. A
    tie under this condition reflects an ABSENCE OF DISCRIMINATING POWER
    in the test data, not evidence the methods perform equivalently.
    """
    if not evaluated_rows:
        return False
    return all(
        _rank_position(r["vector_ranked"], r["true_id"]) == 1
        and _rank_position(r["reranked_ranked"], r["true_id"]) == 1
        for r in evaluated_rows
    )


def _conclusion(aggregate: dict, evaluated_rows: list[dict]) -> str:
    """Mechanical, non-tuned verdict from the actual aggregate numbers --
    see module docstring: this is NOT a scripted outcome, it's a
    comparison of whichever real Hit@K numbers this run produced.

    Two SEPARATE, independent caveats can each apply and are both
    appended when they do (neither replaces the other -- see module
    docstring's CEILING EFFECT note for why a tie can be simultaneously
    "too little data" AND "data too easy to discriminate with"):
      - N-size: too few rows to be statistically meaningful.
      - Ceiling effect: every row already hits position 1 for both
        methods, so this run couldn't have shown a difference even with
        more rows of the same character.
    """
    n = aggregate["n"]
    if n == 0:
        return "NO CONCLUSION -- zero evaluable rows (see SKIPPED lines above)."

    reranked_wins = sum(1 for k in HIT_KS if aggregate["reranked"][k] > aggregate["vector"][k])
    vector_wins = sum(1 for k in HIT_KS if aggregate["vector"][k] > aggregate["reranked"][k])

    if reranked_wins > vector_wins:
        verdict = "KEEP -- reranking matched or beat vector-only at every K measured here"
    elif vector_wins > reranked_wins:
        verdict = "DROP -- vector-only matched or beat reranking at every K measured here"
    else:
        verdict = "DEFER -- tied/mixed results across K, no clear winner at this N"

    caveats = []
    if n < SMOKE_TEST_N_THRESHOLD:
        caveats.append(
            f"DIRECTIONAL ONLY, NOT CONCLUSIVE: N={n} is far below a "
            f"statistically meaningful sample ( < {SMOKE_TEST_N_THRESHOLD}). This is "
            f"a smoke-test-scale signal about THIS reranker on THIS handful of rows, "
            f"not a confident answer to 'does reranking help ClauseGuard in general' -- "
            f"that requires M18's future real hand-labeled data (15-20+ contracts)."
        )
    if _ceiling_effect_detected(evaluated_rows):
        caveats.append(
            f"CEILING EFFECT (a SEPARATE limitation from the N-size caveat above, if "
            f"present): every one of these {n} row(s) already places the true clause at "
            f"position 1 for BOTH methods -- these queries were constructed as clean, "
            f"unambiguous matches (see m19_fixture.json's own per-row notes), so neither "
            f"method had any room to outperform the other on any row here. This reflects "
            f"an ABSENCE OF DISCRIMINATING POWER in the current test data, NOT evidence "
            f"that vector-only and reranked retrieval perform equivalently -- more rows "
            f"like these would not resolve that question. Meaningfully harder/more "
            f"ambiguous labeled queries are needed to actually distinguish the two methods."
        )

    if caveats:
        return f"{verdict}. " + " ".join(caveats)
    return verdict


def print_report(result: dict) -> None:
    n = result["aggregate"]["n"]

    print("=" * 72)
    print("ClauseGuard M22 Reranker A/B Eval -- Hit@K (vector-only vs. reranked)")
    print("=" * 72)

    for row in result["skipped"]:
        print(
            f"SKIPPED contract_id={row.contract_id} clause_id={row.clause_id}: "
            f"no real Clause rows found for this contract_id -- excluded from N, not counted as a miss."
        )

    print()
    print(f"N = {n} labeled row(s) evaluated (of {n + len(result['skipped'])} total in the input file)")
    if n == 0:
        print("No evaluable rows -- cannot compute Hit@K. See SKIPPED lines above for why.")
        print("=" * 72)
        return

    if n < SMOKE_TEST_N_THRESHOLD:
        print(
            f"*** WARNING: N={n} is far below a statistically meaningful sample size "
            f"(< {SMOKE_TEST_N_THRESHOLD}). This run is DIRECTIONAL evidence about this "
            f"reranker's behavior on this handful of rows only -- it does NOT measure "
            f"real retrieval quality in general. Do not treat these numbers as "
            f"representative until real, hand-labeled data (M18's future work, "
            f"15-20+ real contracts) exists. ***"
        )
    if _ceiling_effect_detected(result["evaluated"]):
        print(
            f"*** CEILING EFFECT WARNING (SEPARATE from the N-size warning above, if "
            f"printed): every evaluated row already places the true clause at position 1 "
            f"for BOTH methods -- these queries were constructed as clean, unambiguous "
            f"matches (see m19_fixture.json's own per-row notes), so neither method had "
            f"any room to outperform the other on any row here. This reflects an ABSENCE "
            f"OF DISCRIMINATING POWER in the current test data, NOT evidence the two "
            f"methods perform equivalently -- more rows like these would not settle it; "
            f"meaningfully harder/more ambiguous labeled queries are needed. ***"
        )
    print()

    header = f"{'K':<4}{'Vector Hit@K':<16}{'Reranked Hit@K':<16}"
    print(header)
    print("-" * len(header))
    for k in HIT_KS:
        print(
            f"{k:<4}"
            f"{_fmt_rate(result['aggregate']['vector'][k]):<16}"
            f"{_fmt_rate(result['aggregate']['reranked'][k]):<16}"
        )

    print()
    print("Per-row detail:")
    for r in result["evaluated"]:
        row = r["row"]
        v_pos = _rank_position(r["vector_ranked"], r["true_id"])
        rr_pos = _rank_position(r["reranked_ranked"], r["true_id"])
        print(f"  query={row.query!r}  category={row.category!r}")
        print(f"    true clause_id = {r['true_id']}")
        print(f"    vector   ranked = {r['vector_ranked']}  (true clause at position {v_pos} of {r['n_candidates']})")
        print(f"    reranked ranked = {r['reranked_ranked']}  (true clause at position {rr_pos} of {r['n_candidates']})")
        for k in HIT_KS:
            v_hit = hit_at_k(r["vector_ranked"], r["true_id"], k)
            rr_hit = hit_at_k(r["reranked_ranked"], r["true_id"], k)
            flag = "  <-- DISAGREE" if v_hit != rr_hit else ""
            print(f"    Hit@{k}: vector={v_hit}  reranked={rr_hit}{flag}")
        print()

    disagreements = _disagreements(result["evaluated"])
    print("-" * 72)
    print(f"DISAGREEMENT CASES (vector and reranked differ on Hit@K at some K): {len(disagreements)}")
    print("-" * 72)
    if not disagreements:
        print("None -- vector-only and reranked agreed on top-K membership for every row at every K tested.")
    for d in disagreements:
        row = d["row"]
        v_pos = _rank_position(d["vector_ranked"], d["true_id"])
        rr_pos = _rank_position(d["reranked_ranked"], d["true_id"])
        print(f"  query={row.query!r}  category={row.category!r}")
        print(f"    true clause_id = {d['true_id']}")
        print(f"    vector   position = {v_pos} of {d['n_candidates']}")
        print(f"    reranked position = {rr_pos} of {d['n_candidates']}")
        print(f"    disagreement at K = {d['disagreeing_ks']}")
        direction = "reranking HELPED" if (rr_pos or 999) < (v_pos or 999) else "reranking HURT"
        print(f"    net effect: {direction} (lower position number = better)")
        print()

    print("-" * 72)
    print(f"CONCLUSION: {_conclusion(result['aggregate'], result['evaluated'])}")
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
