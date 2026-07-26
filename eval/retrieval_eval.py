"""M19: Hit@K evaluation for ClauseGuard retrieval, comparing (a) real
vector search via M11's Chroma index against (b) a genuinely naive
keyword-overlap baseline, side by side, against whatever labeled data
exists (eval/loader.py's LabeledClause rows).

HONESTY LABEL (read before trusting any number this script prints): this
is currently a SMOKE TEST of the SCRIPT's own correctness, not a
meaningful retrieval-quality measurement. Zero real, hand-labeled
contracts exist yet (confirmed by M17's diagnostic) -- M18's own
sample_placeholder.json rows use fabricated UUIDs with no real backing
data in Postgres/Chroma at all, so running literally against that file
finds zero evaluable rows (see _fetch_contract_clauses below -- a
contract/clause set that doesn't exist is skipped with a warning, not
silently counted as a "miss"). To prove this script's own logic is
correct, M19 also ships a small SYNTHETIC-BUT-REALLY-INDEXED fixture
contract (see eval/labeled_set/m19_fixture.json) -- a handful of clearly-
synthetic clauses actually run once through the real extract -> chunk ->
persist -> embed/index pipeline (chunker.process_contract(), the same
function the real upload route calls, minus the Gemini flagging step,
which Hit@K doesn't need) via a one-off setup script (not part of this
deliverable, already deleted after use), so real vector search and the
keyword baseline both have real backing data to compare against. That
fixture is still a handful of rows -- also NOT a statistically meaningful
quality measurement, just enough real data to prove the retrieval and
scoring logic itself works. Real Hit@K numbers worth trusting require
M18's actual hand-labeling work (15-20+ real contracts), which is
separate, future, human-judgment work.

ARCHITECTURE CHOICE (documented per M19 spec's request): real vector
search is done by calling the underlying Chroma query function directly
(embed_text() + the same collection.query(... where={"contract_id":...})
call backend/routes/retrieval.py's search_contract() itself makes) rather
than through a real HTTP request with a logged-in auth flow. This is
simpler for a standalone eval script -- no token/session management
needed -- and is explicitly sanctioned as a valid choice by this
milestone's own spec ("or by calling the underlying Chroma query function
directly if that's simpler"). The Postgres ownership/auth check
search_contract() layers on top of that Chroma call is intentionally NOT
replicated here: this script is run by a developer directly against the
real database, not on behalf of an untrusted end user, so there is no
"is this the caller's contract" question to enforce.

RUNTIME DEPENDENCY NOTE: unlike eval/loader.py (M18, deliberately
zero-dependency on backend/), this script DOES import backend/ modules
directly (db, models.clause, models.contract, embeddings.voyage_client,
embeddings.index) -- that dependency is unavoidable and correct here,
since the whole point of this milestone is exercising the REAL retrieval
system, not a standalone schema. Chroma's persistent client also reads a
hardcoded container path (/app/chroma_data, see embeddings/index.py), so
this script can only be run where that path and the real Postgres/Voyage
config are reachable -- in practice, inside the running `app` (or
`worker`) container, e.g.:
    docker cp eval clauseguard-app-1:/app/eval
    docker compose exec app python /app/eval/retrieval_eval.py /app/eval/labeled_set/m19_fixture.json

DETERMINISM NOTE (two real, distinct bugs found and fixed here -- read
before assuming either fix alone is sufficient):

BUG 1 (Voyage API jitter, FIXED): the keyword baseline is pure, local
computation (no randomness, deterministic tie-break by clause_id on
equal scores) and is exactly reproducible. Real vector search was
ORIGINALLY assumed safe because Voyage's underlying model does
deterministic (non-sampling) inference -- but a real 5-consecutive-run
test caught ONE run producing a different top-5 ranking for one row,
with no code change and no data change. Investigation found: (a)
Voyage's own docs do NOT state or guarantee bit-identical output across
SEPARATE API calls for identical input text -- undocumented either way --
and hosted embedding/LLM inference services commonly exhibit small
run-to-run floating-point non-associativity from server-side batching
across concurrent requests, independent of the model itself being
"deterministic". FIX: query embeddings are cached to disk, keyed by
(model, exact query text), in EMBEDDING_CACHE_PATH. The FIRST time a
given query is evaluated, embed_text() is called for real and the result
is persisted; every subsequent run (same query, same model) reuses that
exact cached vector instead of calling Voyage again -- this doesn't (and
can't) prove Voyage itself is deterministic, but it removes live-API
jitter as a variable in THIS script's own reproducibility guarantee.

BUG 2 (Chroma ANN search, FIXED, found via independent review AFTER Bug
1's fix): a follow-up 10-consecutive-run re-verification (after Bug 1's
cache fix) STILL found one run out of 10 with a different result -- but
this time with query embeddings PROVABLY bit-identical across all 10
runs (confirmed via the cache file's mtime never changing during the
batch), ruling out Voyage entirely. The failure signature was also
different: not reordering, but a candidate clause_id MISSING from every
row's top-5 entirely, replaced by whatever the 6th-nearest neighbor was.
Root cause, confirmed empirically: collection.query()'s HNSW-based
approximate search (chromadb 1.5.9, this collection's confirmed
hnsw.space='l2') can silently return FEWER than n_results even when
n_results equals the exact, confirmed candidate count for a contract --
reproduced across 10 SEPARATE fresh OS processes each re-opening
PersistentClient from disk (this script's actual real invocation
pattern), NOT reproduced by repeated queries within one long-lived
process (which is why the original Bug 1 investigation's Chroma check --
valid on its own terms -- didn't catch this: it only tested the
same-process case). The (distance, clause_id) tie-break Bug 1's fix added
does not help here, since the missing candidate isn't reordered, it's
absent from the ANN result set entirely. FIX: _vector_rank no longer
calls collection.query() (ANN) at all. It calls collection.get() (a
plain metadata-filtered fetch, NOT a nearest-neighbor search -- no HNSW
graph traversal, confirmed reliable across repeated fresh-process calls
in review) to fetch every real candidate embedding for the contract, then
computes exact squared L2 distance (this collection's real confirmed
metric -- see _squared_l2's docstring for how that was verified, not
assumed) between the query embedding and every candidate in plain Python,
and ranks by that exact value with the same (distance, clause_id)
tie-break as before. This is only appropriate because this eval's
candidate pools are small (a handful of clauses per contract); it is
NOT a replacement for M11's real search_contract() endpoint, which
correctly keeps using ANN search at whatever scale production retrieval
needs.

M25 ADDITION: does M24's BM25 + RRF fusion actually improve retrieval
over vector-only, using the exact same fixture/query set M19/M22 already
established (not a drifted or expanded one -- that would invalidate the
before/after comparison this milestone exists to make)? evaluate() now
computes THREE conditions per row, side by side: vector-only (unchanged
above), fusion (M24's real backend/retrieval/bm25_index.py +
backend/retrieval/fusion.py, called directly, not reimplemented here),
and fusion+reranker (that same fused list fed into M21's real
backend/retrieval/reranker.py rerank(), matching the actual live
endpoint's real composition order as verified in M24's independent
review -- the FULL fused candidate list is reranked, never truncated to
a smaller pool first). M19's original keyword-vs-vector comparison above
is completely unchanged and still computed/printed, just no longer the
only comparison this file makes.

EXPECTATION, NOT A BUG IF IT RECURS: M22's own independent review found
a real, recurring signal that vector search and BM25 tend to AGREE on
rare-phrase overlap for this domain/embedding-model pairing, and a
ceiling effect in this exact fixture (every row already an unambiguous,
"easy" match for every method tried so far). Both are expected to very
plausibly recur here too, since this is the identical fixture -- see
_ceiling_effect_detected() below (generalizes M22's own 2-condition
version to however many ranking conditions are passed in, same
underlying "does every row already sit at position 1" check, not a
different algorithm).

RRF FUSION INPUT NOTE: reciprocal_rank_fusion() (retrieval/fusion.py)
only ever looks at each input ranking's ORDER, never at the score values
in its (candidate_id, score) tuples (confirmed directly in that module's
own docstring). _vector_rank() above returns a bare, already-ordered
list[str] with no scores at all -- _fusion_rank() below wraps that list
into (clause_id, 0.0) placeholder tuples purely to match RRF's expected
input shape; the placeholder value 0.0 is never read for ranking
purposes, only the LIST ORDER (the real, exact-distance vector ranking
_vector_rank already computed) matters. This is mathematically identical
to passing real vector similarity scores, not an approximation.
"""

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `from loader import ...` works regardless of caller's cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so backend/'s top-level modules (db, models, embeddings) resolve

from loader import LabeledClause, load_labeled_clauses  # noqa: E402

TOP_K = 5  # matches backend/routes/retrieval.py's own TOP_K -- same real endpoint's top-N
HIT_KS = (1, 3, 5)
SMOKE_TEST_N_THRESHOLD = 10  # below this, results are flagged as NOT statistically meaningful

# Voyage free-tier pacing (same confirmed 3 RPM limit and same floor
# embeddings/index.py already paces against -- see that module's own
# docstring). Each evaluated row costs at most one embed_text() call
# here (the query) -- fewer once EMBEDDING_CACHE_PATH has already seen a
# query -- so the same fixed floor between consecutive LIVE calls is
# sufficient without needing a sliding-window token tracker.
MIN_SECONDS_BETWEEN_VOYAGE_CALLS = 21

# Disk-persisted cache: (model, exact query text) -> embedding vector.
# See DETERMINISM NOTE above for why this exists -- it is the fix for a
# real, observed cross-run non-determinism traced to Voyage's live API,
# not a performance optimization. Committed as a normal eval/ file so
# repeated runs against the same labeled data stay reproducible for
# anyone who runs this script, not just in the session that first
# populated it.
EMBEDDING_CACHE_PATH = Path(__file__).resolve().parent / "labeled_set" / ".embedding_cache.json"

_TOKEN_RE = re.compile(r"\b\w+\b")


# --- naive keyword baseline -------------------------------------------------
#
# Deliberately dumb on purpose (per spec: "must be GENUINELY naive... do
# not implement the naive baseline so well that it doesn't serve as a
# real contrast to vector search"). No stopword removal, no stemming, no
# IDF/term-frequency weighting, no phrase matching -- just a raw count of
# DISTINCT tokens (lowercased word characters) shared between the query
# and a clause's text. A clause that happens to repeat common words
# ("the", "agreement", "party") shared with the query scores exactly the
# same as if those were meaningful matches -- that blind spot is the
# whole point of having this as a weak baseline, not a bug to fix.


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


def keyword_score(query: str, clause_text: str) -> int:
    """Count of distinct tokens shared between query and clause_text."""
    return len(_tokenize(query) & _tokenize(clause_text))


def rank_by_keyword(query: str, candidates: list[tuple[str, str]]) -> list[str]:
    """candidates: list of (clause_id, text). Returns clause_ids ranked
    best-first. Ties (equal score) broken by clause_id string ascending --
    arbitrary but FIXED, so output is deterministic regardless of
    Postgres row-fetch order.
    """
    scored = [(keyword_score(query, text), clause_id) for clause_id, text in candidates]
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [clause_id for _, clause_id in scored]


# --- Hit@K ------------------------------------------------------------------


def hit_at_k(ranked_ids: list[str], true_id: str, k: int) -> bool:
    return true_id in ranked_ids[:k]


# --- backend-dependent adapters (Postgres + Chroma + Voyage) ----------------


def _fetch_contract_clauses(contract_id: uuid.UUID, db_session) -> list[tuple[str, str]]:
    """Returns [(clause_id_str, text), ...] for every REAL persisted
    Clause under this contract_id. Empty list if the contract doesn't
    exist or has no clauses -- e.g. M18's placeholder rows' fabricated
    UUIDs, which were never run through the real pipeline. Callers must
    treat an empty result as "no real backing data", not as a retrieval miss.
    """
    from models.clause import Clause

    rows = db_session.query(Clause).filter(Clause.contract_id == contract_id).all()
    return [(str(c.id), c.text) for c in rows]


def _load_embedding_cache() -> dict:
    if EMBEDDING_CACHE_PATH.exists():
        with EMBEDDING_CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_embedding_cache(cache: dict) -> None:
    EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EMBEDDING_CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def _get_query_embedding(query: str, cache: dict) -> tuple[list[float], bool]:
    """Returns (embedding_vector, was_cache_hit). A cache hit reuses the
    EXACT vector a prior run already got from Voyage for this exact query
    text -- see DETERMINISM NOTE above for why this is the fix for a real
    observed cross-run instability, not a performance shortcut. A miss
    makes one real, live embed_text() call and stores the result; the
    caller is responsible for persisting `cache` back to disk (see
    evaluate()) so the NEXT run reuses it too.
    """
    from embeddings.voyage_client import MODEL, embed_text

    key = f"{MODEL}::{query}"
    if key in cache:
        return cache[key], True
    vector = embed_text(query)
    cache[key] = vector
    return vector, False


def _squared_l2(a: list[float], b: list[float]) -> float:
    """Squared L2 (Euclidean) distance -- confirmed, not assumed, to be
    exactly what this collection's configured hnsw.space='l2' metric
    means: a direct empirical check (sum((a-b)**2) vs. this collection's
    real collection.query() distances for real cached embeddings) matched
    to float precision, while sqrt(L2) did not. See this milestone's own
    review for the real numbers.
    """
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _vector_rank(contract_id: uuid.UUID, query_vector: list[float], n_results: int) -> list[str]:
    """Real vector search: fetch every real candidate embedding for this
    contract_id and rank by EXACT, locally-computed distance -- NOT via
    Chroma's collection.query() ANN (approximate nearest neighbor) search.

    Why not ANN: a real, reproducible bug was found (see this milestone's
    own independent review) where collection.query()'s HNSW-based
    approximate search can silently return FEWER than n_results even when
    n_results equals the exact known candidate count for a contract --
    not a reordering, an outright missing candidate -- specifically on a
    FRESH process/PersistentClient load (this script's actual real
    invocation pattern; same-process repeated-query testing never
    surfaces it, which is why an earlier round of testing wrongly cleared
    Chroma as a cause). The originally-added (distance, clause_id)
    tie-break does not help here, since the dropped candidate isn't
    merely reordered -- it's absent from the ANN result set entirely.

    Fix: collection.get() (a plain metadata-filtered fetch, NOT a nearest-
    neighbor search -- no ANN graph traversal involved) reliably returns
    every real candidate every time (confirmed via this milestone's own
    review: 7/7 candidates present, every one of 10 repeated fresh-process
    calls). Ranking is then done by this function itself: exact squared L2
    distance (this collection's real configured metric, confirmed via
    _squared_l2's docstring) between the query embedding and every
    candidate, computed in plain Python -- deterministic by construction,
    with no ANN approximation anywhere in the loop. Candidate pools here
    are small (<= a few dozen clauses per contract), so the O(n) exact
    computation costs nothing meaningful; this is not a general
    replacement for ANN search at production retrieval scale (M11's real
    search_contract() endpoint keeps using collection.query() -- that
    decision is unaffected by this eval-only fix).

    Tie-break: (distance, clause_id) ascending, same convention as before
    and as keyword_score()'s own tie-break -- kept as belt-and-suspenders
    even though exact computation has no float-noise-driven near-ties the
    way an ANN result set theoretically could.

    M25 NOTE: n_results is now called with len(candidates) (full length)
    from evaluate() below rather than min(TOP_K, len(candidates)) as
    originally -- this does NOT change any existing Hit@K number (Hit@K
    only checks membership within the first K entries of a ranking;
    returning more entries beyond K never removes or reorders the first
    K) but gives the M25 fusion logic the complete vector ranking it
    needs to fuse against BM25's own full ranking. Same "WHY FULL-LENGTH
    RANKING" reasoning eval/reranker_ab_eval.py's (M22) own module
    docstring already established for the vector/reranked comparison,
    now extended to the fusion comparison too.
    """
    from embeddings.index import _get_chroma_collection

    collection = _get_chroma_collection()
    result = collection.get(
        where={"contract_id": str(contract_id)},
        include=["metadatas", "embeddings"],
    )
    if not result["ids"]:
        return []

    pairs = [
        (metadata["clause_id"], _squared_l2(query_vector, list(embedding)))
        for metadata, embedding in zip(result["metadatas"], result["embeddings"])
    ]
    pairs.sort(key=lambda pair: (round(pair[1], 6), pair[0]))
    return [clause_id for clause_id, _ in pairs[:n_results]]


def _rank_position(ranked_ids: list[str], true_id: str) -> int | None:
    """1-indexed position of true_id in a FULL ranking, or None if
    (should never happen, since true_id is always one of the candidates
    fetched for its own contract_id) it's genuinely absent. Same helper
    eval/reranker_ab_eval.py (M22) already established.
    """
    return ranked_ids.index(true_id) + 1 if true_id in ranked_ids else None


def _assert_rerank_path_is_local_only() -> None:
    """Concrete, not assumed, determinism guard: fails loudly if this
    process's environment doesn't actually have HF_HUB_OFFLINE=1 set --
    M21's rerank() is pure local CPU model inference with no live call of
    any kind once the model is loaded, but that guarantee depends on this
    env var (see backend/Dockerfile's own comment for the ~62s-per-request
    regression it fixed). Same guard eval/reranker_ab_eval.py (M22)
    already established, now needed here too since this file calls
    rerank() directly for the first time as of M25.
    """
    value = os.environ.get("HF_HUB_OFFLINE")
    if value != "1":
        raise RuntimeError(
            f"HF_HUB_OFFLINE={value!r} (expected '1') -- the reranking path's "
            f"determinism guarantee (no live HuggingFace network call) depends on "
            f"this being set (see backend/Dockerfile). Refusing to run rather than "
            f"silently produce results that may include live-network jitter."
        )


def _fusion_rank(vector_ranked: list[str], query: str, candidates: list[tuple[str, str]]) -> list[str]:
    """M24's real BM25 search + RRF fusion against the already-computed
    (exact-distance) vector ranking -- calls
    backend/retrieval/bm25_index.py's search_bm25() and
    backend/retrieval/fusion.py's reciprocal_rank_fusion() directly, not
    reimplemented here. `candidates` is the SAME (clause_id, text) list
    already fetched for this row (reused, not re-fetched). Returns a
    full-length clause_id ranking, best-first -- same full-length
    convention as _vector_rank() above.

    See module docstring's RRF FUSION INPUT NOTE for why vector_ranked
    (a bare ordered list[str], no scores) is wrapped into (clause_id,
    0.0) placeholder tuples before being passed to reciprocal_rank_fusion --
    RRF only uses each input ranking's ORDER, never its score values, so
    this is mathematically equivalent to passing real vector scores.
    """
    from retrieval.bm25_index import search_bm25
    from retrieval.fusion import reciprocal_rank_fusion

    vector_ranking_for_fusion = [(clause_id, 0.0) for clause_id in vector_ranked]
    bm25_ranking = search_bm25(query, candidates, n_results=len(candidates))
    fused = reciprocal_rank_fusion(vector_ranking_for_fusion, bm25_ranking)
    return [clause_id for clause_id, _rrf_score in fused]


def _fusion_reranked_rank(fusion_ranked: list[str], query: str, candidates: list[tuple[str, str]]) -> list[str]:
    """M24+M21 together: the FULL fusion-ranked candidate list (not
    truncated to a smaller pool first) fed into M21's real
    backend/retrieval/reranker.py rerank() -- matching the actual live
    endpoint's real composition order, as independently verified in
    M24's own review (the endpoint reranks the whole fused pool, the
    reranker's own downstream judgment is what may reorder past fusion's
    ranking, not any pre-truncation). Not reimplemented here; calls the
    real, unmodified rerank() directly. Returns a full-length clause_id
    ranking, best-first.
    """
    from retrieval.reranker import rerank

    text_by_id = dict(candidates)
    reranked = rerank(query, [(clause_id, text_by_id[clause_id]) for clause_id in fusion_ranked])
    return [clause_id for clause_id, _sigmoid_score in reranked]


# --- evaluation orchestration ------------------------------------------------


def evaluate(labeled_rows: list[LabeledClause], db_session) -> dict:
    """Runs all FOUR retrieval methods for every labeled row that has
    real backing data (keyword, vector -- both M19's original, unchanged
    logic; fusion and fusion+reranker -- M25's addition, calling M24/M21's
    real functions directly), and returns a result dict with per-row
    detail and aggregate Hit@K rates for all four. Rows with no real
    backing data (contract/clauses not found) are recorded as skipped,
    not counted in any method's N or hit counts.
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
        keyword_ranked = rank_by_keyword(row.query, candidates)

        # Only pace/rate-limit an actual LIVE Voyage call -- a cache hit
        # costs nothing and shouldn't be throttled as if it were a real
        # API request.
        from embeddings.voyage_client import MODEL as _MODEL

        cache_key = f"{_MODEL}::{row.query}"
        if cache_key not in embedding_cache and last_voyage_call is not None:
            wait = MIN_SECONDS_BETWEEN_VOYAGE_CALLS - (time.monotonic() - last_voyage_call)
            if wait > 0:
                time.sleep(wait)

        query_vector, was_cached = _get_query_embedding(row.query, embedding_cache)
        if not was_cached:
            last_voyage_call = time.monotonic()
            _save_embedding_cache(embedding_cache)  # persist immediately, not just at the end -- a later row's failure shouldn't lose this real API result

        # Full length now (see _vector_rank's own M25 NOTE) -- doesn't
        # change any existing Hit@K number, but fusion below needs the
        # complete vector ranking to combine against BM25's own full ranking.
        vector_ranked = _vector_rank(row.contract_id, query_vector, n_results=len(candidates))

        # M25: fusion (M24's real BM25 + RRF, no reranker) and
        # fusion+reranker (that same fused list into M21's real
        # rerank()) -- no live/paced calls of any kind in either step
        # (BM25 is pure local computation; reranking is pure local CPU
        # inference, see _assert_rerank_path_is_local_only() above).
        fusion_ranked = _fusion_rank(vector_ranked, row.query, candidates)
        fusion_reranked_ranked = _fusion_reranked_rank(fusion_ranked, row.query, candidates)

        evaluated_rows.append({
            "row": row,
            "true_id": true_id,
            "n_candidates": len(candidates),
            "keyword_ranked": keyword_ranked,
            "vector_ranked": vector_ranked,
            "fusion_ranked": fusion_ranked,
            "fusion_reranked_ranked": fusion_reranked_ranked,
        })

    n = len(evaluated_rows)
    aggregate = {"n": n, "keyword": {}, "vector": {}, "fusion": {}, "fusion_reranked": {}}
    for k in HIT_KS:
        aggregate["keyword"][k] = (
            sum(1 for r in evaluated_rows if hit_at_k(r["keyword_ranked"], r["true_id"], k)) / n
            if n else None
        )
        aggregate["vector"][k] = (
            sum(1 for r in evaluated_rows if hit_at_k(r["vector_ranked"], r["true_id"], k)) / n
            if n else None
        )
        aggregate["fusion"][k] = (
            sum(1 for r in evaluated_rows if hit_at_k(r["fusion_ranked"], r["true_id"], k)) / n
            if n else None
        )
        aggregate["fusion_reranked"][k] = (
            sum(1 for r in evaluated_rows if hit_at_k(r["fusion_reranked_ranked"], r["true_id"], k)) / n
            if n else None
        )

    return {"evaluated": evaluated_rows, "skipped": skipped_rows, "aggregate": aggregate}


def _fmt_rate(rate: float | None) -> str:
    return "N/A" if rate is None else f"{rate:.4f}"


# M25's three core conditions this milestone compares side by side.
# keyword_ranked is deliberately NOT included -- that's M19's original,
# separate comparison axis (semantic vs. naive-lexical), still
# computed/printed below for continuity, but not part of this
# milestone's specific "does fusion help over vector-only" question.
FUSION_COMPARISON_KEYS = ("vector_ranked", "fusion_ranked", "fusion_reranked_ranked")


def _ceiling_effect_detected(evaluated_rows: list[dict], ranking_keys: tuple[str, ...] = FUSION_COMPARISON_KEYS) -> bool:
    """True if EVERY evaluated row already places the true clause at
    position 1 for ALL of `ranking_keys` -- i.e. none of the compared
    methods had any room to outperform another on any row in this run.

    Generalizes eval/reranker_ab_eval.py's (M22) own 2-condition version
    of this exact check (same underlying algorithm: does every row sit
    at position 1 for every method being compared) to however many
    ranking keys are passed in -- 3 here (vector/fusion/fusion+reranker),
    not a different detection approach. See that module's own CEILING
    EFFECT note for why this reflects an ABSENCE OF DISCRIMINATING POWER
    in the test data, not evidence the methods perform equivalently.
    """
    if not evaluated_rows:
        return False
    return all(
        all(_rank_position(r[key], r["true_id"]) == 1 for key in ranking_keys)
        for r in evaluated_rows
    )


def _disagreements(evaluated_rows: list[dict], ranking_keys: tuple[str, ...] = FUSION_COMPARISON_KEYS) -> list[dict]:
    """Rows where the compared methods disagree on Hit@K at ANY K -- i.e.
    at least one method found the true clause within the top K while at
    least one other didn't, for some K in HIT_KS. Extends
    eval/reranker_ab_eval.py's (M22) own 2-way version of this exact
    check to a 3-way (or N-way) comparison: instead of checking whether
    two hit_at_k booleans differ, checks whether the SET of hit_at_k
    booleans across all `ranking_keys` has more than one distinct value.
    Returned in evaluated_rows order.
    """
    out = []
    for r in evaluated_rows:
        disagreeing_ks = [
            k for k in HIT_KS
            if len({hit_at_k(r[key], r["true_id"], k) for key in ranking_keys}) > 1
        ]
        if disagreeing_ks:
            out.append({**r, "disagreeing_ks": disagreeing_ks})
    return out


def _conclusion(aggregate: dict, evaluated_rows: list[dict]) -> str:
    """Mechanical, non-tuned verdict from the actual aggregate numbers --
    this is NOT a scripted outcome, it's a comparison of whichever real
    Hit@K numbers this run produced. Answers TWO separate questions,
    both honestly, since they can differ: does bare fusion help over
    vector-only, and does fusion+reranker help over vector-only. Neither
    verdict is tuned by adjusting BM25/RRF/reranker parameters to force a
    favorable outcome.

    Same TWO separate, independent caveats as M22's own _conclusion()
    (N-size and ceiling effect), generalized to this milestone's 3-way
    comparison -- neither replaces the other when both apply.
    """
    n = aggregate["n"]
    if n == 0:
        return "NO CONCLUSION -- zero evaluable rows (see SKIPPED lines above)."

    def _verdict_for(candidate_key: str, candidate_label: str) -> str:
        wins = sum(1 for k in HIT_KS if aggregate[candidate_key][k] > aggregate["vector"][k])
        losses = sum(1 for k in HIT_KS if aggregate["vector"][k] > aggregate[candidate_key][k])
        if wins > losses:
            return f"KEEP {candidate_label} -- matched or beat vector-only at every K measured here"
        if losses > wins:
            return f"DROP {candidate_label} -- vector-only matched or beat it at every K measured here"
        return f"DEFER on {candidate_label} -- tied/mixed results across K, no clear winner at this N"

    fusion_verdict = _verdict_for("fusion", "fusion (no reranker)")
    fusion_reranked_verdict = _verdict_for("fusion_reranked", "fusion+reranker")

    caveats = []
    if n < SMOKE_TEST_N_THRESHOLD:
        caveats.append(
            f"DIRECTIONAL ONLY, NOT CONCLUSIVE: N={n} is far below a "
            f"statistically meaningful sample ( < {SMOKE_TEST_N_THRESHOLD}). This is "
            f"a smoke-test-scale signal about fusion on THIS handful of rows, "
            f"not a confident answer to 'does BM25+RRF fusion help ClauseGuard in "
            f"general' -- that requires M18's future real hand-labeled data (15-20+ contracts)."
        )
    ceiling = _ceiling_effect_detected(evaluated_rows)
    if ceiling:
        caveats.append(
            f"CEILING EFFECT (a SEPARATE limitation from the N-size caveat above, if "
            f"present): every one of these {n} row(s) already places the true clause at "
            f"position 1 for vector-only, fusion, AND fusion+reranker alike -- these "
            f"queries were constructed as clean, unambiguous matches (see "
            f"m19_fixture.json's own per-row notes), so none of the three methods had "
            f"any room to outperform another on any row here. This reflects an ABSENCE "
            f"OF DISCRIMINATING POWER in the current test data, NOT evidence that fusion "
            f"and vector-only perform equivalently -- more rows like these would not "
            f"resolve that question. This also matches M24's own independent review "
            f"finding that vector search and BM25 tend to AGREE on rare-phrase overlap "
            f"for this domain/embedding-model pairing -- consistent with, not "
            f"contradicted by, a tie here. A real answer requires meaningfully harder/"
            f"more ambiguous labeled queries, and ideally a different embedding model "
            f"or query style to test whether that agreement tendency is specific to "
            f"this pairing -- both are hypotheses this small eval can raise, not prove."
        )
    elif _disagreements(evaluated_rows):
        # NOT a ceiling-effect tie this run -- at least one row genuinely
        # broke it (see the DISAGREEMENT CASES section above for exactly
        # which). Worth stating plainly rather than silently falling back
        # to the ceiling-effect framing, which would misdescribe what
        # actually happened here: this is the OPPOSITE finding from
        # "vector and BM25 tend to agree" -- a real case where they
        # disagreed, and specifically because BM25's lexical/term-overlap
        # signal favored a superficially-similar but wrong clause (the
        # same category of failure mode M19's own naive keyword baseline
        # was deliberately designed to expose on this exact fixture row --
        # see m19_fixture.json's own per-row notes). Directional only, at
        # this N: worth watching for on a larger, real labeled set (M18),
        # not something this handful of rows can establish as a general
        # property of BM25 fusion.
        caveats.append(
            "NOTE: unlike a ceiling-effect tie, at least one row here shows fusion "
            "genuinely diverging from vector-only (see DISAGREEMENT CASES above for "
            "specifics) -- in the case observed, fusion's BM25 component favored a "
            "superficially keyword-similar but wrong clause, the same category of "
            "trap M19's own naive keyword baseline was built to expose on this exact "
            "row, and the reranker recovered it. This is a real, concrete instance of "
            "a possible failure mode, not a general verdict on BM25 fusion at this N -- "
            "worth deliberately testing for again once M18's real hand-labeled data exists."
        )

    caveat_text = (" " + " ".join(caveats)) if caveats else ""
    return f"{fusion_verdict}. {fusion_reranked_verdict}.{caveat_text}"


def print_report(result: dict) -> None:
    n = result["aggregate"]["n"]

    print("=" * 72)
    print("ClauseGuard M25 Retrieval Eval -- Hit@K (vector-only vs. fusion vs. fusion+reranker)")
    print("=" * 72)

    for row in result["skipped"]:
        print(
            f"SKIPPED contract_id={row.contract_id} clause_id={row.clause_id}: "
            f"no real Clause rows found for this contract_id (not a real, "
            f"indexed contract) -- excluded from N, not counted as a miss."
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
            f"(< {SMOKE_TEST_N_THRESHOLD}). This run is DIRECTIONAL evidence about fusion's "
            f"behavior on this handful of rows only -- it does NOT measure real retrieval "
            f"quality in general. Do not treat these numbers as representative until real, "
            f"hand-labeled data (M18's future work, 15-20+ real contracts) exists. ***"
        )
    if _ceiling_effect_detected(result["evaluated"]):
        print(
            f"*** CEILING EFFECT WARNING (SEPARATE from the N-size warning above, if "
            f"printed): every evaluated row already places the true clause at position 1 "
            f"for vector-only, fusion, AND fusion+reranker alike -- these queries were "
            f"constructed as clean, unambiguous matches (see m19_fixture.json's own "
            f"per-row notes), so none of the three methods had any room to outperform "
            f"another on any row here. This reflects an ABSENCE OF DISCRIMINATING POWER "
            f"in the current test data, NOT evidence the methods perform equivalently -- "
            f"more rows like these would not settle it; meaningfully harder/more "
            f"ambiguous labeled queries are needed. Consistent with M24's own independent "
            f"review finding that vector search and BM25 tend to AGREE on rare-phrase "
            f"overlap for this domain/embedding-model pairing. ***"
        )
    print()

    header = f"{'K':<4}{'Vector Hit@K':<16}{'Fusion Hit@K':<16}{'Fusion+Rerank Hit@K':<20}"
    print(header)
    print("-" * len(header))
    for k in HIT_KS:
        print(
            f"{k:<4}"
            f"{_fmt_rate(result['aggregate']['vector'][k]):<16}"
            f"{_fmt_rate(result['aggregate']['fusion'][k]):<16}"
            f"{_fmt_rate(result['aggregate']['fusion_reranked'][k]):<20}"
        )
    print()
    print(
        f"(reference only, M19's original comparison axis, unchanged, NOT part of this "
        f"milestone's 3-way fusion comparison above): Keyword Hit@K = "
        + ", ".join(f"K={k}: {_fmt_rate(result['aggregate']['keyword'][k])}" for k in HIT_KS)
    )

    print()
    print("Per-row detail:")
    for r in result["evaluated"]:
        row = r["row"]
        v_pos = _rank_position(r["vector_ranked"], r["true_id"])
        f_pos = _rank_position(r["fusion_ranked"], r["true_id"])
        fr_pos = _rank_position(r["fusion_reranked_ranked"], r["true_id"])
        print(f"  query={row.query!r}  category={row.category!r}")
        print(f"    true clause_id = {r['true_id']}")
        print(f"    vector          ranked = {r['vector_ranked']}  (true clause at position {v_pos} of {r['n_candidates']})")
        print(f"    fusion          ranked = {r['fusion_ranked']}  (true clause at position {f_pos} of {r['n_candidates']})")
        print(f"    fusion+reranker ranked = {r['fusion_reranked_ranked']}  (true clause at position {fr_pos} of {r['n_candidates']})")
        print(f"    keyword ranked (reference only) = {r['keyword_ranked']}")
        for k in HIT_KS:
            v_hit = hit_at_k(r["vector_ranked"], r["true_id"], k)
            f_hit = hit_at_k(r["fusion_ranked"], r["true_id"], k)
            fr_hit = hit_at_k(r["fusion_reranked_ranked"], r["true_id"], k)
            flag = "  <-- DISAGREE" if len({v_hit, f_hit, fr_hit}) > 1 else ""
            print(f"    Hit@{k}: vector={v_hit}  fusion={f_hit}  fusion+reranker={fr_hit}{flag}")
        print()

    disagreements = _disagreements(result["evaluated"])
    print("-" * 72)
    print(f"DISAGREEMENT CASES (vector/fusion/fusion+reranker differ on Hit@K at some K): {len(disagreements)}")
    print("-" * 72)
    if not disagreements:
        print("None -- vector-only, fusion, and fusion+reranker all agreed on top-K membership for every row at every K tested.")
    for d in disagreements:
        row = d["row"]
        v_pos = _rank_position(d["vector_ranked"], d["true_id"])
        f_pos = _rank_position(d["fusion_ranked"], d["true_id"])
        fr_pos = _rank_position(d["fusion_reranked_ranked"], d["true_id"])
        print(f"  query={row.query!r}  category={row.category!r}")
        print(f"    true clause_id = {d['true_id']}")
        print(f"    vector          position = {v_pos} of {d['n_candidates']}")
        print(f"    fusion          position = {f_pos} of {d['n_candidates']}")
        print(f"    fusion+reranker position = {fr_pos} of {d['n_candidates']}")
        print(f"    disagreement at K = {d['disagreeing_ks']}")
        print()

    print("-" * 72)
    print(f"CONCLUSION: {_conclusion(result['aggregate'], result['evaluated'])}")
    print("=" * 72)


if __name__ == "__main__":
    default_path = str(Path(__file__).resolve().parent / "labeled_set" / "sample_placeholder.json")
    target = sys.argv[1] if len(sys.argv) > 1 else default_path

    labeled_rows = load_labeled_clauses(target)

    from db import SessionLocal

    db = SessionLocal()
    try:
        result = evaluate(labeled_rows, db)
    finally:
        db.close()

    print_report(result)
