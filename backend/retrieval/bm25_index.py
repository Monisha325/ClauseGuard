"""M24: per-contract BM25 keyword search, to be fused (see
retrieval/fusion.py) with M11's vector search before M21's reranking
stage. Uses rank-bm25==0.2.2's BM25Okapi (the standard Okapi BM25
variant, pinned exactly -- see requirements.txt).

SCOPING -- THE HIGHEST-RISK BUG THIS FILE EXISTS TO AVOID (read before
changing anything here): a BM25 index that isn't correctly scoped per
contract could surface one contract's clauses when searching a
different contract -- a real authorization-adjacent bug, given every
other part of this system (Chroma's `where={"contract_id": ...}` filter,
Postgres's `Contract.user_id` ownership check, eval/retrieval_eval.py's
own per-contract clause fetch) is built around strict per-contract/
per-user isolation.

THE ACTUAL SCOPING MECHANISM: search_bm25() below takes NO contract_id
and touches NO database or global state at all -- it only ever operates
on the exact `candidates` list its caller passes in, and builds a brand
new BM25Okapi index from scratch on every single call (see that
function's own docstring for why this is a deliberate safety property,
not a missed caching opportunity). There is no module-level index, no
cache keyed by contract_id, no shared corpus of any kind that two
different calls could ever collide on -- correctness of scoping reduces
entirely to "did the caller pass in the right candidates", the same
scoping contract retrieval/reranker.py's rerank() and
eval/retrieval_eval.py's _vector_rank() already both rely on. See this
milestone's own testing step 6 for a concrete, real cross-contract
leakage test proving this holds, not just asserting it "should".

fetch_contract_clauses() below is the one function in this file that DOES
take a contract_id -- it is a thin, real Postgres query
(`Clause.contract_id == contract_id`), copied in spirit from
eval/retrieval_eval.py's own already-reviewed _fetch_contract_clauses()
(M19), not a fresh reimplementation of that filter. It exists purely as
a convenience for callers that want "give me this contract's real
candidates" in one call; nothing downstream of it (search_bm25 itself)
ever sees or uses contract_id.
"""

import re
import uuid

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, no stemming/stopword removal -- BM25's own
    IDF term already down-weights common words that appear in most
    documents; a real, distinctive term appearing in exactly one
    candidate gets the large IDF boost that makes BM25 useful here in
    the first place. Same tokenization convention as
    eval/retrieval_eval.py's own keyword baseline, for consistency.
    """
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def fetch_contract_clauses(contract_id: uuid.UUID, db_session) -> list[tuple[str, str]]:
    """Every real persisted Clause row for THIS contract_id, and only
    this contract_id -- returns [(clause_id_str, text), ...]. This is
    the actual per-contract scoping boundary: callers that want a
    correctly-scoped BM25 search should fetch candidates through this
    function (or an equivalent contract_id-filtered query, e.g.
    retrieval.py's existing Chroma `where` filter) and pass the result to
    search_bm25() below -- never a global, unfiltered clause fetch.
    """
    from models.clause import Clause

    rows = db_session.query(Clause).filter(Clause.contract_id == contract_id).all()
    return [(str(c.id), c.text) for c in rows]


def search_bm25(
    query: str,
    candidates: list[tuple[str, str]],
    n_results: int | None = None,
) -> list[tuple[str, float]]:
    """Rank `candidates` against `query` via BM25Okapi. Returns
    [(clause_id, bm25_score), ...] sorted descending by score -- the same
    (clause_id, score) shape convention retrieval/reranker.py's rerank()
    and eval/retrieval_eval.py's _vector_rank() already use, so BM25
    results can be fused with vector results (retrieval/fusion.py) or
    handed to the reranker interchangeably.

    `candidates`: list of (clause_id, text) pairs. SCOPING CONTRACT: the
    caller is responsible for this list already being scoped to exactly
    one contract (e.g. via fetch_contract_clauses() above) -- this
    function has no way to check that on its own, by design (see module
    docstring's SCOPING section: no contract_id parameter here at all is
    itself the safeguard, not an oversight).

    NO SHARED STATE: builds a fresh BM25Okapi index from `candidates`
    every call. For this system's real candidate-pool sizes (a handful
    to a few dozen clauses per contract), this costs nothing meaningful
    -- BM25Okapi's construction is pure local tokenization + term-count
    bookkeeping, not a live API call (contrast with M23's
    centroid_fallback.py, where caching centroids was necessary
    specifically because THAT computation required real, paced,
    rate-limited Voyage API calls; no equivalent cost exists here).
    Avoiding a cache also means avoiding the one failure mode a cache
    would introduce: two different contracts' candidates ever being
    served from the same cached index by mistake.
    """
    if not candidates:
        return []

    ids = [clause_id for clause_id, _text in candidates]
    tokenized_corpus = [_tokenize(text) for _clause_id, text in candidates]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(ids, scores), key=lambda pair: pair[1], reverse=True)
    if n_results is not None:
        ranked = ranked[:n_results]
    return [(clause_id, float(score)) for clause_id, score in ranked]
