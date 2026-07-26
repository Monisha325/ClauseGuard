"""M24: Reciprocal Rank Fusion (RRF), combining M11's vector search
ranking and M24's BM25 ranking (retrieval/bm25_index.py) into a single
fused ranking, inserted BEFORE M21's reranking stage. Pipeline order:
vector + BM25 -> RRF fusion (this file) -> rerank -> M20 threshold gate.

RRF FORMULA -- VERIFIED AGAINST THE STANDARD DEFINITION BEFORE WRITING
THIS (Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion Outperforms
Condorcet and Individual Rank Learning Methods", SIGIR 2009 -- the
paper that introduced RRF and the source of the "k=60" constant this
milestone's spec asks for by name):

    RRFscore(d) = sum over each input ranking r of  1 / (k + rank_r(d))

where rank_r(d) is document d's position in ranking r. THE OFF-BY-ONE
THIS MILESTONE EXPLICITLY FLAGS: the paper's own convention is
1-INDEXED rank (the top result of a ranking is rank 1, not rank 0) --
reciprocal_rank_fusion() below uses `enumerate(ranking, start=1)`
specifically to match this, not `start=0`. Using 0-indexed rank would
still produce a valid-looking, monotonic fused ordering (it's a
subtle bug, not a crash) but every score would be systematically
1/(k+rank) too large compared to the standard formula -- exactly the
kind of silently-wrong-but-plausible-looking output this project's
established review discipline exists to catch (same category of risk
as M20's "gate on the wrong score" bug).

A document absent from one of the input rankings contributes NOTHING
from that ranking to its sum -- it is simply omitted from that inner
term, not penalized with some substitute large-rank value. This is the
standard definition's own treatment of a document that one method
didn't retrieve at all, not a design choice specific to this file.

RRF only ever looks at ORDER (rank) within each input list, never at the
input rankings' own score values -- this is RRF's whole point: it lets
you combine BM25's unbounded, corpus-dependent scores with vector
search's bounded 0-1 similarity scores WITHOUT needing to normalize
either onto a shared scale first, since rank position is already
directly comparable across differently-scaled rankings.

k=60 (RRF_K below) per this milestone's spec -- also the original paper's
own reported value and a common default in production hybrid-search
systems.
"""

RRF_K = 60


def reciprocal_rank_fusion(
    *rankings: list[tuple[str, float]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Fuse any number of ranked (candidate_id, score) lists via RRF.

    Each input ranking must already be sorted best-first -- e.g.
    retrieval/bm25_index.py's search_bm25() output, or a vector-search
    ranking built the same way retrieval/routes.py's existing Chroma
    query + `1/(1+distance)` sort already produces. This milestone only
    ever calls this with exactly 2 rankings (vector, BM25), but the
    function itself doesn't assume a fixed count -- RRF's formula is
    defined as a sum over however many input rankings exist.

    Returns [(candidate_id, rrf_score), ...] sorted descending by
    rrf_score. A candidate appearing in more than one input ranking
    accumulates a contribution from each (see module docstring); a
    candidate appearing in only one ranking still gets a score from that
    ranking alone.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (candidate_id, _original_score) in enumerate(ranking, start=1):
            scores[candidate_id] = scores.get(candidate_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
