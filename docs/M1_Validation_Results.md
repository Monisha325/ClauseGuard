# ClauseGuard — Phase 2 Validation Results (M30 addendum)

This document closes the loop back to the original v4 architecture doc's
three explicitly-stated "risk, not yet resolved" items. It is a
synthesis, not new evidence: every number and finding below was produced
by a specific milestone (M17-M29) and is cited to it. Nothing here was
re-measured or re-run to produce this document. Where a milestone's
answer was inconclusive or negative, that is stated plainly, not rounded
toward a more confident-sounding verdict.

**Method note on N.** Every retrieval/classification-quality eval in this
project (M19, M22, M25, M26, M27) has run against the same 5-row labeled
fixture (`eval/labeled_set/m19_fixture.json`, 1 contract). The eval
code's own `SMOKE_TEST_N_THRESHOLD = 10` constant flags any N below that
as not statistically meaningful — which is every quality number below.
This is stated once here and then attached to each individual finding, so
it is never silently forgotten mid-document.

**Starting point (M17).** Before Phase 2, the pipeline's mechanical
components (M8-M16) had been built but none of the v4 doc's three risk
items had any real measurement behind them at all — no Hit@K number, no
threshold sanity check, no latency figure under realistic input. M17 is
the diagnostic baseline that established this: it did not answer any of
the three questions, it confirmed they were genuinely open. Phase 2
(M17-M29) is the body of work that first put real numbers against them.

---

## 1. Reranker domain-fit (ms-marco-MiniLM on legal clause text)

**v4 doc's original concern:** ms-marco-MiniLM-L6-v2 was trained on
Bing web-search queries against web passages, not legal contract text.
Whether cross-encoder reranking actually helps ClauseGuard's retrieval
quality, versus plain vector search, was unknown.

**What was measured:**

- **M22** (reranker A/B, Hit@K, N=5, same fixture as M19): every one of
  the 5 rows already placed the true clause at position 1 for BOTH
  vector-only and reranked retrieval — a **ceiling effect**, not a tie
  that favors either method. M22's own independent review found this
  reflects an absence of discriminating power in the fixture, not
  equivalent performance. Verdict: **DEFER** (tied results, N below
  threshold, ceiling effect both present).
- **M22 also found**, outside the 5 scored rows, a real jargon/
  abbreviation-sensitivity gap: a query using "NDA" against a
  Confidentiality clause whose own text never contains that abbreviation
  caused vector-only search to rank the true clause **last, position 7
  of 7**. Reranking partially recovered it to **position 5 of 7** — still
  outside a top-3 cutoff, only just inside a top-5 one. This is a single
  constructed case, not one of the 5 scored fixture rows, and is reported
  as a directional finding, not a Hit@K statistic.
- **M25** (fusion + reranker re-run, Hit@K, N=5, same fixture): bare
  BM25+RRF fusion (no reranker) **regressed** Hit@1 relative to
  vector-only — **0.8 vs 1.0** — on a real keyword-trap row, where BM25's
  lexical overlap favored a superficially similar but wrong clause (the
  same failure class M19's own naive keyword baseline was built to
  expose). Verdict for fusion alone: **DROP** at this N. Adding the
  reranker on top of that same fused list recovered the miss, bringing
  fusion+reranker back to parity with vector-only. Verdict for
  fusion+reranker: **DEFER** (tied, not proven better, at this N).

**Honest conclusion — deliberately not a yes/no:** the evidence is
directionally positive for reranking (it recovered a real regression in
M25, and partially recovered a real jargon-miss in M22) but the base
rate this evidence rests on is thin (N=5 throughout) and most of the
fixture cannot discriminate between methods at all (ceiling effect,
M22/M25 both). **Domain-fit is not confirmed and not ruled out.** The
only way to move past DEFER is M18's still-not-done hand-labeling
expansion (15-20+ real contracts, explicitly including harder/jargon-
heavy queries — M22's own stated gap), not more runs of the current
fixture.

---

## 2. Confidence threshold calibration

**v4 doc's original concern:** per-category confidence thresholds gate
which reranked results are shown as "confident" vs "low confidence" —
whether the actual threshold VALUES are correctly calibrated was
unknown, separate from whether the gating mechanism itself is wired
correctly.

**Mechanism — verified correct (M20):** `backend/retrieval/thresholds.py`
gates on the reranker's post-sigmoid score (`reranker.rerank()`'s real
0-1 output), never on raw Chroma cosine/L2 similarity — confirmed by
reading the actual call site in `routes/retrieval.py` directly. This is
the "v3 bug" class of error (threshold applied to the wrong score) that
M20 exists to prevent, and it does not recur here.

**Threshold values — provenance, not independently derived:** the five
values (Limitation of Liability 0.62, Indemnification 0.71, Termination
0.65, Governing Law/Jurisdiction 0.55, Confidentiality 0.68) were given
directly by the project owner as M20's spec. **No copy of the real v4
architecture doc exists anywhere in this repository** (confirmed via a
repo-wide search at M20 build time) — these numbers have never been
independently checked against the source document itself.

**Calibration — genuinely unanswered (M26):** M26's threshold sanity
check ran every one of the 5 fixture rows' real post-sigmoid rerank
scores against its category's threshold. Result: **all 5 categories
flagged INSUFFICIENT DATA**, because the fixture provides only ~1 labeled
example per category (N=1 per category, N=5 total). This is not a
validation result — it is an honest non-answer. **No threshold in this
table has been calibration-checked against real labeled outcomes.**

**Related context, not the same measurement (M23):** M23's own
independent review of the centroid-fallback classifier (a different
scoring system — cosine similarity, not reranker sigmoid — used for
clause *categorization*, not confidence gating) found real stage-2 cases
sitting only ~0.085 apart on either side of its own 0.5 decision boundary.
This does not bear on M20's threshold values directly (different score,
different purpose), but it is a concrete illustration that this
project's real data does produce cases sitting close to a decision
boundary — relevant context for why "insufficient data" at M26 should
not be read as a formality.

**Honest conclusion:** the gating mechanism is correct. **The threshold
values themselves remain unvalidated** — this has not changed since M20
first set them, and will not change until M18's real labeled data exists
in enough volume per category to clear M26's own insufficient-data flag.

---

## 3. Latency budget (rerank ~180ms/450ms, end-to-end ~2.5s/5s)

**v4 doc's original targets:** rerank stage ~180ms p50 / ~450ms p95;
full contract pipeline ~2.5s p50 / ~5s p95.

**Rerank latency — fixed once, then re-verified twice at different
scales:**

- **M21** (initial benchmark, then fix): the reranker's first-reported
  ~244ms p95 held only for unrealistically short (~41-token) synthetic
  test text. Against realistic legal-clause-length text (220-297
  tokens), real p95 was **1506-1688ms — 3-4x over the ~450ms target**
  (N not recorded for either of these two pre-fix figures in
  `reranker.py`'s own docstring — a gap in the original source, noted
  here rather than silently filled in with an invented number).
  Root-caused to CPU self-attention cost scaling with input length,
  fixed via truncation (`MAX_SEQ_LENGTH=128`). Re-measured post-fix at
  **p95=403.2ms across 20 realistic-length candidates (N=20)** — under
  target.
- **M28** (real, persistent, always-on per-call timing instrumentation,
  added after M21's one-off benchmark): at the production-correct
  candidate pool size (20, matching `RERANK_CANDIDATES`), real measured
  latency was **p50=312.46ms, p95=345.85ms (N=20)**. p50 **exceeds** the
  ~180ms target — a real, confirmed miss, not a rounding-up situation.
  p95 stays within the ~450ms target.

**End-to-end pipeline latency — essentially unanswered at this N
(M28):** only **n=2** genuine complete pipeline runs exist with real
timing data. One clean run: **5502.447ms — 2.2x over the ~2.5s p50
target**, a substantial, not modest, overage on the one reliable data
point. The other run's total (72432.285ms) is dominated by a real ~65-
second Voyage rate-limit backoff wait, not representative steady-state
latency, and is excluded from any percentile claim. With n=2, "p95" is
mathematically indistinguishable from "the max" and is not reported as a
meaningful percentile here.

**Honest conclusion:** rerank p95 is within budget; rerank **p50 is not**
(312ms vs 180ms target, M28, N=20). End-to-end latency has one real data
point showing a 2.2x overage and is not answerable at n=2 in either
direction — this is not "probably fine," it is genuinely unmeasured at
adequate scale.

---

## 4. Resolved and remaining issues (separate from the three original risk items)

These were not part of the v4 doc's original three concerns — they are
real defects Phase 2's own testing surfaced along the way.

**503-vs-429 Gemini retry gap (M27, re-confirmed M28) — FIXED (M30
hardening pass).**
`backend/pipeline/run_contract.py`'s `_is_rate_limit_error()` only
matched `429` `ClientError` responses. Gemini's `5xx` errors raise
`ServerError` — confirmed directly against the SDK's own
`APIError.raise_error` dispatch (`ClientError` and `ServerError` are
sibling subclasses of `APIError`, not parent/child) — so a real Gemini
503 was never retried. First observed as a real, live 503 during M27's
own eval run. Recurred again, live, during M28's own testing. Fixed in
a dedicated post-Phase-2 hardening pass: `_retry_reason()` (renamed from
`_is_rate_limit_error()`) now also retries a `ServerError` with
`code == 503`, using the same backoff constants as the 429 case —
deliberately narrow (503 specifically, not "any ServerError"/"any 5xx"),
so a genuine, non-transient error is never silently retried. Verified
directly: a real `genai_errors.ServerError(503, ...)` instance now
triggers real retry attempts and real backoff waits before eventually
succeeding; 429 retries confirmed unchanged (no regression); three
separate genuinely non-retryable cases (a validation error, a 400
`ClientError`, and a 500 `ServerError`) were each confirmed to still NOT
retry, ruling out a "retry everything" regression.

**`persist_chunks()` foreign-key re-run-safety bug (M28, new discovery)
— FIXED (M30 hardening pass).**
`backend/ingestion/chunker.py`'s `persist_chunks()` deletes a contract's
existing `Clause` rows before inserting new ones, in one transaction.
`backend/pipeline/run_contract.py`'s own `FlaggedClause` cleanup used to
happen only at the very end of a full pipeline run — too late to
protect that earlier delete. Re-running the pipeline on a contract that
already had `FlaggedClause` rows from a prior run caused
`persist_chunks()`'s delete to fail with a real
`psycopg2.errors.ForeignKeyViolation` on `flagged_clauses_clause_id_fkey`
— reproduced live during M28's own testing, and reproduced fresh again
during M30 (via a real end-to-end `run_contract_pipeline()` call)
immediately before the fix was deployed. Fixed by moving the old
`FlaggedClause` delete to before `persist_chunks()` runs, on the same
`db_session`, uncommitted — so it rides inside `persist_chunks()`'s own
delete-then-insert-then-commit transaction instead of being a separate,
earlier commit. Verified directly: the same previously-crashing contract
now completes 3 consecutive real pipeline runs cleanly, with exactly 1
`Clause` row and 1 correctly-linked `FlaggedClause` row each time (no
orphans, no duplicates); a real, forced `NotNullViolation` inside
`persist_chunks()`'s own commit step confirmed its rollback still
restores BOTH tables to their exact prior state together, preserving the
original all-or-nothing guarantee rather than weakening it.

**Thin labeled dataset (all of Phase 2, M19-M27).** Every retrieval/
classification/generation quality eval since M19 has run against the
same 5-row, 1-contract fixture. This is the single blocking constraint
behind every "DEFER" and "INSUFFICIENT DATA" verdict in this document —
not a limitation of any individual milestone's methodology. M18's
planned expansion (15-20+ real, hand-labeled contracts) is the actual
prerequisite for turning Section 1's DEFER, Section 2's unvalidated
thresholds, or Section 3's n=2 end-to-end measurement into real answers.
It has not happened as of M29.

---

## 5. Summary

| Question | Status | Source |
|---|---|---|
| Does reranking help on legal text? | DEFER — directionally positive (recovered 1 real regression, M25; partially recovered 1 separate weak-retrieval case, M22), N=5, ceiling effect | M22, M25 |
| Are confidence thresholds calibrated? | NOT VALIDATED — mechanism correct, all 5 categories INSUFFICIENT DATA at N=1 each | M20, M26, M23 |
| Rerank latency in budget? | PARTIAL — p95 yes (345.85ms<450ms), p50 no (312.46ms>180ms), N=20 | M21, M28 |
| End-to-end latency in budget? | UNANSWERED — n=2, one clean point at 2.2x the p50 target | M28 |
| 503 retry gap | FIXED (M30) — verified: real 503 now retries, 429 unaffected, non-retryable errors still don't retry | M27, M28, M30 |
| `persist_chunks()` FK bug | FIXED (M30) — verified: 3 consecutive re-runs clean, no orphans, atomicity preserved | M28, M30 |
| Labeled dataset adequate for calibration? | NO — N=5 throughout, below the eval suite's own N=10 threshold | M19-M27 |

Phase 2 (M17-M29) put real, traceable numbers behind all three of the
v4 doc's original risk items for the first time. Two of the three remain
open questions (reranker domain-fit, threshold calibration); the third
(latency) is a confirmed partial miss, not a pass. Two additional real
defects were found during Phase 2 and have since been fixed and verified
in a dedicated M30 hardening pass (see Section 4) — not part of the
original three risk items, and not a change to any of Sections 1-3's
findings. None of this is a failure of Phase 2's own work — it is what
honest measurement of a real system at N=5 produces, and it is stated
here exactly as measured.
