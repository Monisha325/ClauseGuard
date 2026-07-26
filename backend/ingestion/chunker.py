"""Clause-boundary chunking for ClauseGuard contracts.

Takes M5's extract_text() output (list of {"page_number": int, "text": str})
and returns a list of {"heading_path": str, "text": str, "page_number": int}
chunks, sized against a real tokenizer's token count (not char/word count),
split along detected clause headings and, when a clause is oversized,
sentence boundaries.
"""

import logging
import re
import uuid

import tiktoken

from classification.heading_match import classify_clause
from models.clause import Clause

logger = logging.getLogger("clauseguard.ingestion.chunker")

MAX_TOKENS = 512
MIN_TOKENS = 40
UNTITLED = "UNTITLED"

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


# --- heading detection ---------------------------------------------------
#
# Known, accepted gaps (not chased for perfect coverage, per spec):
#   - "(i)" is always treated as a level-3 roman-numeral sub-item without
#     validating it's part of a real i/ii/iii/... sequence.
#   - The ALL-CAPS heuristic can false-positive on short all-caps emphasis
#     text (e.g. legally-required "conspicuous" warranty disclaimers) and
#     can miss headings in title case or other formatting not covered here.
#   - A numbered line like "3. This happens to start a sentence." can
#     false-positive as a heading if a clause contains an ordinary
#     numbered list that isn't actually a new clause boundary.
#   - No support for headings spanning multiple physical lines.

_NUMBERED_L3 = re.compile(r"^\d+\.\d+\.\d+\b")
_NUMBERED_L2 = re.compile(r"^\d+\.\d+\b")
_NUMBERED_L1 = re.compile(r"^\d+\.\s")
_SECTION = re.compile(r"^Section\s+(\d+(?:\.\d+)*)\b", re.IGNORECASE)
_ARTICLE = re.compile(r"^Article\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
_PAREN_ROMAN = re.compile(r"^\((i|ii|iii|iv|v|vi|vii|viii|ix|x)\)", re.IGNORECASE)
_PAREN_ALPHA = re.compile(r"^\([a-zA-Z]\)")
_PAREN_NUM = re.compile(r"^\(\d+\)")
_ALLCAPS = re.compile(r"^[A-Z][A-Z0-9 ,&/\-]{2,59}$")

# A bare "(N)" at the start of a line is ambiguous between a real numbered
# clause marker (e.g. "(60) Termination.") and a mid-sentence quantity
# reference that happened to wrap onto a new line right at the
# parenthetical (e.g. "...more than sixty\n(60) days past due...",
# "...clause\n(12) hereof...").
#
# Structural signal, not word-based: a real clause marker starts a NEW
# clause, which means the text immediately before it should end a sentence
# (terminal punctuation), be the very start of the document, or
# immediately follow another detected heading. A mid-sentence continuation,
# by definition, follows text with NO terminal punctuation — regardless of
# which word happens to follow the "(N)". This targets the actual
# structural cause of the ambiguity instead of enumerating specific
# following words.
#
# An earlier version of this check used a denylist of common unit words
# (day/days, month/months, etc.) checked against the word immediately
# following "(N)" — first as the sole check, then as a "secondary
# tiebreaker" applied even when the preceding-punctuation signal was
# permissive. Both were tried and both failed in practice: the denylist
# alone didn't generalize (false positives on "hereof"/"thereof"/"above"/
# "aforementioned" and adjective-interposed cases like "(10) working
# days"), and applying it even as a secondary check reintroduced the exact
# false-negative it was meant to avoid — a genuine heading like "(5) Days
# for Cure." immediately after a normally-punctuated sentence was still
# incorrectly rejected, because "days" matched the denylist regardless of
# how reliable the preceding punctuation was. There is no cheap, reliable
# way to distinguish a genuinely-ending sentence from one that merely
# looks like it ends (e.g. a preceding abbreviation like "No." or "Corp."),
# so rather than ship a check that silently reintroduces a known bug, the
# denylist has been removed entirely. The preceding-punctuation check is
# now the ONLY signal for this pattern.
#
# Known, accepted residual limitations (not chased for perfect coverage):
#   - A genuine new clause preceded by text ending in an abbreviation
#     (e.g. "...as provided by ABC Corp.\n(5) Termination.") will be
#     treated as permissive (the abbreviation's period looks like a
#     sentence end), which is the correct/safe direction to err in here —
#     but the converse also holds: a mid-sentence continuation immediately
#     after such an abbreviation could be misclassified as a heading. Not
#     defended against.
#   - A genuine new clause preceded by text ending without punctuation
#     (e.g. a drafting/OCR error) could be misclassified as a
#     continuation and suppressed.
#   - No support for headings spanning multiple physical lines (the
#     preceding-line check only looks at the single immediately-prior
#     line, not further back).

# Matches text ending in '.', '!', or '?', optionally followed by a closing
# quote/paren/bracket — i.e. "looks like the end of a sentence".
_TERMINAL_PUNCTUATION = re.compile(r"[.!?]['\")\]]*$")


def _ends_with_terminal_punctuation(text: str) -> bool:
    return bool(_TERMINAL_PUNCTUATION.search(text.strip()))


def _detect_heading(line: str, preceding_ends_sentence: bool = True) -> tuple[int, str] | None:
    """Return (level, heading_text) if line looks like a clause heading, else None.

    preceding_ends_sentence: whether the text immediately before this line
    ends a sentence (or this is the start of the document / immediately
    follows another heading). Only consulted for the "(N)" pattern — see
    the comment above _PAREN_NUM for why. Defaults to True so a bare/
    standalone call (e.g. without surrounding document context) behaves
    permissively rather than unexpectedly suppressing a heading.
    """
    stripped = line.strip()
    if not stripped:
        return None

    if _NUMBERED_L3.match(stripped):
        return (3, stripped)
    if _NUMBERED_L2.match(stripped):
        return (2, stripped)
    section_match = _SECTION.match(stripped)
    if section_match:
        level = 1 + section_match.group(1).count(".")
        return (level, stripped)
    if _ARTICLE.match(stripped):
        return (1, stripped)
    if _NUMBERED_L1.match(stripped):
        return (1, stripped)
    if _PAREN_ROMAN.match(stripped):
        return (3, stripped)
    if _PAREN_ALPHA.match(stripped):
        return (2, stripped)
    if _PAREN_NUM.match(stripped):
        if not preceding_ends_sentence:
            # The preceding text doesn't look like it ends a sentence, so
            # this "(N)" is almost certainly a mid-sentence continuation —
            # reject regardless of which word follows it.
            return None
        return (2, stripped)
    if len(stripped.split()) <= 8 and _ALLCAPS.match(stripped):
        return (1, stripped)

    return None


# --- sentence splitting ---------------------------------------------------
#
# Simple heuristic, not a full NLP sentence tokenizer, per spec's "simple
# heuristic" allowance: splits after '.'/'!'/'?' followed by whitespace and
# a capital letter or '('. Known gap: will mis-split on abbreviations like
# "U.S.", "e.g.", "Mr. Smith" — accepted rather than chased for full
# accuracy.

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


# --- candidate chunk construction -----------------------------------------


def _build_candidate_chunks(pages: list[dict]) -> list[dict]:
    """Walk pages/lines, detect headings, and split into raw candidate
    chunks at heading boundaries.

    Each candidate is {"heading_path": str, "lines": [(page_number, line), ...]}.
    heading_path is the " > "-joined breadcrumb of currently-nested headings
    (shallower headings pop deeper ones off the stack), or UNTITLED if no
    heading has been seen yet.
    """
    candidates: list[dict] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[tuple[int, str]] = []

    # Tracks the preceding-context signal for the "(N)" ambiguity check in
    # _detect_heading: the previous non-blank line's text (for the
    # terminal-punctuation check) and whether that previous line was
    # itself a detected heading (also treated as a permissive context,
    # since a clause marker immediately following another heading is a
    # normal nested-numbering pattern). Both start as "permissive" so the
    # very first line of the document is never incorrectly rejected as a
    # continuation of text that doesn't exist.
    prev_line_text: str | None = None
    prev_was_heading = False

    def flush():
        if current_lines:
            path = " > ".join(h for _, h in heading_stack) if heading_stack else UNTITLED
            candidates.append({"heading_path": path, "lines": list(current_lines)})
            current_lines.clear()

    for page in pages:
        page_number = page["page_number"]
        for line in page["text"].splitlines():
            if not line.strip():
                continue
            stripped_line = line.strip()

            preceding_ends_sentence = (
                prev_line_text is None
                or prev_was_heading
                or _ends_with_terminal_punctuation(prev_line_text)
            )

            heading = _detect_heading(stripped_line, preceding_ends_sentence)
            if heading is not None:
                # Flush whatever was accumulated under the OLD heading
                # context before changing the stack for the new heading.
                flush()
                level, text = heading
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, text))
                current_lines.append((page_number, stripped_line))
                prev_was_heading = True
            else:
                current_lines.append((page_number, stripped_line))
                prev_was_heading = False

            prev_line_text = stripped_line

    flush()
    return candidates


# --- oversized-clause splitting -------------------------------------------


def _finalize_candidate(candidate: dict) -> list[dict]:
    """Turn a raw candidate chunk into one or more sized output chunks.

    If the whole candidate is within MAX_TOKENS, it's returned as a single
    chunk. Otherwise it's split at sentence boundaries (never mid-sentence)
    into multiple chunks, each still under MAX_TOKENS where possible.

    page_number convention: a chunk's page_number is the page its first
    content (first line, or for a sentence-split sub-chunk, its first
    sentence) actually came from — not necessarily every page it touches.
    """
    lines = candidate["lines"]
    heading_path = candidate["heading_path"]
    full_text = "\n".join(text for _, text in lines)

    if _count_tokens(full_text) <= MAX_TOKENS:
        return [{
            "heading_path": heading_path,
            "text": full_text,
            "page_number": lines[0][0],
        }]

    # Reconstruct per-page paragraphs (joining consecutive lines from the
    # same page) before sentence-splitting, so a paragraph wrapped across
    # multiple physical lines isn't fragmented at line breaks that aren't
    # real sentence ends.
    spans: list[tuple[int, str]] = []
    for page_number, text in lines:
        if spans and spans[-1][0] == page_number:
            spans[-1] = (page_number, spans[-1][1] + " " + text)
        else:
            spans.append((page_number, text))

    tagged_sentences: list[tuple[int, str]] = []
    for page_number, span_text in spans:
        for sentence in _split_sentences(span_text):
            tagged_sentences.append((page_number, sentence))

    sub_chunks: list[dict] = []
    current: list[str] = []
    current_tokens = 0
    current_page: int | None = None

    for page_number, sentence in tagged_sentences:
        sentence_tokens = _count_tokens(sentence)
        if current and current_tokens + sentence_tokens > MAX_TOKENS:
            sub_chunks.append({
                "heading_path": heading_path,
                "text": " ".join(current),
                "page_number": current_page,
            })
            current = []
            current_tokens = 0
            current_page = None
        if current_page is None:
            current_page = page_number
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        sub_chunks.append({
            "heading_path": heading_path,
            "text": " ".join(current),
            "page_number": current_page,
        })

    # A single sentence longer than MAX_TOKENS on its own is kept whole
    # rather than split mid-sentence — an accepted, documented tradeoff
    # since "never split mid-sentence" takes priority over the size bound
    # for this pathological edge case.
    return sub_chunks


# --- undersized-chunk merging ----------------------------------------------


def _merge_undersized(chunks: list[dict], min_tokens: int) -> list[dict]:
    """Merge chunks under min_tokens into an adjacent chunk rather than
    leaving tiny fragments.

    Convention: merge BACKWARD into the previous chunk by default (the
    previous chunk's heading_path and page_number are kept, since it's
    the earlier/governing context). If backward isn't available (no
    previous chunk) or would push the result over MAX_TOKENS, try
    FORWARD into the next chunk instead — the merged result keeps the
    next chunk's heading_path (the real governing heading) but the
    earlier fragment's page_number (since its content starts first).

    HARD CONSTRAINT, checked before every merge: MAX_TOKENS always wins
    over the MIN_TOKENS preference. If merging in EITHER direction would
    push the result over MAX_TOKENS, that merge is skipped. If neither
    direction has room, the undersized fragment is left as its own
    (under-min) chunk rather than ever produce a chunk over 512 tokens —
    downstream embedding/retrieval depends on that ceiling never being
    violated, whereas falling short of the 40-token floor is just a
    quality preference against leaving lots of tiny, low-context
    fragments, not a hard requirement. (This was a real, confirmed bug:
    a 45-item oversized clause produced three in-bounds sub-chunks
    (511, 506, 23 tokens) from _finalize_candidate, and the old version
    of this function blindly merged the trailing 23-token fragment
    backward into the 506-token neighbor, producing a 529-token chunk —
    over the limit.)

    This check applies uniformly to every chunk passed in regardless of
    origin — a genuinely short real clause and a fragment produced by
    _finalize_candidate's oversized-clause sentence-splitting (e.g. one
    half of a sentence that got cut by a PDF page boundary during that
    function's per-page span reconstruction) are treated identically
    here; this function has no way to distinguish them and doesn't need
    to. Note for completeness: a NORMAL (non-oversized) clause that
    happens to span a page boundary is NOT at risk of that specific
    "sentence cut by a page boundary" fragmentation — _finalize_candidate
    returns a non-oversized candidate as a single whole chunk regardless
    of how many pages its lines touch; the per-page span-then-sentence-
    split reconstruction that can produce a cut-sentence fragment only
    runs in the oversized branch. But since the MAX_TOKENS check here is
    unconditional for every chunk, that distinction doesn't matter for
    correctness — any undersized chunk from any source is protected.

    A single-chunk document is left as-is even if under min_tokens —
    there's nothing to merge it with.

    KNOWN LIMITATION (confirmed via testing, not just theoretical): when
    several short, real clauses appear consecutively — or when a false-
    positive heading match (see _detect_heading's known gaps) creates
    several spurious small candidates in a row — this function's
    forward/backward merging can cascade across MULTIPLE original
    headings before finally producing one output chunk large enough to
    stand on its own. The resulting chunk's heading_path reflects only
    whichever single candidate happened to be the one that pushed the
    running total over min_tokens (forward-merge case) or the earliest
    anchor chunk (backward-merge case) — not a summary of everything
    that got folded in. The chunk's TEXT is always complete and correct
    (nothing is lost or duplicated), but heading_path can end up stale
    or attached to a less-representative heading than a human would
    pick. Observed concretely in testing: a "DELIVERABLES" clause
    followed by a false-positive-detected numbered list collapsed into
    one chunk labeled "3. A final report upon completion of all
    services." instead of "DELIVERABLES". Acceptable for this
    milestone's scope (a pure chunking transform with no classification
    layer yet), but worth knowing before heading_path is relied on for
    retrieval/display in later milestones.
    """
    if len(chunks) <= 1:
        return chunks

    result = [dict(c) for c in chunks]

    i = 0
    while i < len(result):
        if _count_tokens(result[i]["text"]) >= min_tokens:
            i += 1
            continue

        text = result[i]["text"]
        did_merge = False

        # Primary: backward into the previous chunk, only if it stays
        # within MAX_TOKENS.
        if i > 0:
            candidate_text = result[i - 1]["text"] + "\n" + text
            if _count_tokens(candidate_text) <= MAX_TOKENS:
                result[i - 1]["text"] = candidate_text
                del result[i]
                did_merge = True
                # Don't advance i: the list shifted left by one, so the
                # element that follows is now at this same index.

        # Fallback: forward into the next chunk, only if backward wasn't
        # available/didn't fit and forward stays within MAX_TOKENS.
        if not did_merge and i < len(result) - 1:
            candidate_text = text + "\n" + result[i + 1]["text"]
            if _count_tokens(candidate_text) <= MAX_TOKENS:
                result[i + 1] = {
                    "heading_path": result[i + 1]["heading_path"],
                    "text": candidate_text,
                    "page_number": result[i]["page_number"],
                }
                del result[i]
                did_merge = True
                # Don't advance i: it now points at the just-merged chunk,
                # which may itself still be undersized and need another
                # pass (cascading merge, handled naturally by re-visiting).

        if not did_merge:
            # Neither neighbor has room without exceeding MAX_TOKENS (or
            # neither neighbor exists) — leave this fragment as its own
            # chunk rather than violate the hard 512 ceiling.
            i += 1

    return result


# --- public entry point -----------------------------------------------------


def chunk_document(pages: list[dict]) -> list[dict]:
    """Turn extract_text()'s page-level output into clause-sized chunks."""
    if not pages:
        return []

    candidates = _build_candidate_chunks(pages)
    if not candidates:
        return []

    sized: list[dict] = []
    for candidate in candidates:
        sized.extend(_finalize_candidate(candidate))

    return _merge_undersized(sized, MIN_TOKENS)


# --- persistence --------------------------------------------------------
#
# Kept separate from chunk_document() on purpose: chunk_document() stays a
# pure transformation (list[dict] in, list[dict] out) with no DB
# dependency, independently testable exactly as before. This function is
# the only piece of M7 that touches the database, and it takes
# chunk_document()'s OUTPUT as a plain argument rather than calling
# chunk_document() itself — the caller (a script, and eventually a Celery
# task in a later milestone) is responsible for running extract_text() and
# chunk_document() first, then handing the result here.


def persist_chunks(contract_id: uuid.UUID, chunks: list[dict], db_session) -> int:
    """Write chunk_document()'s output to the clauses table, linked to contract_id.

    Expects an already-open db_session (e.g. from SessionLocal()) rather
    than creating its own engine/session, so it composes with whatever
    calls it — consistent with how the rest of this codebase passes
    sessions around (see routes/auth.py, routes/contracts.py).

    Re-run safety: any existing clauses for this contract_id are deleted
    before the new set is inserted, and the delete + inserts are all part
    of the SAME transaction (a single commit() at the end). If anything
    fails before that commit — a bad row, a dropped connection, etc. —
    the whole transaction rolls back, so the contract is left with
    whatever clause rows it had BEFORE this call, never a partial mix of
    old and new. Running this twice in a row for the same contract_id
    replaces its clause rows rather than accumulating duplicates.

    Bulk insert: all Clause objects for this call are built up front and
    added in one batch, with a single commit — not one commit per row —
    since a real contract can have 50+ clauses.

    M8/M23 note: category is assigned here, at persist time, via
    classify_clause() (classification/heading_match.py) — M8's stage-1
    heading-match, falling back to M23's embedding-centroid stage 2 only
    when stage 1 returns None — called per-chunk rather than run as a
    separate pass over already-persisted rows. Classification is
    synchronous (stage 2 makes a real, live embedding API call only for
    the subset of chunks stage 1 couldn't classify), so it doesn't need
    its own DB round-trip/transaction; this keeps persist_chunks() the
    single place a Clause row is fully formed, the same way every other
    field here is set once, in one place, at insert time.
    """
    clause_rows = [
        Clause(
            contract_id=contract_id,
            heading_path=chunk["heading_path"],
            text=chunk["text"],
            page_number=chunk["page_number"],
            category=classify_clause(chunk["heading_path"], chunk["text"]),
        )
        for chunk in chunks
    ]

    try:
        db_session.query(Clause).filter(Clause.contract_id == contract_id).delete()
        db_session.add_all(clause_rows)
        db_session.commit()
    except Exception:
        db_session.rollback()
        raise

    return len(clause_rows)


# --- full-pipeline orchestration -----------------------------------------
#
# M10 adds embedding + Chroma indexing (embeddings/index.py) as a step that
# runs AFTER persist_chunks() commits the contract's clauses to Postgres —
# index_contract_clauses() reads Clause rows back from the DB rather than
# taking chunks directly, so it stays usable on its own (e.g. re-indexing
# without re-chunking) and persist_chunks() stays a pure Postgres step that
# never depends on Voyage/Chroma being reachable.


def process_contract(contract_id: uuid.UUID, file_path: str, db_session) -> int:
    """Run the full pipeline for one contract: extract -> chunk -> persist
    (Postgres) -> embed + index (Chroma). Returns the number of clauses
    indexed into Chroma.

    Convenience orchestrator for scripts/tasks that want the whole
    pipeline in one call; each step remains independently callable and
    independently testable on its own.
    """
    from embeddings.index import index_contract_clauses
    from ingestion.extract import extract_text

    pages = extract_text(file_path)
    chunks = chunk_document(pages)
    persist_chunks(contract_id, chunks, db_session)
    return index_contract_clauses(contract_id, db_session)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: python -m ingestion.chunker <path-to-pdf-or-docx>", file=sys.stderr)
        sys.exit(1)

    from ingestion.extract import extract_text

    pages = extract_text(sys.argv[1])
    chunks = chunk_document(pages)
    for i, c in enumerate(chunks):
        tokens = _count_tokens(c["text"])
        print(f"--- chunk {i + 1} | heading_path={c['heading_path']!r} | page={c['page_number']} | tokens={tokens} ---")
        print(c["text"])
        print()
