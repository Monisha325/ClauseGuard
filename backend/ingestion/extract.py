"""Text extraction for ClauseGuard contracts.

Converts a stored PDF or DOCX file into a list of {"page_number": int,
"text": str, "ocr_derived": bool} entries, in document order. Pure
function, no DB writes, no API surface.

M31 ADDITION: a PDF page whose directly-extracted text is under
MIN_TEXT_CHARS_PER_PAGE now falls back to OCR (ingestion/ocr.py's
ocr_page()) -- M5's original design correctly left this as "a later
milestone's job"; this is that milestone. ocr_derived=True marks any
page whose FINAL text came from OCR rather than direct extraction, so
downstream consumers can tell the difference (OCR'd text carries a real
recognition-error risk that direct extraction doesn't). Every page dict
now always has this key (including DOCX's single-entry return, which is
always False -- DOCX has no scanned-image concept and no OCR code path
is ever reached for it), so downstream consumers can rely on the key's
presence regardless of source format.

DETECTION THRESHOLD -- KNOWN, ACCEPTED FALSE-POSITIVE COST (documented,
not silently ignored): MIN_TEXT_CHARS_PER_PAGE=50 (this milestone's own
spec) is a per-page character-count heuristic, not a real "does this
page contain a scanned image" detector -- a real page with genuinely
little text (e.g. a one-line section-divider page, ~20-30 chars) will
also trigger OCR, unnecessarily. Accepted deliberately rather than
building a smarter detector (e.g. checking embedded-image coverage),
per this milestone's own explicit scope ("no OCR quality scoring beyond
the basic char-count heuristic"): the cost of a false positive here is
bounded and low-severity, not a correctness risk -- OCR re-rasterizes
and re-reads the SAME real content actually rendered on the page, so it
will very likely recognize substantively the same short text back (at
worst replacing a handful of already-correct characters with a
very-likely-identical OCR'd version, marked ocr_derived=True even though
the content didn't meaningfully change) -- wasted CPU time on that one
page, not wrong or lost content.
"""

import ctypes
import logging
import os
import signal
from pathlib import Path

import billiard
import docx
import fitz  # PyMuPDF

from ingestion.ocr import ocr_page

logger = logging.getLogger("clauseguard.ingestion.extract")

# See module docstring's DETECTION THRESHOLD note for the false-positive
# tradeoff this constant accepts. Measured against the page's text AFTER
# stripping whitespace, not the raw length, so a page of pure blank
# lines doesn't dodge OCR by accident.
MIN_TEXT_CHARS_PER_PAGE = 50


class UnsupportedFileTypeError(Exception):
    """Raised when extract_text() is given a file it doesn't know how to handle."""


def extract_text(file_path: str) -> list[dict]:
    """Extract text from a PDF or DOCX file.

    Returns a list of {"page_number": int, "text": str, "ocr_derived":
    bool} dicts, one per real PDF page, or a single entry for DOCX (see
    _extract_docx for why). The return shape is identical regardless of
    input format.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {file_path}")

    file_type = _detect_file_type(path)
    if file_type == "pdf":
        return _extract_pdf(path)
    return _extract_docx(path)


def _detect_file_type(path: Path) -> str:
    """Sniff the actual file content rather than trusting the extension
    alone — an extension can be wrong or spoofed, but a PDF always starts
    with the "%PDF-" magic bytes and a DOCX is always a zip archive.
    """
    with open(path, "rb") as f:
        header = f.read(8)

    if header.startswith(b"%PDF-"):
        return "pdf"

    if header.startswith(b"PK\x03\x04"):
        # DOCX (like other Office Open XML formats) is a zip archive, so
        # the magic bytes alone don't distinguish it from .xlsx/.pptx/.zip
        # — the extension still matters here.
        if path.suffix.lower() == ".docx":
            return "docx"
        raise UnsupportedFileTypeError(
            f"'{path.name}' is a zip-based file but not a .docx "
            f"(got extension '{path.suffix or '(none)'}')."
        )

    raise UnsupportedFileTypeError(
        f"Unsupported file type for '{path.name}': content does not match "
        "a known PDF or DOCX signature. Only .pdf and .docx are supported."
    )


def _clean_text(text: str) -> str:
    """Replace characters that can't round-trip through UTF-8 instead of
    letting a decode error kill the whole extraction, or silently passing
    through mangled bytes with no indication anything was wrong.
    """
    cleaned = text.encode("utf-8", errors="replace").decode("utf-8")
    if cleaned != text:
        logger.warning("Replaced non-UTF-8-safe characters during extraction.")
    return cleaned


def _extract_pdf(path: Path) -> list[dict]:
    pages = []
    doc = fitz.open(path)
    try:
        for page_index in range(len(doc)):
            page_number = page_index + 1  # PyMuPDF is 0-indexed; documents are 1-indexed
            try:
                text = doc[page_index].get_text()
            except Exception:
                # A single bad page (corrupt content stream, unsupported
                # font, etc.) shouldn't kill extraction for every other
                # page in the document — log it and move on with empty text.
                # Note this also makes the page eligible for the OCR
                # fallback below (empty text is well under the
                # threshold) -- a reasonable bonus: if the text LAYER
                # extraction failed but the page can still be
                # rasterized, OCR may recover something real.
                logger.warning(
                    "Failed to extract text from page %d of %s; returning empty text for this page.",
                    page_number,
                    path.name,
                    exc_info=True,
                )
                text = ""

            ocr_derived = False
            real_char_count = len(text.strip())
            if real_char_count < MIN_TEXT_CHARS_PER_PAGE:
                # An image-only/scanned page with no/minimal text layer --
                # or, per the module docstring's known tradeoff, a
                # genuinely short-real-text page. Try OCR; ocr_page()
                # itself decides if ITS result is usable and returns None
                # if not, in which case the original (already
                # under-threshold) text is kept as-is, unchanged from
                # pre-M31 behavior.
                ocr_text = ocr_page(doc[page_index])
                if ocr_text is not None:
                    logger.info(
                        "Page %d of %s had %d real char(s) (< %d threshold) -- "
                        "OCR produced %d char(s), using OCR text for this page.",
                        page_number, path.name, real_char_count,
                        MIN_TEXT_CHARS_PER_PAGE, len(ocr_text),
                    )
                    text = ocr_text
                    ocr_derived = True

            pages.append({
                "page_number": page_number,
                "text": _clean_text(text),
                "ocr_derived": ocr_derived,
            })
    finally:
        doc.close()
    return pages


def _extract_docx(path: Path) -> list[dict]:
    """DOCX has no native page concept: pagination is decided at render/
    print time by the viewer (depending on page size, margins, fonts), and
    is not stored anywhere in the .docx XML. Rather than fabricate fake
    page boundaries (e.g. splitting by character count), we treat the
    whole document as a single logical unit: page_number=1 with all
    paragraph text concatenated in document order. This is the simpler,
    honest option — later milestones (chunking) should not assume DOCX
    page numbers mean anything beyond "the whole document".

    ocr_derived is always False here -- DOCX has no scanned-image concept
    in the same way a PDF does (no image-only "page" to rasterize), and
    no OCR code path is ever reached for this format. The key is still
    present (rather than omitted) so its presence doesn't depend on
    which file format produced a given page dict.
    """
    document = docx.Document(str(path))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    return [{"page_number": 1, "text": _clean_text(text), "ocr_derived": False}]


# --- M32: process-level extraction timeout ---------------------------------
#
# Protects against resource exhaustion during PARSING (a "PDF bomb",
# pathologically deep/nested structure, or similar malformed-but-passes-
# upload-validation file) hanging or exhausting memory inside a Celery
# worker process. extract_text() itself above is completely UNCHANGED --
# this is a new, additive wrapper around it, not a restructuring of its
# internal parsing logic.
#
# WHY A SEPARATE PROCESS, NOT A THREAD OR SIGNAL-BASED TIMEOUT (researched
# before choosing, not assumed): extract_text() calls into PyMuPDF (a C
# library) and, for M31's OCR fallback pages, pytesseract (which itself
# shells out to the real `tesseract` binary). A Python-level thread
# timeout (e.g. threading.Timer) cannot interrupt a thread that's blocked
# inside a C call holding the GIL -- the timer fires on schedule, but the
# blocked thread keeps running the C call to completion regardless,
# however long that takes. signal.alarm() has the identical fundamental
# problem: a Python signal handler only ever runs BETWEEN Python bytecode
# instructions, which never happens while control is stuck inside a
# single, long-running native call. Neither mechanism can forcibly
# reclaim control from code that isn't cooperating. A separate OS process
# has no such limitation: process.kill() (SIGKILL) terminates it
# unconditionally from OUTSIDE, regardless of what it's doing internally
# or whether it would ever return control on its own.
EXTRACTION_TIMEOUT_SECONDS = 120  # see extract_text_with_timeout()'s own docstring for the reasoning behind this number

# GRANDCHILD-ZOMBIE FIX (confirmed reproducible defect, found by this
# milestone's own independent review -- see extract_text_with_timeout()'s
# own docstring for the full incident and why a plain os.killpg() alone
# does NOT fix it): SIGKILL is a no-op against a process that has
# ALREADY exited and is sitting as a zombie waiting to be reaped -- you
# cannot "kill" something already dead. A tesseract call that finishes
# a fraction of a second before the extraction subprocess is killed
# becomes exactly such a zombie, still parented to the (now-dead)
# extraction subprocess. Only that subprocess's OWN parent -- this
# worker process -- can reap it, but by default an orphan reparents
# straight to the container's PID 1 (which has no idea it exists and
# never calls waitpid() for it), not to us, even though we're the
# nearest living ancestor. PR_SET_CHILD_SUBREAPER (Linux-specific, via
# prctl(2); no stdlib wrapper exists, hence ctypes) marks THIS process
# as the reparenting target for any of its descendants' orphans instead
# of skipping straight to PID 1 -- verified directly against a minimal
# reproduction of this exact fork/orphan/zombie sequence before relying
# on it here. Set once, idempotently, for this worker process's entire
# lifetime (a prctl flag, not a per-call state) -- calling it again on
# a process that's already a subreaper is a harmless no-op.
_PR_SET_CHILD_SUBREAPER = 36
_child_subreaper_enabled = False


def _ensure_child_subreaper() -> None:
    global _child_subreaper_enabled
    if _child_subreaper_enabled:
        return
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        logger.warning(
            "prctl(PR_SET_CHILD_SUBREAPER) failed (errno %d: %s) -- a "
            "grandchild process (e.g. tesseract) orphaned by a future "
            "extraction timeout may be left as an unreapable zombie "
            "until this worker process restarts. Non-fatal: the "
            "extraction timeout itself still works correctly either way.",
            errno, os.strerror(errno),
        )
        return
    _child_subreaper_enabled = True


def _reap_orphaned_descendants() -> None:
    """Non-blocking sweep for any of this process's descendants that
    were just reparented to us (as the child subreaper, see
    _ensure_child_subreaper above) after the extraction subprocess we
    killed died without reaping them itself -- e.g. a tesseract call
    that finished a moment before the kill. Bounded: os.WNOHANG makes
    every waitpid() call return immediately rather than block, and the
    loop stops the instant there's nothing left to reap.
    """
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            break


class ExtractionTimeoutError(Exception):
    """Raised by extract_text_with_timeout() when extract_text() does not
    complete within EXTRACTION_TIMEOUT_SECONDS, or when the extraction
    subprocess exits without producing a result (crashed, OOM-killed by
    the OS, etc.) -- both are real, distinct failure modes from a normal
    UnsupportedFileTypeError or a per-page extraction warning; both leave
    the caller (pipeline/run_contract.py, ultimately worker/tasks.py's
    already-existing exception handling) with a clear, specific reason
    the pipeline failed, rather than a worker silently hung forever.
    """


def _extract_worker(file_path: str, result_queue) -> None:
    """Runs in a SEPARATE process (see extract_text_with_timeout below).
    Puts exactly one ("ok", pages) or ("error", repr(exc)) onto
    result_queue -- never raises across the process boundary itself,
    since arbitrary exception types don't reliably pickle/unpickle
    through a spawned process.

    os.setsid() FIRST, before anything else (confirmed reproducible gap,
    found by this milestone's own independent review): extract_text()'s
    OCR fallback (M31) shells out to the real `tesseract` binary as a
    grandchild of THIS process. Killing only this process (the direct
    child, as extract_text_with_timeout() used to do) does not touch
    that grandchild -- if tesseract is mid-flight or has just exited but
    not yet been reaped by this process at the moment of the kill, it is
    orphaned and, once it exits, becomes a zombie that nothing reaps by
    default (reparented straight to the container's PID 1, which has no
    reason to wait() on a process it doesn't know about) -- confirmed to
    persist until the whole container restarts. os.setsid() makes this
    process the leader of a brand-new session and process group (whose
    ID becomes this process's own PID) -- every subprocess it spawns
    afterward (tesseract included) inherits that SAME group, so the
    parent can kill the whole group at once (see
    extract_text_with_timeout below) rather than just this one process.
    That alone is NOT sufficient by itself, though (verified directly,
    not assumed) -- see extract_text_with_timeout's own docstring for
    why a SECOND, separate fix (child-subreaper + explicit reap) is also
    required for a grandchild that already exited before the kill.
    """
    os.setsid()
    try:
        pages = extract_text(file_path)
        result_queue.put(("ok", pages))
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any real failure must reach the parent, not vanish silently in the subprocess
        result_queue.put(("error", repr(exc)))


def extract_text_with_timeout(
    file_path: str, timeout_seconds: float = EXTRACTION_TIMEOUT_SECONDS
) -> list[dict]:
    """Runs extract_text() in a separate OS process with a real, hard
    timeout -- see the module section header above for why a process
    (not a thread or signal) is the only mechanism that can actually
    reclaim control from a stuck C-library call.

    TIMEOUT VALUE (a judgment call, documented not guessed): M31's OCR
    fallback measured ~3-5s per scanned page at the 300 DPI this project
    uses. A large but entirely legitimate, fully-scanned real contract
    (e.g. 20-30 pages, every page needing OCR) could reasonably take
    60-150s. EXTRACTION_TIMEOUT_SECONDS=120 is chosen to comfortably cover
    that realistic legitimate case while still bounding a genuinely
    malicious or pathological file to a fixed, real recovery time rather
    than an unbounded hang.

    "spawn" (not "fork") is used deliberately: extract_text() is a pure
    function with no shared state worth preserving across the process
    boundary (no open DB session, no live API client) -- spawning a
    completely fresh interpreter avoids any risk of inheriting
    possibly-fork-unsafe global state from two separate C libraries
    (PyMuPDF, Tesseract), at the cost of a small, one-time fresh-
    interpreter startup (negligible next to a 120s budget).

    USES billiard.get_context("spawn"), NOT stdlib multiprocessing (a
    correction made after this milestone's own independent review):
    Celery's prefork pool (via billiard) marks its own worker processes
    as daemonic. Stdlib multiprocessing.Process.start() unconditionally
    asserts the CALLING process is not daemonic ("daemonic processes are
    not allowed to have children"), and separately, spawn's bootstrap
    data pickling hits Python 3.12's AuthenticationString hardening
    (stdlib's own __reduce__ unconditionally refuses to pickle it). An
    earlier version of this function worked around both by temporarily
    overwriting multiprocessing.current_process()._config directly (a
    private, undocumented attribute) and restoring it in a finally block.
    That workaround was verified correct under repeated and failure-
    injected conditions, but rested on undocumented CPython internals
    with no API stability guarantee. billiard -- already a hard Celery
    dependency present in every deployment, not a new dependency being
    added here -- solves both problems natively: its own
    Process.start() has no daemonic-parent check at all (confirmed by
    reading billiard/process.py directly), and its own
    AuthenticationString.__reduce__ only blocks pickling when
    get_spawning_popen() is None, i.e. it already permits pickling during
    a legitimate spawn. No private-internals workaround is needed at all
    with billiard, so none remains here.

    Raises ExtractionTimeoutError on a real timeout (the subprocess --
    and, per the process-group kill below, anything IT spawned, like a
    tesseract OCR call -- is forcibly killed either way; this function
    never leaves a runaway process OR a zombie grandchild behind) or on
    the subprocess exiting without a result. Re-raises extract_text()'s
    own real failures as a RuntimeError wrapping the original
    exception's repr() (the original exception TYPE does not survive
    the process boundary, only its string form).

    GRANDCHILD-ZOMBIE FIX, TWO PARTS (confirmed reproducible defect,
    found and root-caused by this milestone's own independent review --
    see _ensure_child_subreaper's own docstring for the full mechanism):
    a first attempt that ONLY added os.killpg() (killing the whole
    process group instead of just the direct child) was verified, via
    the SAME 1-second-granularity /proc monitoring that originally found
    this bug, to NOT be sufficient by itself -- the zombie persisted
    unchanged. Root cause: SIGKILL cannot affect a process that has
    ALREADY exited and is just waiting to be reaped (a tesseract call
    that finishes a moment before the kill is exactly such a case) --
    you cannot "kill" something already dead, you can only reap it, and
    only its own direct parent (the extraction subprocess, which we just
    killed) can legally do that. The real fix has two parts, both
    required: (1) os.killpg() below still matters, for any grandchild
    that's genuinely still ALIVE at the moment of the kill; (2)
    _ensure_child_subreaper() (called once, at the top of this
    function) plus _reap_orphaned_descendants() (called after the kill)
    together catch the OTHER case -- a grandchild that already exited
    and would otherwise be orphaned straight to the container's PID 1
    (which has no idea it exists and never reaps it) is instead
    reparented to THIS process, which then explicitly reaps it.
    """
    _ensure_child_subreaper()

    ctx = billiard.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_extract_worker, args=(file_path, result_queue))
    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        logger.warning(
            "extract_text() exceeded %.0fs for %s -- killing the extraction "
            "subprocess and its process group (a real, hard process-level "
            "timeout; see this module's own docstring for why a "
            "thread/signal-based timeout would not have been able to do "
            "this).",
            timeout_seconds, file_path,
        )
        # Kill the WHOLE process group (this process plus anything it
        # spawned, e.g. tesseract for OCR -- see _extract_worker's own
        # os.setsid() call), not just this single pid -- catches any
        # grandchild that's still ALIVE at this moment (an already-EXITED
        # one is handled separately below, by the child-subreaper reap
        # sweep, since SIGKILL is a no-op against something already
        # dead). os.getpgid() is looked up fresh here rather than
        # assumed to equal process.pid, in case setsid() is ever slower
        # to run than expected -- ProcessLookupError means the process
        # already exited on its own between the is_alive() check above
        # and here, in which case there is nothing left to signal.
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        process.join(timeout=5)
        if process.is_alive():
            # SIGTERM alone didn't finish the job within the grace
            # period -- escalate to SIGKILL, still group-wide. Unlike
            # SIGTERM, SIGKILL cannot be ignored or caught, so this is a
            # guaranteed, unconditional kill regardless of what any
            # LIVE process in the group is doing.
            try:
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.join()
        # Catches the case os.killpg() above cannot: a grandchild (e.g.
        # tesseract) that had ALREADY exited on its own, moments before
        # the kill, and was orphaned when its own parent (the extraction
        # subprocess) died without reaping it -- see
        # _ensure_child_subreaper()'s docstring for the full mechanism.
        _reap_orphaned_descendants()
        result_queue.close()
        result_queue.join_thread()
        raise ExtractionTimeoutError(
            f"Extraction did not complete within {timeout_seconds:.0f}s for "
            f"{file_path!r} -- the file may be malformed, pathologically "
            "structured, or otherwise designed to exhaust parsing resources."
        )

    try:
        if result_queue.empty():
            # The process exited (crashed, OOM-killed by the OS, etc.)
            # without ever putting a result -- a real, distinct failure
            # mode from a timeout, but must not hang this caller waiting
            # on an empty queue either.
            raise ExtractionTimeoutError(
                f"Extraction subprocess for {file_path!r} exited unexpectedly "
                f"(exit code {process.exitcode}) without producing a result."
            )
        status, payload = result_queue.get()
    finally:
        result_queue.close()
        result_queue.join_thread()

    if status == "error":
        raise RuntimeError(f"extract_text() failed in subprocess: {payload}")
    return payload


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path-to-pdf-or-docx>", file=sys.stderr)
        sys.exit(1)

    for entry in extract_text(sys.argv[1]):
        print(f"--- page {entry['page_number']} ---")
        print(entry["text"])
        print()
