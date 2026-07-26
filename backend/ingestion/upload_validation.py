"""M32: pre-pipeline upload validation -- size limits and TRUE, byte-
level content verification, checked BEFORE a file is written to its
final storage location or a Contract row is created. The upload
endpoint (routes/contracts.py) is the first milestone to treat this path
as a real, adversarial attack surface rather than assuming a good-faith
client; M4's original scope explicitly deferred all of this.

TWO SEPARATE SIZE CHECKS, DELIBERATELY, NOT ONE (see KNOWN FAILURE MODE
this milestone's own spec calls out): a client-supplied Content-Length
header is a cheap, useful FAST-PATH rejection (reject an obviously
oversized upload before reading a single byte of the body) but is
NEVER trusted as the actual enforcement mechanism -- it can be omitted,
wrong, or deliberately spoofed (a client can claim a small size while
sending a large body, or vice versa). The REAL limit is enforced by
counting actual bytes as they are read from the upload stream and
written to disk, in validate_and_stream_to_disk() below -- this is the
check a lying or missing header cannot bypass.

TRUE, BYTE-LEVEL MIME VERIFICATION, NOT TRUST (the other known risk this
milestone's spec explicitly flags): the client-supplied Content-Type
header and the filename's extension are NEVER trusted for the actual
PDF/DOCX determination. detect_real_file_type() below inspects the
file's own real magic bytes (the same two well-known, stable file-format
signatures ingestion/extract.py's own _detect_file_type() already
checks: PDF's "%PDF-" and ZIP's "PK\x03\x04" -- a DOCX is a ZIP archive
internally). A .txt or .exe renamed to .pdf has neither signature and is
rejected regardless of its extension or claimed content-type.

WHY THIS ISN'T JUST "CALL extract.py's _detect_file_type()" (deliberate,
not an oversight): that function takes an already-on-disk Path and reads
its own bytes from it -- by the time a file exists on disk in this
project's storage layout, this milestone's whole point (reject bad
uploads "at the door", before they ever reach storage) has already been
missed. This module's own detect_real_file_type() runs against bytes
already read from the live upload stream, before anything is written
anywhere. The two functions deliberately check the identical two magic-
byte signatures (not bespoke, drift-prone application logic -- these are
public, stable file-format specifications), so there is no meaningful
risk of the two diverging in practice; ingestion/extract.py's own
_detect_file_type() still runs again, unchanged, when the pipeline later
calls extract_text() on the now-validated stored file -- real defense in
depth, not a single point of failure.

ACCEPTED, PRE-EXISTING GAP (inherited from M5's own _detect_file_type()
design, not introduced or newly accepted here): a real .xlsx or .pptx
file (also a ZIP archive internally) renamed to a .docx extension would
pass this check, since ZIP-based Office formats are only disambiguated
by extension, not by inspecting the ZIP's internal contents (e.g.
word/document.xml) -- the same limitation ingestion/extract.py's own
_detect_file_type() already has and this milestone does not touch or
restructure. Out of scope here: this milestone's own stated test cases
(a .txt/.exe/image renamed to .pdf or .docx) do not exercise this gap,
since none of those formats are ZIP-based to begin with.

EXPLICITLY OUT OF SCOPE (per this milestone's own spec, not an
oversight): virus/malware scanning is a real, much larger, different
concern -- a signature/heuristic-based magic-byte check says nothing
about whether a genuinely well-formed PDF/DOCX contains a malicious
embedded macro, exploit, or payload. Accepted as a known, explicit gap,
not attempted here.
"""

import logging
from pathlib import Path
from typing import BinaryIO

logger = logging.getLogger("clauseguard.ingestion.upload_validation")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25MB, per this milestone's own spec

# Read/write in fixed-size chunks rather than loading the whole upload
# into memory at once -- itself a real resource-exhaustion vector on top
# of a spoofed Content-Length (an attacker claiming a small size while
# streaming an enormous body must not be able to force this process to
# buffer all of it in memory before the real byte-count check below ever
# gets a chance to reject it).
_CHUNK_SIZE = 1024 * 1024  # 1MB

# The same two real, well-known file-format magic-byte signatures
# ingestion/extract.py's own _detect_file_type() checks -- see this
# module's own docstring for why this isn't imported from there directly.
_PDF_MAGIC = b"%PDF-"
_ZIP_MAGIC = b"PK\x03\x04"


class UploadValidationError(Exception):
    """Base class for any pre-pipeline upload rejection. Callers
    (routes/contracts.py) catch this (or its specific subclasses below)
    and convert it into a clear, specific 4xx HTTP response -- never a
    generic 500 or a raw stack trace reaching the client."""


class FileTooLargeError(UploadValidationError):
    """Raised when an upload exceeds MAX_UPLOAD_BYTES -- either via the
    cheap Content-Length fast-path check, or (the real, unspoofable
    enforcement) via the true running byte count during streaming
    read/write."""


class InvalidFileTypeError(UploadValidationError):
    """Raised when a file's real, inspected bytes do not match a known
    PDF or DOCX signature -- regardless of what its filename extension or
    claimed Content-Type said."""


def validate_content_length_header(declared_size: int | None) -> None:
    """Fast-path rejection using the CLIENT-SUPPLIED Content-Length
    header -- cheap (rejects before reading any of the body), but NEVER
    trusted as the actual size limit. A missing header, or one that
    understates the real size, does NOT grant a pass: the authoritative
    check is validate_and_stream_to_disk()'s own running byte count,
    counted from real bytes actually read off the wire, which a client
    cannot lie its way past.
    """
    if declared_size is not None and declared_size > MAX_UPLOAD_BYTES:
        raise FileTooLargeError(
            f"Declared upload size ({declared_size} bytes) exceeds the "
            f"{MAX_UPLOAD_BYTES}-byte (25MB) limit."
        )


def detect_real_file_type(header_bytes: bytes, filename: str | None) -> str:
    """TRUE, byte-level MIME detection against `header_bytes` (the first
    few real bytes already read from the upload stream -- 8 bytes is
    sufficient for both signatures checked here). Returns "pdf" or
    "docx", or raises InvalidFileTypeError -- never trusts the filename
    extension or a Content-Type header for the PDF/ZIP determination
    itself; the extension is consulted ONLY for the same zip-format
    disambiguation ingestion/extract.py's own _detect_file_type() already
    relies on (see this module's own docstring's ACCEPTED, PRE-EXISTING
    GAP note for the one limitation that carries over from that design).
    """
    name = (filename or "").lower()

    if header_bytes.startswith(_PDF_MAGIC):
        return "pdf"

    if header_bytes.startswith(_ZIP_MAGIC):
        if name.endswith(".docx"):
            return "docx"
        raise InvalidFileTypeError(
            f"'{filename}' is a zip-based file but not a .docx (got "
            f"extension {Path(name).suffix or '(none)'!r}). Rejected -- "
            "real file bytes were inspected, not the claimed extension."
        )

    raise InvalidFileTypeError(
        f"'{filename}' does not match a known PDF or DOCX file signature. "
        "Its real bytes were inspected directly (not its claimed "
        "content-type or filename extension) -- only .pdf and .docx are "
        "supported."
    )


def validate_and_stream_to_disk(
    source_file: BinaryIO,
    destination_path: Path,
    filename: str | None,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> int:
    """Copies source_file's real contents to destination_path in fixed-
    size chunks, enforcing BOTH real checks as bytes actually arrive --
    never after the fact, never on a value the client merely claimed:

      1. MIME/magic-byte check on the very first chunk, BEFORE any byte
         is written to disk -- an invalid file is rejected without ever
         touching storage.
      2. A running true byte-count against max_bytes, checked on every
         chunk -- the moment real bytes written exceeds the limit, the
         partial file is deleted and FileTooLargeError is raised,
         regardless of any Content-Length the client claimed earlier.

    On ANY rejection, the partially-written file (if one was started) is
    deleted -- a rejected upload never leaves bytes behind in storage.

    Returns the real total byte count written, for logging/diagnostics.
    """
    total = 0
    type_checked = False
    try:
        with destination_path.open("wb") as out_file:
            while True:
                chunk = source_file.read(_CHUNK_SIZE)
                if not chunk:
                    break

                if not type_checked:
                    # The first chunk is always large enough to contain
                    # both magic-byte signatures (4-5 bytes) unless the
                    # entire upload itself is smaller than that -- in
                    # which case detect_real_file_type()'s own
                    # startswith() checks correctly fail to match either
                    # real signature, no special-casing needed.
                    detect_real_file_type(chunk[:8], filename)
                    type_checked = True

                total += len(chunk)
                if total > max_bytes:
                    raise FileTooLargeError(
                        f"Upload exceeded the {max_bytes}-byte (25MB) limit "
                        f"during actual read (real bytes written so far: "
                        f"{total}) -- rejected regardless of any claimed "
                        "Content-Length."
                    )
                out_file.write(chunk)
    except (FileTooLargeError, InvalidFileTypeError):
        destination_path.unlink(missing_ok=True)
        raise

    if total == 0:
        # An empty upload has no magic bytes at all to inspect -- reject
        # explicitly rather than silently treating "nothing" as valid.
        destination_path.unlink(missing_ok=True)
        raise InvalidFileTypeError(
            f"'{filename}' is empty -- cannot verify it is a real PDF or DOCX."
        )

    return total
