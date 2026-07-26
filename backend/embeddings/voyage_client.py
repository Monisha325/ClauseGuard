"""Stage-1 smoke test for Voyage AI embeddings: a single text -> vector.

M9 scope only: one function, one real API call per invocation, no batching,
no ChromaDB indexing, no retry/backoff, no caching (all deferred to M10+).

SDK choice: the official `voyageai` PyPI package (actively maintained,
frequent releases through 0.4.x as of this writing) is used instead of a
raw HTTP call, since it handles request/response shaping and auth headers
for us and is the vendor-recommended integration path.

API key: read from the VOYAGE_API_KEY environment variable (see
.env.example), the same pattern config.py uses for JWT_SECRET_KEY. Unlike
JWT_SECRET_KEY, this is validated at CALL time, not import/startup time —
this module isn't wired into the app's startup path yet (that's M10), so
importing it must not crash processes that don't need it; the failure
belongs at the point where a real API call is actually attempted.
"""

import os

import voyageai

MODEL = "voyage-3-lite"


class VoyageEmbeddingError(Exception):
    """Raised for any failure embedding text via Voyage AI — missing API
    key, a rejected request, or a network/API-level failure. Always
    raised with context rather than letting a raw SDK/HTTP exception
    propagate unexplained.
    """


def embed_text(text: str) -> list[float]:
    """Return the voyage-3-lite embedding vector for `text` as a plain
    list of floats (never the SDK's raw response object).

    Raises VoyageEmbeddingError if VOYAGE_API_KEY is unset, or if the API
    call fails for any reason (bad key, network error, rate limit,
    malformed request, etc.).
    """
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise VoyageEmbeddingError(
            "VOYAGE_API_KEY is not set in the environment. Set it in "
            ".env (see .env.example) before calling embed_text()."
        )

    client = voyageai.Client(api_key=api_key)

    try:
        result = client.embed([text], model=MODEL)
    except Exception as exc:
        raise VoyageEmbeddingError(
            f"Voyage AI embedding request failed (model={MODEL!r}): {exc}"
        ) from exc

    return list(result.embeddings[0])
