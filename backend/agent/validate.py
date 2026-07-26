"""M33: retry-once-on-validation-failure logic for flag_clause()'s
Groq tool-call output.

ORTHOGONAL to pipeline/run_contract.py's existing _flag_with_retry() /
_retry_reason() (M13/M28): that logic retries on API-level failures --
429 rate limit, 503 service unavailable -- calls that never got a
successful response back from Groq at all, handled entirely outside
this module and NOT modified by it. This module handles the opposite,
previously-unhandled case: Groq's tool call comes back successfully
(no API error, no exception from the request itself), but the returned
arguments fail Pydantic validation against FlagClauseOutput (missing
field, wrong type, invalid enum value). M13's original design treated
this exactly like any other FlagClauseError -- straight to the
flagging_failed sentinel, zero retry. Since a schema-invalid tool call
is plausibly a one-off model mistake rather than a systemic failure,
one retry -- with the specific validation error fed back into the
prompt -- gives the model a concrete, actionable chance to self-correct
before this clause is written off.

WHY EXACTLY ONE RETRY, NOT A LOOP (a real, explicitly accepted risk,
not an oversight): MAX_VALIDATION_RETRIES=1 is a hardcoded, deliberate
ceiling -- validate_with_retry() below iterates a plain, bounded
`range(...)`, not a `while True`, so it is structurally incapable of
attempting more than MAX_VALIDATION_RETRIES + 1 (2) total calls,
regardless of how many times the model keeps getting it wrong. If a
real clause's real model output fails validation twice in a row,
that's exactly what needs_manual_review exists for -- a visible,
human-reviewable placeholder, not an unbounded retry loop silently
burning API quota on a clause the model may never assess correctly.

COMPOSITION WITH THE EXISTING API-ERROR RETRY (verified, not assumed):
`attempt` (the callback flag_clause.py passes in) raising anything
OTHER than a bare pydantic ValidationError -- e.g. FlagClauseError from
an actual network/API failure -- is NOT caught here at all; it
propagates straight through validate_with_retry() unmodified, all the
way out of flag_clause(), for pipeline/run_contract.py's own,
completely separate _flag_with_retry() to catch and retry on its own
axis. Each top-level flag_clause() call gets its own fresh
validate_with_retry() invocation (and thus its own fresh 1-retry
budget) -- the two retry mechanisms never share state and never
double-count each other's attempts.

M35 ADDITION: optional contract_id/clause_id parameters, threaded in
from pipeline/run_contract.py's loop via flag_clause()'s own new
optional parameters (see agent/flag_clause.py's docstring) purely so
THIS module can write its own audit entries (models/audit_log.py, via
observability/audit.py's write_audit_entry()) for a validation failure,
a validation-retry success, and the terminal needs_manual_review state
-- the three events this milestone's spec explicitly assigns to this
file. Both default to None so this function's original signature keeps
working unchanged for any caller that doesn't have real IDs to give it
(e.g. this module's own tests, or flag_clause.py's __main__ smoke test)
-- when either is None, the audit write is skipped entirely rather than
attempted with a null contract_id (which would just fail the column's
NOT NULL constraint on every single call). write_audit_entry() itself
never raises (see its own docstring), so even a real DB failure during
one of these writes cannot interrupt this function's actual validation-
retry logic -- only the audit trail for that one event is lost.
"""

import logging
import uuid
from typing import Callable

from pydantic import ValidationError

from agent.tools import FlagClauseOutput
from observability.audit import (
    EVENT_VALIDATION_FAILURE,
    EVENT_VALIDATION_NEEDS_MANUAL_REVIEW,
    EVENT_VALIDATION_RETRY_SUCCESS,
    write_audit_entry,
)

logger = logging.getLogger("clauseguard.agent.validate")

MAX_VALIDATION_RETRIES = 1  # exactly one retry -- see module docstring


class NeedsManualReviewError(Exception):
    """Raised by validate_with_retry() when flag_clause()'s Groq tool
    call output fails Pydantic validation on BOTH the original attempt
    and its one retry.

    A distinct terminal state from FlagClauseError's OTHER failure
    modes (missing API key, rejected request, no tool call returned at
    all, an actual network/API error) -- those represent real
    API/infrastructure failures, and pipeline/run_contract.py's
    FLAGGING_FAILED_SEVERITY sentinel remains reserved for exactly
    those, per M13's original design, unchanged by this milestone. This
    exception means something categorically different: Groq
    responded successfully TWICE, but its own output was schema-invalid
    both times -- worth a human's attention, not the same bucket as an
    outage. Deliberately NOT a subclass of FlagClauseError (this module
    has no reason to import agent/flag_clause.py at all, and keeping
    these two exception families separate keeps the two failure modes
    -- infrastructure vs. model-output-quality -- from ever being
    conflated by an incautious `except FlagClauseError` catching this
    too by accident); pipeline/run_contract.py catches this type
    explicitly and distinctly.
    """


def validate_with_retry(
    attempt: Callable[[str | None], FlagClauseOutput],
    clause_ref: str,
    contract_id: uuid.UUID | None = None,
    clause_id: uuid.UUID | None = None,
) -> FlagClauseOutput:
    """Calls `attempt(validation_error)` up to MAX_VALIDATION_RETRIES + 1
    times (2, with the current constant): first with
    validation_error=None (the original attempt, no retry context yet),
    then -- ONLY if that raises a pydantic ValidationError -- once more
    with the specific error message string from the previous attempt,
    so `attempt`'s own caller (flag_clause.py) can feed it back into a
    corrective retry prompt that names the exact schema problem.

    `attempt` raising anything OTHER than a bare pydantic ValidationError
    (e.g. FlagClauseError from an actual API failure) is NOT caught
    here -- it propagates straight through unmodified, so this
    validation-retry logic never intercepts, masks, or double-counts
    pipeline/run_contract.py's own, separate API-error retry (see this
    module's own docstring, COMPOSITION section).

    Raises NeedsManualReviewError if every attempt fails validation --
    structurally cannot retry further than MAX_VALIDATION_RETRIES (the
    loop below is a plain, bounded `range()`, not a condition that could
    drift into looping indefinitely).

    contract_id/clause_id (M35): when both are real (non-None) values,
    every validation failure, every successful retry, and the terminal
    needs_manual_review state is written as an AuditLog row via
    write_audit_entry() -- see this module's own docstring, M35 ADDITION,
    for why these default to None and why a failed audit write can never
    affect the return value or control flow below.
    """
    validation_error: str | None = None
    total_attempts = MAX_VALIDATION_RETRIES + 1

    for attempt_number in range(total_attempts):
        try:
            result = attempt(validation_error)
        except ValidationError as exc:
            attempts_remaining = total_attempts - (attempt_number + 1)
            logger.warning(
                "flag_clause validation FAILED for clause_ref=%s (attempt "
                "%d/%d) -- %s.",
                clause_ref, attempt_number + 1, total_attempts,
                f"retrying with the validation error fed back into the prompt ({attempts_remaining} retry(ies) left)"
                if attempts_remaining > 0
                else "no retries left, marking needs_manual_review",
            )
            if contract_id is not None:
                write_audit_entry(
                    contract_id=contract_id,
                    clause_id=clause_id,
                    event_type=EVENT_VALIDATION_FAILURE,
                    details=(
                        f"clause_ref={clause_ref} attempt {attempt_number + 1}/"
                        f"{total_attempts} failed schema validation "
                        f"({attempts_remaining} retry(ies) left): {exc}"
                    ),
                )
            validation_error = str(exc)
            continue

        if attempt_number > 0:
            logger.info(
                "flag_clause validation retry SUCCEEDED for clause_ref=%s "
                "(attempt %d/%d).",
                clause_ref, attempt_number + 1, total_attempts,
            )
            if contract_id is not None:
                write_audit_entry(
                    contract_id=contract_id,
                    clause_id=clause_id,
                    event_type=EVENT_VALIDATION_RETRY_SUCCESS,
                    details=(
                        f"clause_ref={clause_ref} validation retry succeeded "
                        f"on attempt {attempt_number + 1}/{total_attempts}"
                    ),
                )
        return result

    if contract_id is not None:
        write_audit_entry(
            contract_id=contract_id,
            clause_id=clause_id,
            event_type=EVENT_VALIDATION_NEEDS_MANUAL_REVIEW,
            details=(
                f"clause_ref={clause_ref} failed schema validation on all "
                f"{total_attempts} attempt(s) (1 original + "
                f"{MAX_VALIDATION_RETRIES} retry). Last validation error: "
                f"{validation_error}"
            ),
        )

    raise NeedsManualReviewError(
        f"flag_clause tool call output failed Pydantic validation for "
        f"clause_ref={clause_ref} on all {total_attempts} attempt(s) "
        f"(1 original + {MAX_VALIDATION_RETRIES} retry). Last validation "
        f"error: {validation_error}"
    )
