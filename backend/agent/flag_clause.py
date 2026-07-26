"""M12 smoke test: send a single hardcoded clause to the LLM with the
flag_clause tool FORCED (tool_choice pinned to this one named tool), and
return a validated FlagClauseOutput.

Scope: ONE clause, ONE tool call. No retrieval integration (pulling real
clauses from Chroma/Postgres), no ReAct loop, no persistence, no FastAPI
route — all later milestones. This proves the tool-calling integration
itself works correctly, in isolation, before anything is built on top of
it.

PROVIDER MIGRATION (Gemini -> Groq, 2026-07-09): this project originally
ran this call against Gemini (google.genai SDK, gemini-2.5-flash-lite).
Migrated to Groq (OpenAI-compatible SDK) because Gemini's free tier's
real, confirmed daily cap (GenerateRequestsPerDayPerProjectPerModel-
FreeTier, 20 requests/day/model) repeatedly caused real, multi-hour
lockouts during actual use -- most recently reproduced live on
2026-07-09 (a single 4-clause contract burning its entire daily budget).
Model chosen: llama-3.3-70b-versatile -- confirmed via Groq's own live
docs (console.groq.com/docs/rate-limits) to be a current, generally-
available production model with real free-tier tool-calling support,
verified directly against the real API before this migration (three
real calls: a genuinely risky indemnification clause correctly scored
"high", a moderate termination-notice clause scored "medium", and a
boilerplate counterparts clause scored "low" -- content-differentiated
output, not a fixed/default response). Groq's own free-tier limits for
this model (30 RPM / 1,000 RPD / 12K TPM / 100K TPD, confirmed via
console.groq.com/docs/rate-limits) are dramatically more generous than
Gemini's 20 RPD, though NOT literally uncapped -- see
pipeline/run_contract.py's own module docstring for how the retry logic
now handles Groq's real error types.

Tool-use is forced, not hoped for: tool_choice is pinned to
{"type": "function", "function": {"name": FLAG_CLAUSE_TOOL_NAME}} (Groq's
real, confirmed equivalent of Gemini's tool_config mode="ANY" +
allowed_function_names -- verified live: the model cannot respond with
plain text instead of calling the named tool), so this is a real
API-level guarantee, not something left to prompt wording.

Output budget: MAX_OUTPUT_TOKENS below is set well below the model's own
much larger completion budget on purpose: if a call ever produces
truncated output, it's cheaper and faster to diagnose against a small
ceiling than to have silently burned a large number of tokens first.
"""

import hashlib
import json
import logging
import os
import uuid

from groq import Groq
from pydantic import ValidationError

from agent.tools import FLAG_CLAUSE_TOOL_NAME, FlagClauseOutput, build_flag_clause_tool
from agent.validate import validate_with_retry

logger = logging.getLogger("clauseguard.agent.flag_clause")

MODEL = "llama-3.3-70b-versatile"
MAX_OUTPUT_TOKENS = 2048


def _clause_ref(clause_text: str) -> str:
    """Short, stable identifier for a clause for log correlation, since
    flag_clause() only receives raw clause text, not a clause_id -- a
    hash prefix lets repeated/duplicate log lines for the same clause be
    matched up across calls without logging the full clause text every
    time.
    """
    return hashlib.sha256(clause_text.encode("utf-8")).hexdigest()[:12]


class FlagClauseError(Exception):
    """Raised for any failure flagging a clause via the LLM — missing API
    key, a rejected request, a response with no tool call at all, or a
    tool call whose arguments don't validate against FlagClauseOutput.
    Always raised with context rather than letting a raw SDK exception or
    malformed data propagate unexplained.
    """


def _build_prompt(clause_text: str, validation_error: str | None) -> str:
    """The base prompt is unchanged from M12. M33 ADDITION: when
    validation_error is not None, this is a RETRY prompt -- the ORIGINAL
    clause text (unchanged) plus the specific Pydantic validation error
    from the previous attempt, so the model has a concrete, actionable
    reason to correct its own tool call rather than a generic "try
    again."
    """
    base = (
        "You are a contract risk analyst. Assess the following contract "
        "clause for risk to the party reviewing it, and call the "
        f"{FLAG_CLAUSE_TOOL_NAME} tool with your assessment.\n\nClause:\n"
        + clause_text
    )
    if validation_error is None:
        return base
    return (
        base
        + "\n\nYour previous tool call's arguments failed validation "
        "against the required schema, with this specific error:\n"
        + validation_error
        + "\n\nCall the tool again with corrected arguments that satisfy "
        "this schema exactly."
    )


def _call_groq_once(
    client: Groq,
    tool: dict,
    clause_ref: str,
    prompt: str,
) -> FlagClauseOutput:
    """Exactly ONE real Groq request plus ONE validation attempt -- no
    retry logic of its own (see validate_with_retry() in agent/validate.py
    for that, and flag_clause() below for how the two compose).

    Raises FlagClauseError for any API/infrastructure failure (the
    request itself erroring, no tool call returned at all, the wrong
    tool called) -- unchanged in spirit from the original Gemini version.

    Raises a raw pydantic ValidationError (deliberately NOT wrapped in
    FlagClauseError) if the tool call's own arguments fail schema
    validation -- left unwrapped specifically so validate_with_retry()
    can catch it to drive the retry-with-feedback logic, and so this
    failure mode is never confused with (or accidentally caught by
    code expecting only) an actual API/infrastructure failure.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": FLAG_CLAUSE_TOOL_NAME}},
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        )
    except Exception as exc:
        raise FlagClauseError(
            f"Groq request failed (model={MODEL!r}): {exc}"
        ) from exc

    # Observability: log finish_reason + token usage for EVERY real call
    # that got a response back, success or failure, so a pattern like
    # "finish_reason keeps coming back non-tool_calls" is visible in
    # aggregate over time rather than only discoverable by accident
    # during a specific investigation.
    choice = response.choices[0]
    logger.info(
        "flag_clause call clause_ref=%s model=%s finish_reason=%s usage=%s",
        clause_ref, MODEL, choice.finish_reason, response.usage,
    )

    tool_calls = choice.message.tool_calls
    if not tool_calls:
        raise FlagClauseError(
            "Groq did not return a tool call for flag_clause — got a "
            f"plain-text response instead: {choice.message.content!r}"
        )

    call = tool_calls[0]
    if call.function.name != FLAG_CLAUSE_TOOL_NAME:
        raise FlagClauseError(
            f"Groq called unexpected tool {call.function.name!r}, expected "
            f"{FLAG_CLAUSE_TOOL_NAME!r}"
        )

    try:
        args = json.loads(call.function.arguments)
    except json.JSONDecodeError as exc:
        raise FlagClauseError(
            f"flag_clause tool call arguments were not valid JSON: {exc} "
            f"(raw: {call.function.arguments!r})"
        ) from exc

    try:
        return FlagClauseOutput.model_validate(args)
    except ValidationError as exc:
        # M33: logged HERE, at the point of failure, so BOTH the original
        # attempt's and (if it also fails) the retry's raw output and
        # validation error each get a distinct, real log line -- not just
        # a single summary after the fact once validate_with_retry()
        # gives up.
        logger.warning(
            "flag_clause tool call arguments failed validation for "
            "clause_ref=%s -- raw args: %r; validation error: %s",
            clause_ref, args, exc,
        )
        raise


def flag_clause(
    clause_text: str,
    contract_id: uuid.UUID | None = None,
    clause_id: uuid.UUID | None = None,
) -> FlagClauseOutput:
    """Send clause_text to Groq and return its flag_clause assessment as
    a validated FlagClauseOutput. Raises FlagClauseError on an
    API/infrastructure failure.

    M33: raises agent.validate.NeedsManualReviewError (not
    FlagClauseError) if the tool call's output fails Pydantic validation
    on both the original attempt and its one retry -- see
    agent/validate.py's validate_with_retry() for the full mechanism.

    M35: contract_id/clause_id are optional pure pass-through -- this
    function has no use for them itself, it only forwards them to
    validate_with_retry() so THAT module can write its own audit entries
    with real IDs. Both default to None so any existing caller that
    doesn't have real IDs (e.g. this module's own __main__ smoke test
    below) is unaffected.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise FlagClauseError(
            "GROQ_API_KEY is not set in the environment. Set it in "
            ".env (see .env.example) before calling flag_clause()."
        )

    client = Groq(api_key=api_key)
    tool = build_flag_clause_tool()
    clause_ref = _clause_ref(clause_text)

    def attempt(validation_error: str | None) -> FlagClauseOutput:
        prompt = _build_prompt(clause_text, validation_error)
        return _call_groq_once(client, tool, clause_ref, prompt)

    return validate_with_retry(
        attempt, clause_ref, contract_id=contract_id, clause_id=clause_id
    )


if __name__ == "__main__":
    import sys

    clause = sys.argv[1] if len(sys.argv) > 1 else (
        "Consultant shall indemnify, defend, and hold harmless Client, "
        "its officers, directors, and affiliates from and against any "
        "and all claims, damages, losses, liabilities, and expenses of "
        "any kind whatsoever, whether or not arising from Consultant's "
        "negligence, willful misconduct, or breach of this Agreement, "
        "with no cap or limitation of any kind on the amount or duration "
        "of such indemnification obligation, which shall survive "
        "termination of this Agreement in perpetuity."
    )
    result = flag_clause(clause)
    print(result.model_dump_json(indent=2))
