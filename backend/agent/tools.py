"""The flag_clause tool: defined ONCE here as a Pydantic model.

FlagClauseOutput.model_json_schema() is reused directly as the function's
parameters schema sent to the LLM provider, and the same model validates
whatever comes back. There is exactly one schema definition in this
codebase — the API-call shape and the response-validation shape can
never drift apart the way two independently hand-written schemas could.

A plain typing.Literal (not a Pydantic/enum.Enum class) is used for
severity on purpose: Pydantic v2 inlines a Literal's allowed values
directly into the property's own JSON schema (`"enum": [...]`), whereas
an Enum class gets hoisted into a separate `$defs` block referenced via
`$ref`. Using Literal instead of Enum keeps the schema flat and simple.

PROVIDER MIGRATION (Gemini -> Groq, 2026-07-09): this project's
reasoning/tool-calling layer moved from Gemini (google.genai SDK,
FunctionDeclaration/types.Tool) to Groq (OpenAI-compatible SDK,
tools=[{"type": "function", "function": {...}}] dicts) -- confirmed live
against Groq's real API before this migration (see agent/flag_clause.py's
own module docstring for the full reasoning and real evidence). Both
tool-build functions below now return a plain dict in the OpenAI/Groq
function-calling shape instead of a google.genai types.Tool object --
FlagClauseOutput/SuggestNegotiationOutput themselves are UNCHANGED
(provider-agnostic validation shapes, never tied to Gemini specifically).
"""

from typing import Literal

from pydantic import BaseModel, Field

FLAG_CLAUSE_TOOL_NAME = "flag_clause"

FLAG_CLAUSE_TOOL_DESCRIPTION = (
    "Flag a contract clause with a risk assessment: its severity, an "
    "explanation of why it is risky or notable, and a citation back to "
    "the exact clause text being assessed."
)


class FlagClauseOutput(BaseModel):
    """Validated shape of a flag_clause tool call's arguments."""

    severity: Literal["low", "medium", "high"] = Field(
        description=(
            "Overall risk severity of this clause for the party reviewing "
            "it: low, medium, or high."
        )
    )
    explanation: str = Field(
        description=(
            "A concise, specific explanation of why this clause is risky "
            "or notable, referencing what the clause actually says — not "
            "a generic statement that could apply to any clause."
        )
    )
    citation: str = Field(
        description=(
            "The exact clause text (verbatim, or the specific risky "
            "portion of it) this assessment is based on, so the flag is "
            "traceable back to real source text."
        )
    )


def _parameters_schema() -> dict:
    schema = FlagClauseOutput.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema


def build_flag_clause_tool() -> dict:
    """Build the Groq/OpenAI-compatible tool definition for flag_clause,
    derived directly from FlagClauseOutput's own schema -- the same
    single-source-of-truth principle as the original Gemini version,
    just in the {"type": "function", "function": {...}} shape Groq's
    chat.completions.create(tools=[...]) expects (confirmed live against
    the real Groq API, see agent/flag_clause.py).
    """
    return {
        "type": "function",
        "function": {
            "name": FLAG_CLAUSE_TOOL_NAME,
            "description": FLAG_CLAUSE_TOOL_DESCRIPTION,
            "parameters": _parameters_schema(),
        },
    }


# --- M34: suggest_negotiation_point, the SECOND tool in this project ---
#
# TWO-TOOL CONFIGURATION DECISION (verified against M12's actual setup
# above before writing any of this, not assumed): flag_clause's forced
# tool-calling guarantee comes from tool_choice={"type": "function",
# "function": {"name": FLAG_CLAUSE_TOOL_NAME}} (Groq/OpenAI-compatible;
# originally Gemini's tool_config.function_calling_config mode="ANY" +
# allowed_function_names=[FLAG_CLAUSE_TOOL_NAME] -- see
# agent/flag_clause.py's _call_groq_once()) -- a real, independently-
# verified guarantee that the model calls THAT tool specifically, not
# "some tool from the list" or a plain-text response. That guarantee was
# verified for exactly ONE named tool. Putting BOTH tools in the same
# call would force SOME tool from the list, but not deterministically
# WHICH one -- a materially different, never-verified guarantee -- and
# the two tools are semantically unrelated calls anyway (different
# inputs, different callers, different points in the pipeline:
# flag_clause runs on every clause; suggest_negotiation_point only runs
# afterward, for a subset, using a retrieved-context prompt shape
# flag_clause never needs). This tool therefore gets its OWN, completely
# separate API call in agent/suggest_negotiation.py, with its own
# tool_choice scoped to just this one tool name -- the exact same
# single-tool-forcing pattern flag_clause.py already uses, just for a
# different tool.
SUGGEST_NEGOTIATION_TOOL_NAME = "suggest_negotiation_point"

SUGGEST_NEGOTIATION_TOOL_DESCRIPTION = (
    "Suggest ONE concrete negotiation point for a flagged contract clause, "
    "grounded in specific related clauses provided as context. Cite only "
    "the clause_id(s) of the related clauses actually provided that "
    "support this suggestion -- never a clause_id not present in that "
    "context."
)


class SuggestNegotiationOutput(BaseModel):
    """Validated shape of a suggest_negotiation_point tool call's
    arguments. cited_clause_ids is intentionally NOT constrained to a
    minimum length here (an empty list is valid): requiring at least one
    citation could pressure the model into fabricating one when it
    genuinely has no directly-relevant related clause to point to, which
    would work directly against this tool's whole grounding purpose.
    """

    suggestion: str = Field(
        description=(
            "A concrete, actionable negotiation point for the flagged "
            "clause -- specific wording or a specific change to propose, "
            "not generic advice."
        )
    )
    rationale: str = Field(
        description=(
            "Why this suggestion makes sense, referencing the related "
            "clause(s) cited in cited_clause_ids."
        )
    )
    cited_clause_ids: list[str] = Field(
        description=(
            "The clause_id(s), from the provided related-clauses context "
            "ONLY, that support this suggestion. Must never include a "
            "clause_id that was not explicitly provided in that context."
        )
    )


def _suggest_negotiation_parameters_schema() -> dict:
    schema = SuggestNegotiationOutput.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema


def build_suggest_negotiation_tool() -> dict:
    """Build the Groq/OpenAI-compatible tool definition for
    suggest_negotiation_point, derived directly from
    SuggestNegotiationOutput's own schema -- the exact same pattern
    build_flag_clause_tool() above uses for its tool.
    """
    return {
        "type": "function",
        "function": {
            "name": SUGGEST_NEGOTIATION_TOOL_NAME,
            "description": SUGGEST_NEGOTIATION_TOOL_DESCRIPTION,
            "parameters": _suggest_negotiation_parameters_schema(),
        },
    }
