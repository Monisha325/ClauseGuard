"""M18: schema + loader for human-labeled ground-truth clause data.

Standalone and offline on purpose: this module has ZERO runtime
dependency on backend/ (no imports from it, no DB, no network, no
Docker). A labeler or eval script should be able to run this with
nothing but `pip install pydantic` -- it does not need the app stack
running. This is why REAL_CATEGORIES/SEVERITY_VALUES below are this
module's OWN copies rather than imports from backend/classification or
backend/agent, even though those real modules are lightweight enough to
import in principle -- keeping eval/ genuinely decoupled matters more
than avoiding a small duplication risk, which is mitigated instead by
the cross-check note below and by this milestone's own testing (step 5
re-verifies both value sets directly against the real source files
before this is considered done).

LabeledClause is the single source of truth for the schema: it is
defined ONCE here, and eval/labeled_set/schema.json is the JSON Schema
DERIVED from it (via LabeledClause.model_json_schema()), not a
separately hand-maintained document -- the same "define once, derive
the rest" pattern backend/agent/tools.py already established for
flag_clause's own tool schema, so the two representations can never
drift apart.

Validation goes through this Pydantic model directly, not a generic
jsonschema-library round-trip through schema.json -- Pydantic's own
ValidationError already carries structured, per-field detail, which
produces materially better error messages than a generic JSON Schema
validator would, and this milestone's error-quality requirement is
explicit: a labeler filling this out by hand needs to find and fix a
mistake from the error message alone.
"""

import json
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

# Cross-checked directly against backend/classification/heading_match.py's
# CATEGORIES tuple (verified by reading the file, not retyped from
# memory or milestone-doc recollection). Must be kept in sync by hand if
# that tuple ever changes -- the same kind of manual-sync obligation
# heading_match.py's own docstring already places on itself for its
# external-doc dependency ("If M20/M26's threshold table is ever
# revised, the CATEGORIES tuple here must be kept in sync with it").
REAL_CATEGORIES = (
    "Limitation of Liability",
    "Indemnification",
    "Termination",
    "Governing Law / Jurisdiction",
    "Confidentiality",
)

# classify_heading() genuinely returns None for a heading that doesn't
# match any category -- M8's own docs are explicit that this is "the
# correct, expected result at this stage, not an error or a guess." JSON
# has no clean way to make "null" one member of an otherwise-string enum
# that also needs to reject truly-missing data, so labeled rows use the
# literal string "Unclassified" to represent that same real outcome.
CATEGORY_VALUES = REAL_CATEGORIES + ("Unclassified",)

# Cross-checked directly against backend/agent/tools.py's
# FlagClauseOutput.severity: Literal["low", "medium", "high"]. Must be
# kept in sync by hand if that Literal ever changes.
SEVERITY_VALUES = ("low", "medium", "high")


class LabeledClause(BaseModel):
    """One human-labeled ground-truth row for one real, persisted clause.

    Two fields exist purely to absorb real labeling ambiguity rather
    than forcing a single, possibly-lossy judgment call:

    - `alternate_category`: for a clause that genuinely reads as two
      categories at once (e.g. a clause that is arguably both
      Indemnification and Limitation of Liability). Optional -- most
      rows will leave this null, and that's the expected common case,
      not an incomplete row.
    - `notes`: a free-text release valve for anything else worth
      recording -- most commonly a borderline severity call ("felt like
      it was between medium and high because..."). Severity is only a
      3-point ordinal scale, so a second structured `alternate_severity`
      field would be more machinery than the ambiguity actually needs;
      free text is the right amount of flexibility for that specific
      case. `notes` is also the general escape hatch for any other
      judgment call a labeler wants on record.

    Neither flexibility field is required. A clean, unambiguous row
    should have both as null/omitted -- this is normal, not a sign the
    row is incomplete.
    """

    contract_id: uuid.UUID = Field(
        description="Matches Contract.id (Postgres UUID) for the real contract this clause belongs to."
    )
    clause_id: uuid.UUID = Field(
        description="Matches Clause.id (Postgres UUID) for the specific persisted clause being labeled."
    )
    category: Literal[*CATEGORY_VALUES] = Field(
        description=(
            "The labeler's primary category judgment. Must be one of M8's 5 "
            "real categories, or 'Unclassified' if the clause genuinely fits "
            "none of them -- a real, valid outcome, not an error."
        )
    )
    alternate_category: Literal[*CATEGORY_VALUES] | None = Field(
        default=None,
        description=(
            "A second, plausible category if this clause is genuinely "
            "ambiguous between two categories. Leave null/omit when the "
            "primary category call was unambiguous (the common case)."
        ),
    )
    is_risky: bool = Field(
        description="The labeler's judgment on whether this clause poses a real, notable risk to the reviewing party."
    )
    severity: Literal[*SEVERITY_VALUES] = Field(
        description=(
            "Must match flag_clause's real severity scale exactly (see "
            "backend/agent/tools.py's FlagClauseOutput). If the call feels "
            "genuinely borderline between two levels, record the single best "
            "call here and use `notes` for the uncertainty."
        )
    )
    query: str = Field(
        min_length=1,
        description=(
            "A natural-language query the labeler believes SHOULD retrieve "
            "this clause via semantic search -- what a later Hit@K eval will "
            "issue against the real retrieval endpoint."
        ),
    )
    notes: str | None = Field(
        default=None,
        description=(
            "Free-text field for any labeler uncertainty -- a borderline "
            "severity call, reasoning behind an alternate_category, or any "
            "other judgment call worth recording. Optional; leave null/omit "
            "if there's nothing notable about this row."
        ),
    )


class LabeledDataError(Exception):
    """Raised for any problem loading or validating a labeled-clause
    file. Always includes the specific row (1-indexed, matching how a
    human would count entries in the file) and field at fault -- this
    file is filled out by hand, not just consumed by machines, so a bare
    "validation failed" message is not good enough.
    """


def load_labeled_clauses(path: str | Path) -> list[LabeledClause]:
    """Load and validate a labeled-clause JSON file.

    Returns a list of validated LabeledClause objects in file order.
    Raises LabeledDataError (not a raw json.JSONDecodeError or
    pydantic.ValidationError) with a row-and-field-specific message if
    the file isn't valid JSON, isn't a list, or any row fails schema
    validation.
    """
    path = Path(path)

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise LabeledDataError(f"{path}: not valid JSON -- {exc}") from exc

    if not isinstance(raw, list):
        raise LabeledDataError(
            f"{path}: expected the top-level JSON value to be a list of "
            f"labeled-clause objects, got {type(raw).__name__}."
        )

    results: list[LabeledClause] = []
    for i, row in enumerate(raw):
        try:
            results.append(LabeledClause.model_validate(row))
        except ValidationError as exc:
            field_errors = "; ".join(
                f"field '{'.'.join(str(p) for p in err['loc'])}': {err['msg']} "
                f"(got: {err.get('input')!r})"
                for err in exc.errors()
            )
            raise LabeledDataError(
                f"{path}: row {i + 1} of {len(raw)} is invalid -- {field_errors}"
            ) from exc

    return results


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "labeled_set/sample_placeholder.json"
    clauses = load_labeled_clauses(target)
    print(f"Loaded {len(clauses)} labeled clause(s) from {target}:\n")
    for i, clause in enumerate(clauses, start=1):
        print(f"--- row {i} ---")
        print(clause.model_dump_json(indent=2))
        print()
