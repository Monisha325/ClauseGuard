"""M35: persistent audit log for validation failures, retries, and final
flag decisions across the real pipeline (M13/M28's API-error retry,
M33's schema-validation retry, M34's grounding-check retry all write
here -- see observability/audit.py's write_audit_entry() for the actual
write path, and pipeline/run_contract.py + agent/validate.py for the
real call sites).

contract_id carries an explicit index (mirroring FlaggedClause.contract_id's
own established index=True precedent, models/flagged_clause.py) -- an
audit log queried by contract_id with no index on that column becomes
unusable at real scale, and this project has a standing discipline
against declaring an index in the Python model and merely assuming it
took effect (see observability/audit.py's module docstring and this
milestone's own test script for the real `\\d audit_log` schema check).

clause_id is nullable: some events are contract-level (e.g. a pipeline-
wide failure) rather than tied to one specific clause. Its FK uses
ondelete="SET NULL" -- REPRODUCED LIVE during this milestone's own
testing (not theoretical): pipeline/run_contract.py's persist_chunks()
deletes and re-inserts a contract's Clause rows on every pipeline
re-run (fresh clause_id UUIDs each time, same as the bug M28 already
fixed once for flagged_clauses -- see that model's own history in
pipeline/run_contract.py's docstring). Without ondelete="SET NULL", a
second pipeline run on a contract that already has ANY audit_log rows
from its first run hits a real psycopg2.errors.ForeignKeyViolation on
persist_chunks()'s own `DELETE FROM clauses` -- old audit_log rows still
reference the about-to-be-deleted clause_id. Unlike flagged_clauses
(current-state, safe to delete-and-replace on every re-run), the audit
log is explicitly meant to persist HISTORICALLY across re-runs -- so the
fix here is letting the now-orphaned clause_id go to NULL (the row and
its contract_id/event_type/details/created_at survive; only the pointer
to a clause row that no longer exists is cleared), not pre-deleting
audit_log rows the way run_contract.py pre-deletes flagged_clauses.

event_type is a free-text String, not a DB-level enum -- same
established convention as FlaggedClause.severity and Contract.status
elsewhere in this codebase (no migration needed to add a new event type
later). See observability/audit.py for the actual set of values this
milestone writes.

details is a free-text column holding whatever event-specific context
is useful (the actual validation error, the actual severity assigned,
the actual retry reason) -- not a structured JSON column, since nothing
here needs to be queried on its own sub-fields, only read by a human or
grepped as text; a plain unbounded String keeps this consistent with
FlaggedClause.explanation/citation's own existing convention rather than
introducing a new column type for no querying benefit this milestone
actually needs.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID

from db import Base


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # M37 ADDITION -- ondelete="CASCADE", a DELIBERATE DEPARTURE from
    # clause_id's own ondelete="SET NULL" just below, not an
    # inconsistency: that SET NULL choice was for M35's re-index
    # scenario, where the CONTRACT still exists and only its Clause rows
    # get refreshed -- contract_id survives there as a real, still-
    # meaningful anchor to filter/query by, so nulling just clause_id was
    # the right partial degradation. routes/contracts.py's DELETE
    # endpoint (M37) is a different scenario entirely: the user is
    # explicitly, deliberately asking for the WHOLE contract's data to be
    # gone from every store it lives in. If contract_id itself were
    # nulled instead of cascading, these rows would lose their ONLY
    # anchor -- unlike the clause_id case, there is no other surviving
    # column left to ever find, filter, or make sense of an audit_log row
    # with a null contract_id again; it would just become permanent,
    # unlinkable noise, not a preserved historical record. Cascading the
    # whole row is the honest choice here, consistent with this
    # milestone's own goal ("removing data from every store it actually
    # lives in").
    contract_id = Column(UUID(as_uuid=True), ForeignKey("contracts.id", ondelete="CASCADE"), nullable=False, index=True)
    clause_id = Column(UUID(as_uuid=True), ForeignKey("clauses.id", ondelete="SET NULL"), nullable=True, index=True)
    event_type = Column(String, nullable=False)
    details = Column(String, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
