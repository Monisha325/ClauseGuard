import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID

from db import Base


class FlaggedClause(Base):
    __tablename__ = "flagged_clauses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # M37: ondelete="CASCADE" on both FKs below -- a FlaggedClause row is a
    # per-clause artifact of THIS contract's own processing (a risk
    # assessment for a specific clause of a specific contract), not an
    # independent record. clause_id is NOT NULL (can't be nulled the way
    # AuditLog.clause_id can, models/audit_log.py) -- once its Clause
    # cascades away (see models/clause.py's own ondelete), the
    # FlaggedClause row referencing it is equally meaningless and must go
    # too. See routes/contracts.py's DELETE endpoint for the full
    # per-table FK audit this decision is part of.
    clause_id = Column(UUID(as_uuid=True), ForeignKey("clauses.id", ondelete="CASCADE"), nullable=False, index=True)
    # Denormalized alongside clause_id (not just derivable via a join) so
    # the whole table can be queried/filtered per-contract directly, the
    # same way M10's Chroma metadata carries contract_id redundantly for
    # its own scoping needs.
    contract_id = Column(UUID(as_uuid=True), ForeignKey("contracts.id", ondelete="CASCADE"), nullable=False, index=True)
    # "low" / "medium" / "high" (agent/tools.py's FlagClauseOutput), or one
    # of two distinct failure sentinels (see pipeline/run_contract.py):
    # "flagging_failed" -- an actual API/infrastructure failure (M13) --
    # or "needs_manual_review" -- Groq's tool call output failed Pydantic
    # schema validation on both the original attempt and its one retry
    # (M33's agent/validate.py), a categorically different failure mode
    # (the model responded successfully but produced bad data, not an
    # outage). Both are visible placeholder rows rather than a silently-
    # missing clause. No DB-level enum/check constraint on this column --
    # a plain free-text String -- so adding needs_manual_review required
    # no schema migration, just this new value being written.
    severity = Column(String, nullable=False)
    explanation = Column(String, nullable=False)
    citation = Column(String, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
