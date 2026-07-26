import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID

from db import Base


class Clause(Base):
    __tablename__ = "clauses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # M37: ondelete="CASCADE" -- a Clause row is pure structural data
    # (extracted/chunked text) with zero independent meaning once its
    # parent Contract is gone; it is not a historical record worth
    # preserving on its own (contrast with AuditLog, models/audit_log.py,
    # which explicitly IS meant to persist independently in the M35
    # re-index scenario). See routes/contracts.py's DELETE endpoint for
    # the full per-table FK audit this decision is part of.
    contract_id = Column(UUID(as_uuid=True), ForeignKey("contracts.id", ondelete="CASCADE"), nullable=False, index=True)
    heading_path = Column(String, nullable=False)
    text = Column(String, nullable=False)
    page_number = Column(Integer, nullable=False)
    # Stage-1 classification result (see classification/heading_match.py).
    # Nullable: a clause whose heading doesn't match any of the 5 known
    # categories stays category=None — that's the correct, expected
    # result at this stage (stage-2 embedding fallback is M23), not a
    # gap to backfill with a default guess.
    category = Column(String, nullable=True, default=None)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
