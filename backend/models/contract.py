import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID

from db import Base


class Contract(Base):
    __tablename__ = "contracts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    # Free-text, not a DB-level enum, on purpose (unchanged design from
    # M4) -- no migration needed to add new values. M15 uses:
    # "processing" (set by the upload route immediately, before the
    # Celery task even starts) -> "complete" or "failed" (set by
    # worker/tasks.py once the pipeline actually finishes or raises).
    status = Column(String, nullable=False, default="uploaded")

    # M40: the REAL current pipeline stage, updated (and committed) by
    # run_contract_pipeline() itself at the START of each real stage --
    # "extracting" -> "chunking" -> "persisting" -> "embedding" ->
    # "flagging" -> "complete". NULL until the worker actually picks the
    # job up and reaches its first real stage (a genuine, honest gap --
    # Celery dispatch isn't instant -- not a bug to hide).
    #
    # On a FAILURE, this deliberately keeps whatever real stage it was
    # last set to -- worker/tasks.py's _mark_status() never overwrites it
    # when marking status="failed", specifically so a client can see how
    # far the pipeline genuinely got before dying (e.g. current_stage=
    # "chunking" + status="failed" means "died during chunking"), not a
    # generic dead end with the real progress erased. "failed" itself is
    # only ever written as a narrow fallback for the (practically
    # unreachable) case where the pipeline fails before ANY real stage
    # was recorded at all -- see _mark_status()'s own comment.
    current_stage = Column(String, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
