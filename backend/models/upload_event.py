"""M36-FIX (post-M37 holistic review): a minimal, append-only record of
"a genuinely-accepted upload happened for this user" -- deliberately
NOT a foreign key to contracts.id, and deliberately its own table rather
than reusing AuditLog.

WHY THIS EXISTS: M36's daily rate limit originally counted live Contract
rows. M37 then gave users a real DELETE endpoint that permanently removes
Contract rows. Composed together, this was a real, confirmed exploit: a
user could upload a contract (spending real Groq/Voyage quota the
moment the pipeline ran), immediately delete it, and free up their daily
upload slot for another real upload -- with M37's own audit_log CASCADE
(models/audit_log.py) erasing all trace of the original upload too,
leaving zero record to even detect the pattern. Contract rows and
audit_log rows both represent CURRENT/historical PROCESSING state, which
is exactly why M37 correctly cascades them on deletion -- but that same
correctness is what made them the wrong thing for a rate limiter to
count. A rate limit needs to answer "did this user cause a real
upload-triggering event today", a question that must stay true forever
once it's true, regardless of what the user does to the resulting
contract afterward.

WHY NOT A FOREIGN KEY TO contracts.id: the entire point of this table is
to survive a Contract's deletion. A FK to contracts.id would force the
same choice M37 already had to make for audit_log.contract_id (cascade
or null it) -- cascading would recreate the exact bug this table exists
to fix; nulling would work but adds a column this table has no use for
(unlike audit_log, nothing here is ever displayed per-contract). Simplest
and most honest: this table has no relationship to contracts at all. It
only asserts "user_id caused an accepted upload at created_at" -- a fact
that is true independent of whatever happens to any Contract row
afterward.

user_id DOES carry a real FK to users.id (mirroring Contract.user_id's
own established convention, models/contract.py) -- unlike contracts,
users are never deleted anywhere in this codebase, so this FK carries
none of the cascade risk a contracts.id FK would.

INDEXED ON user_id (not just declared, verified in the real schema -- see
this milestone's own test script) -- the exact same "an audit-style table
with no index becomes unusable at scale" risk M35's audit_log was built
to avoid from day one; every real query against this table filters by
user_id (see middleware/rate_limit.py), so this index is load-bearing,
not decorative.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID

from db import Base


class UploadEvent(Base):
    __tablename__ = "upload_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
