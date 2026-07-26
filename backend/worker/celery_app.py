"""M15: Celery app configuration. Redis is both the broker and the result
backend -- there's no reason to run a second kind of infrastructure for
this milestone's scope (task result persistence beyond Contract.status
is explicitly out of scope), so one Redis instance does both jobs.

REDIS_URL is read directly via os.getenv, the same pattern already used
for VOYAGE_API_KEY/GROQ_API_KEY (config.py's central Settings object is
for the FastAPI app's own settings; this module runs standalone in the
worker process too, so it manages its own env var directly rather than
depending on config.py).
"""

import os

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery(
    "clauseguard",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["worker.tasks"],
)

celery_app.conf.update(
    task_track_started=True,
)
