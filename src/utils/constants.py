from uuid import uuid4

DEFAULT_TTL = 300  # seconds for cache expiry
MAX_RETRIES = 5
BACKOFF_FACTOR = 0.5


def new_corr_id() -> str:
    """Return a new correlation ID."""
    return uuid4().hex
