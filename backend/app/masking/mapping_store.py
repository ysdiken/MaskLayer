"""
Redis-backed bidirectional mapping store for PII placeholders.

Each masking job gets its own Redis Hash:
  Key:    mask:{job_id}
  Fields: placeholder → original_value
  TTL:    configurable (default 24 h)

Example keys after masking a document:
  mask:abc-123 → { "{TC_No_1}": "10000000146",
                   "{IBAN_1}":  "TR330006100519786457841326",
                   "{Email_1}": "ali@ornek.com" }

De-masking reads this hash and replaces placeholders in the LLM response
back with original values — entirely on-prem, never touching the network.

KVKK note: the TTL ensures PII is not stored indefinitely. The default
(86 400 s = 24 h) should be tuned to your compliance policy. Setting it to
a shorter window (e.g. 3 600 s) is safer for high-sensitivity documents.

Design:
- All methods are async (redis-py asyncio client).
- The store is stateless; pass the Redis client in at construction time so
  it can be mocked cleanly in tests (fakeredis.aioredis).
- unmask_text() does a single HGETALL then replaces in one pass — O(n) on
  number of placeholders, not on text length.
"""

import re
from redis.asyncio import Redis


class MappingStore:
    def __init__(self, redis: Redis, ttl_seconds: int = 86_400) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    # ── Write ─────────────────────────────────────────────────────────────────

    async def save_mapping(
        self,
        job_id: str,
        mapping: dict[str, str],
    ) -> None:
        """
        Persist a placeholder→original mapping for a job.

        mapping = { "{TC_No_1}": "10000000146", "{Email_1}": "a@b.com", ... }

        Uses a single pipeline (HSET + EXPIRE) so both commands land in one
        round-trip to Redis.
        """
        if not mapping:
            return

        key = _job_key(job_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            await pipe.hset(key, mapping=mapping)   # type: ignore[arg-type]
            await pipe.expire(key, self._ttl)
            await pipe.execute()

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get_original(self, job_id: str, placeholder: str) -> str | None:
        """Return the original value for a single placeholder, or None if expired/missing."""
        value = await self._redis.hget(_job_key(job_id), placeholder)
        return value.decode() if value else None

    async def get_all(self, job_id: str) -> dict[str, str]:
        """Return the full placeholder→original map for a job (empty dict if missing)."""
        raw: dict[bytes, bytes] = await self._redis.hgetall(_job_key(job_id))
        return {k.decode(): v.decode() for k, v in raw.items()}

    # ── De-masking ────────────────────────────────────────────────────────────

    async def unmask_text(self, job_id: str, masked_text: str) -> str:
        """
        Replace all placeholders in masked_text with their original values.

        Fetches the entire mapping in one HGETALL, then uses a single regex
        substitution pass — efficient regardless of placeholder count.

        Returns the original text if all placeholders are found; unknown
        placeholders are left as-is (defensive: LLM may have altered them).
        """
        mapping = await self.get_all(job_id)
        if not mapping:
            return masked_text

        # Build a regex that matches any known placeholder literally.
        # re.escape handles the curly braces and underscores safely.
        pattern = re.compile(
            "|".join(re.escape(ph) for ph in sorted(mapping, key=len, reverse=True))
        )

        def replace(match: re.Match) -> str:
            return mapping.get(match.group(0), match.group(0))

        return pattern.sub(replace, masked_text)

    # ── Housekeeping ──────────────────────────────────────────────────────────

    async def delete(self, job_id: str) -> None:
        """Explicitly delete a job's mapping (e.g. after analyst confirms de-masking)."""
        await self._redis.delete(_job_key(job_id))

    async def ttl(self, job_id: str) -> int:
        """Return remaining TTL in seconds (-2 = key does not exist, -1 = no expiry)."""
        return await self._redis.ttl(_job_key(job_id))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _job_key(job_id: str) -> str:
    return f"mask:{job_id}"
