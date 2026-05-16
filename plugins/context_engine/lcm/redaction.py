"""Safety helpers for LCM persistence and recall."""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.redact import redact_sensitive_text

# Hermes' central redactor catches many common credentials.  Add a few
# LCM-specific/high-signal patterns here so raw transcript persistence has a
# deterministic safety gate even when no LLM classifier is available.
_EXTRA_SECRET_PATTERNS = [
    re.compile(r"\bshpat_[A-Za-z0-9_\-]{8,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
]

_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I), "instruction override"),
    (re.compile(r"exfiltrate|dump\s+(the\s+)?(database|schema|credentials|secrets)", re.I), "exfiltration request"),
    (re.compile(r"reveal\s+(your\s+)?(system\s+prompt|hidden\s+instructions)", re.I), "system prompt extraction"),
]


@dataclass(frozen=True)
class SafetyScan:
    content: str
    secret_redacted: bool
    injection_flag: bool
    injection_reason: str | None = None


def sanitize_for_storage(text: str) -> SafetyScan:
    """Redact secrets and classify prompt-injection text before storage."""

    original = text or ""
    redacted = redact_sensitive_text(original)
    for pattern in _EXTRA_SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    secret_redacted = redacted != original

    injection_reason = None
    for pattern, reason in _INJECTION_PATTERNS:
        if pattern.search(original):
            injection_reason = reason
            break

    return SafetyScan(
        content=redacted,
        secret_redacted=secret_redacted,
        injection_flag=injection_reason is not None,
        injection_reason=injection_reason,
    )
