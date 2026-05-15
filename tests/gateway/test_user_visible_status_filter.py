import sys
import types


fire = types.ModuleType("fire")
setattr(fire, "Fire", lambda *a, **k: None)
firecrawl = types.ModuleType("firecrawl")
setattr(firecrawl, "Firecrawl", object)
fal_client = types.ModuleType("fal_client")

sys.modules.setdefault("fire", fire)
sys.modules.setdefault("firecrawl", firecrawl)
sys.modules.setdefault("fal_client", fal_client)

from gateway.run import _should_suppress_user_visible_status


def test_suppresses_internal_compression_statuses_from_chat_delivery():
    suppressed = [
        ("lifecycle", "📦 Preflight compression: ~118,493 tokens >= 95,200 threshold. This may take a moment."),
        ("lifecycle", "🗜️ Compacting context — summarizing earlier conversation so I can continue..."),
        ("warn", "⚠ Compression summary failed: Request timed out.. Inserted a fallback context marker."),
    ]

    for event_type, message in suppressed:
        assert _should_suppress_user_visible_status(event_type, message) is True


def test_keeps_final_and_actionable_statuses_user_visible():
    assert _should_suppress_user_visible_status("final", "Done") is False
    assert _should_suppress_user_visible_status("warn", "⚠️ Provider authentication failed: login required") is False
    assert _should_suppress_user_visible_status("lifecycle", "❌ Connection to provider failed after 3 attempts.") is False
