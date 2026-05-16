# LCM context-engine spike handbook

Status: opt-in spike only. The built-in default context engine remains `compressor` unless a profile explicitly sets `context.engine: lcm`.

## What shipped

- Engine package: `plugins/context_engine/lcm/`
- Engine name: `lcm`
- Store: profile-local SQLite at `${HERMES_HOME}/context/lcm.sqlite3`
- Recall tools exposed only when the engine is active:
  - `lcm_grep` — keyword search over the active session transcript
  - `lcm_describe` — summary/checkpoint or non-flagged matching context

## Safe activation for a disposable profile

Use only a non-critical test profile during the spike.

```bash
export HERMES_HOME="$HOME/.hermes/profiles/hawk-lcm"
mkdir -p "$HERMES_HOME"
hermes config set context.engine lcm
```

Do not enable `lcm` on the default, Hawk, Blitz, or production profiles until the spike has run cleanly and rollback has been verified.

## Validation checklist

Run from the Hermes repo:

```bash
python -m pytest tests/plugins/test_lcm_context_engine.py \
  tests/agent/test_context_engine.py \
  tests/run_agent/test_plugin_context_engine_init.py \
  -q -o 'addopts='
python -m ruff check plugins/context_engine/lcm tests/plugins/test_lcm_context_engine.py
python -m ty check plugins/context_engine/lcm tests/plugins/test_lcm_context_engine.py
```

Expected gates:

- Plugin discovery returns `lcm` as available.
- Transcript persistence writes only under the active profile's `context/` directory.
- Secret-like values are redacted before persistence.
- Prompt-injection-like turns are flagged and excluded from normal `lcm_describe` recall.
- Restart recovery can search the same profile DB.
- A second profile cannot see the first profile's transcript.

## Monitoring during spike

Inspect the disposable profile only:

```bash
du -h "$HERMES_HOME/context/lcm.sqlite3" "$HERMES_HOME/context/lcm.sqlite3-wal" 2>/dev/null || true
sqlite3 "$HERMES_HOME/context/lcm.sqlite3" \
  "select count(*) from context_items; select count(*) from context_items where secret_redacted=1; select count(*) from context_items where injection_flag=1;"
```

Watch for:

- Unexpected DB growth or WAL growth.
- False-negative redaction findings in stored rows.
- Prompt-injection rows appearing in normal recall output.
- Any write outside `${HERMES_HOME}/context/`.

## Rollback

For a disposable profile:

```bash
export HERMES_HOME="$HOME/.hermes/profiles/hawk-lcm"
hermes config set context.engine compressor
rm -f "$HERMES_HOME/context/lcm.sqlite3" "$HERMES_HOME/context/lcm.sqlite3-shm" "$HERMES_HOME/context/lcm.sqlite3-wal"
```

For production profiles, rollback should normally be a config-only revert to `compressor`; delete DB files only after confirming no investigation needs the spike data.

## Incident response

- Secret redaction miss: stop the spike profile, preserve only redacted evidence, rotate through the approved secure settings path if the credential is live, then delete the affected disposable DB.
- Prompt-injection recall leak: stop the spike profile, capture the offending query/output, and patch `plugins/context_engine/lcm/redaction.py` before re-enabling.
- Cross-profile contamination: stop immediately and verify `HERMES_HOME`; no production profile should have `context.engine: lcm` during the spike.
