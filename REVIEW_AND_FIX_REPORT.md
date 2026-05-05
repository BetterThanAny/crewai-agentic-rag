# Review and Fix Report

## Changes
- Restricted vector-search mock fallback to an explicit `VECTOR_SEARCH_ENABLE_MOCK_FALLBACK` opt-in; production failures now return a clear retrieval-unavailable message.
- Changed Chroma writes from `add()` to `upsert()` so repeated ingestion of the same chunk IDs is idempotent.
- Replaced Docker healthcheck `curl` usage with Python standard-library `urllib.request`.
- Updated and added tests for explicit mock fallback, default failure behavior, and vector-store upsert behavior.

## Verification
- `uv run pytest tests/test_tools.py tests/test_vector_store.py -q` passed.
- Worker also ran `uv run pytest tests/test_tools.py tests/test_e2e.py -q -m 'not slow'`, which passed.
- `git diff --check` passed.

## Remaining
- Full Docker healthcheck validation was not run in a live container.
