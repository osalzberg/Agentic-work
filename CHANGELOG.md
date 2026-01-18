# Changelog

## Unreleased

- Normalize KQL execution status to canonical `exec_stats.status` string.
- Add `is_success()` helper to check status across enum/string values.
- Add HTTP normalization adapter for REST responses to ensure JSON-serializable `exec_stats` and `ui_status`.
- Ensure batch runner handles async agent query generation (avoids coroutine objects in results).
- Include `exec_stats` and `ui_status` in key JSON endpoints for UI parity.
- Add developer docs for canonical `exec_stats` contract.
