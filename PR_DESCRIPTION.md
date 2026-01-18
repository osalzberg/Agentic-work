This PR centralizes and normalizes KQL execution results across CLI and Web:

- Add canonical helpers in `utils.kql_exec`: `execute_kql_query`, `normalize_status`, `is_success`.
- Normalize `exec_stats['status']` to an upper-case string and preserve `raw_status` when SDK enums are present.
- Add `normalize_exec_result_for_http` in `my-first-mcp-server/rest_api.py` to ensure HTTP responses include JSON-friendly `exec_stats` and `ui_status`.
- Update `web_app.py` and `web_interface.py` to include `exec_stats` / `ui_status` in JSON responses and HTML outputs.
- Update `utils/batch_runner.py` to safely execute async agent generation.
- Add tests for exec result normalization.

This change improves parity between CLI and web artifacts, reduces brittleness around SDK enums, and prevents serialization errors when returning exec results over HTTP.
