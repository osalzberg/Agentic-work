import time
from typing import Any, Dict, Optional, Tuple

try:  # Optional Azure SDK imports
    from azure.identity import DefaultAzureCredential  # type: ignore
    from azure.monitor.query import LogsQueryClient  # type: ignore
except Exception:  # pragma: no cover - optional during docs-only use
    DefaultAzureCredential = None  # type: ignore
    LogsQueryClient = None  # type: ignore


def get_logs_client(
    credential: Optional[Any] = None, endpoint: Optional[str] = None
) -> Optional[LogsQueryClient]:
    """Return a `LogsQueryClient` using the provided credential or a default.

    This helper centralizes client creation so call-sites don't instantiate
    `LogsQueryClient` directly. If the Azure SDK isn't available this returns
    `None` and callers should fallback to their test/mocked behavior.
    """
    if LogsQueryClient is None:
        return None
    try:
        if credential is None and DefaultAzureCredential is not None:
            credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
        if credential is None:
            return None
        if endpoint:
            return LogsQueryClient(endpoint, credential)  # type: ignore[arg-type]
        return LogsQueryClient(credential)
    except Exception:
        return None


def execute_kql_query(
    kql: str,
    workspace_id: Optional[str] = None,
    client: Optional[Any] = None,
    credential: Optional[Any] = None,
    endpoint: Optional[str] = None,
    timespan: Optional[Tuple[Any, Any]] = None,
) -> Dict[str, Any]:
    """Execute a KQL query and return structured results and exec metadata.

    Callers may pass an already-created `client` (preferred). If no client is
    provided, this will attempt to create one via `get_logs_client(credential)`.
    Returns a dict with keys: `tables`, `returned_rows_count`, `exec_stats`.
    """
    start = time.time()

    # Prefer injected client
    logs_client = client
    if logs_client is None:
        logs_client = get_logs_client(credential=credential, endpoint=endpoint)

    # If a real client is available, call the Azure SDK
    if logs_client is not None:
        try:
            resp = logs_client.query_workspace(
                workspace_id=workspace_id, query=kql, timespan=timespan
            )
            tables = []
            if hasattr(resp, "tables") and resp.tables:
                for i, table in enumerate(resp.tables):
                    cols = [getattr(c, "name", str(c)) for c in getattr(table, "columns", [])]
                    rows = [list(r) for r in getattr(table, "rows", [])]
                    tables.append({"name": getattr(table, "name", f"table_{i}"), "columns": cols, "rows": rows, "row_count": len(rows)})
            # Normalize status for cross-process compatibility
            raw_status = getattr(resp, "status", None)
            exec_stats = {"status": (raw_status.name if hasattr(raw_status, "name") else str(raw_status)) if raw_status is not None else "UNKNOWN", "raw_status": raw_status}
            elapsed = time.time() - start
            exec_stats["elapsed_sec"] = elapsed
            return {"tables": tables, "returned_rows_count": sum(t.get("row_count", 0) for t in tables), "exec_stats": exec_stats}
        except Exception as e:
            elapsed = time.time() - start
            return {"tables": [], "returned_rows_count": 0, "exec_stats": {"error": str(e), "elapsed_sec": elapsed}}

    # Fallback behavior when no SDK client available: try to delegate to local kql_client
    try:
        from kql_client import execute_query  # type: ignore

        result = execute_query(kql, connection={"workspace_id": workspace_id, "timespan": timespan})
        elapsed = time.time() - start
        exec_stats = result.get("exec_stats", {}) if isinstance(result, dict) else {}
        # Normalize status if present
        if "status" in exec_stats and exec_stats.get("status") is not None:
            s = exec_stats.get("status")
            exec_stats["raw_status"] = s
            exec_stats["status"] = (s.name if hasattr(s, "name") else str(s)).upper()
        exec_stats["elapsed_sec"] = elapsed
        return {"tables": result.get("tables", []), "returned_rows_count": len(result.get("tables", [])), "exec_stats": exec_stats}
    except Exception:
        # Minimal simulated fallback
        elapsed = time.time() - start
        return {"tables": [], "returned_rows_count": 0, "exec_stats": {"simulated": True, "elapsed_sec": elapsed}}


def normalize_status(status: Any) -> Optional[str]:
    """Normalize an SDK enum or string status to an upper-case string, or None."""
    if status is None:
        return None
    try:
        if hasattr(status, "name"):
            return status.name.upper()
    except Exception:
        pass
    try:
        return str(status).upper()
    except Exception:
        return None


def is_success(status: Any) -> bool:
    """Return True if the provided status (enum or string) represents success."""
    n = normalize_status(status)
    return n == "SUCCESS"


# Compatibility note:
# - `execute_kql_query` returns `exec_stats` where `exec_stats['status']` is a
#   normalized upper-case string (e.g. "SUCCESS", "FAILURE", "UNKNOWN").
# - When an SDK response is available we also include `exec_stats['raw_status']`
#   containing the original SDK enum object. Callers that only need a simple
#   success/failure check should use `is_success(exec_stats.get('status') or exec_stats.get('raw_status'))`.


"""
Developer docs: canonical `exec_stats` shape

The canonical `exec_stats` contract used across the codebase is a small
JSON-friendly dictionary attached to KQL execution results. Callers should
rely on the normalized `status` string or `is_success()` helper for success
checks; `raw_status` is present when the Azure SDK returns an enum and is
kept stringified at HTTP boundaries for debugging.

Canonical shape example:

        {
                "status": "SUCCESS",          # normalized upper-case string
                "raw_status": "LogsQueryStatus.SUCCESS",  # optional original enum (stringified)
                "elapsed_sec": 0.123,
                "error": None,                 # optional error message when present
                "ui_status": "success"       # derived for UI convenience (success|failed|no_data|error)
        }

Guidelines:
- Prefer `is_success(exec_stats.get('status') or exec_stats.get('raw_status'))`
    rather than comparing SDK enums or raw strings.
- At HTTP boundaries, ensure `exec_stats` values are JSON-serializable; use
    `str()` to convert SDK enums to strings when needed.
- `ui_status` is provided to simplify UI logic; it is derived from `status`.

Additions to this file should maintain backwards-compatibility with these
expectations.
"""

