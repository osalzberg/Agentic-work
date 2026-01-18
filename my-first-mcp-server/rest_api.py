import logging
from typing import Any, Dict, Optional

import utils.kql_exec as kql_exec
from utils.kql_exec import is_success
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok", "message": "Server is running"}


class KQLRequest(BaseModel):
    workspace_id: str
    query: str
    timespan: Optional[str] = None


class KQLResponse(BaseModel):
    tables: list = []
    error: Optional[str] = None


client = kql_exec.get_logs_client()
if client is None:
    logger.warning("Azure Monitor SDK not available or credential not configured; REST API will use fallback execution path")
# Note: `execute_kql_query` returns `exec_stats['status']` as a normalized
# upper-case string (e.g. "SUCCESS"). The SDK enum is available as
# `exec_stats['raw_status']` when present. Use `is_success(...)` helper.


@app.post("/query")
async def query(request: KQLRequest) -> KQLResponse:
    try:
        logger.info(f"Received query request for workspace: {request.workspace_id}")
        logger.info(f"Query: {request.query}")

        # Handle timespan - fix the logic
        from datetime import datetime, timedelta, timezone

        timespan = None
        if request.timespan:
            # If timespan is provided, use it as an ISO8601 duration
            # For now, we'll use a default timespan since the provided timespan logic was incorrect
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=1)
            timespan = (start_time, end_time)
            logger.info(f"Using timespan: {start_time} to {end_time}")

        # Execute query via canonical helper
        logger.info("Executing query via utils.kql_exec...")
        exec_result = kql_exec.execute_kql_query(kql=request.query, workspace_id=request.workspace_id, client=client, timespan=timespan)
        raw_tables = exec_result.get("tables", [])
        tables = []
        status = exec_result.get("exec_stats", {}).get("status")
        logger.info(f"Query executed with status: {status}")
        # Use canonical helper for success detection
        if is_success(status) or raw_tables:
            logger.info("Processing response tables...")
            for i, table in enumerate(raw_tables):
                try:
                    logger.info(f"Processing table {i}")
                    # Support both SDK table objects and plain dicts returned by our exec wrapper
                    if isinstance(table, dict):
                        cols_iter = table.get("columns", []) or []
                    else:
                        cols_iter = getattr(table, "columns", [])

                    columns = []
                    for col in cols_iter:
                        if hasattr(col, "name"):
                            columns.append(col.name)
                        elif isinstance(col, dict) and "name" in col:
                            columns.append(col["name"])
                        else:
                            columns.append(str(col))

                    # Process rows with better error handling
                    processed_rows = []
                    if isinstance(table, dict):
                        raw_rows = table.get("rows", []) or []
                    else:
                        raw_rows = getattr(table, "rows", [])

                    logger.info(f"Processing {len(raw_rows)} rows")

                    for row_idx, row in enumerate(raw_rows):
                        try:
                            processed_row = []
                            for cell in row:
                                # Convert any non-JSON-serializable types to strings
                                if cell is None:
                                    processed_row.append(None)
                                elif isinstance(cell, (str, int, float, bool)):
                                    processed_row.append(cell)
                                else:
                                    # Convert complex types to string
                                    processed_row.append(str(cell))
                            processed_rows.append(processed_row)
                        except Exception as row_error:
                            logger.error(f"Error processing row {row_idx}: {row_error}")
                            # Skip problematic rows
                            continue

                    name = table.get("name") if isinstance(table, dict) else getattr(table, "name", f"table_{i}")
                    table_dict = {
                        "name": name,
                        "columns": columns,
                        "rows": processed_rows,
                        "row_count": len(processed_rows),
                    }
                    tables.append(table_dict)
                    logger.info(
                        f"Successfully processed table {i} with {len(processed_rows)} rows"
                    )

                except Exception as table_error:
                    logger.error(f"Error processing table {i}: {table_error}")
                    # Continue with other tables
                    continue
            logger.info(f"Returning {len(tables)} tables")
            normalized = normalize_exec_result_for_http(exec_result, tables=tables)
            return KQLResponse(tables=normalized.get("tables", []), error=normalized.get("error"))
        else:
            error_msg = exec_result.get("exec_stats", {}).get("error", "Query failed")
            logger.error(f"Query failed: {error_msg}")
            normalized = normalize_exec_result_for_http(exec_result, tables=[], error=str(error_msg))
            return KQLResponse(error=normalized.get("error"))

    except Exception as e:
        logger.error(f"Exception in query handler: {str(e)}", exc_info=True)
        normalized = normalize_exec_result_for_http({}, tables=[], error=f"Server error: {str(e)}")
        return KQLResponse(error=normalized.get("error"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)


def normalize_exec_result_for_http(exec_result: Dict[str, Any], tables: Optional[list] = None, error: Optional[str] = None) -> Dict[str, Any]:
    """Normalize an exec_result into a JSON-serializable dict suitable for HTTP responses.

    Ensures `exec_stats.status` is a string and adds a `ui_status` field derived from exec_stats.
    Keeps `raw_status` if present but converts it to a string for safety.
    """
    tables = tables or []
    out: Dict[str, Any] = {}
    exec_stats = exec_result.get("exec_stats", {}) if isinstance(exec_result, dict) else {}

    # Normalize status to string
    status = exec_stats.get("status")
    raw_status = exec_stats.get("raw_status")
    if status is None and raw_status is not None:
        try:
            raw_s = str(raw_status)
            # If enum-like string includes a dot, take the last token (e.g., LogsQueryStatus.SUCCESS)
            if "." in raw_s:
                status = raw_s.split(".")[-1].upper()
            else:
                status = raw_s.upper()
        except Exception:
            status = "UNKNOWN"

    if isinstance(status, str):
        normalized_status = status.upper()
    else:
        normalized_status = "UNKNOWN"

    # Derive a simple ui_status for the frontend: 'success'|'failed'|'no_data'|'error'
    ui_status = "error"
    if normalized_status in ("SUCCESS", "SUCCEEDED"):
        ui_status = "success"
    elif normalized_status in ("NO_DATA", "NODATA"):
        ui_status = "no_data"
    elif normalized_status in ("FAILED", "FAILURE", "UNKNOWN"):
        ui_status = "failed"

    out["tables"] = tables
    out["exec_stats"] = {
        "status": normalized_status,
    }
    if raw_status is not None:
        try:
            out["exec_stats"]["raw_status"] = str(raw_status)
        except Exception:
            out["exec_stats"]["raw_status"] = "<unserializable>"

    out["exec_stats"]["ui_status"] = ui_status
    if error:
        out["error"] = str(error)
    else:
        out["error"] = exec_stats.get("error")

    return out
