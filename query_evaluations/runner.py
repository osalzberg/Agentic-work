from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict

from eval.dataset import load_dataset
from eval.canonicalizer import diff_summary
from eval.comparator import load_csv_snapshot, compare_schema, compare_rows
from eval.scorer import score_test, DEFAULT_WEIGHTS
from eval.reporter import write_json
from eval.explainer import grade_kql_similarity
from eval.kql_parser import compare_kql_semantic

# Optional imports for execution
try:
    from azure.identity import DefaultAzureCredential  # type: ignore
    from azure.monitor.query import LogsQueryClient  # type: ignore
    from azure.monitor.query import LogsQueryStatus  # type: ignore
except Exception:
    DefaultAzureCredential = None  # type: ignore
    LogsQueryClient = None  # type: ignore
    LogsQueryStatus = None  # type: ignore

# Prompt-to-KQL
from azure_openai_utils import run_chat, load_config  # type: ignore


# System prompt used for KQL generation
SYSTEM_PROMPT = (
    "Translate the user prompt to KQL (Kusto Query Language) for Azure Log Analytics workspace. "
    "The queries are about AKS and containers. "
    "IMPORTANT: Use ContainerLogV2 instead of ContainerLog for container log queries. "
    "Return ONLY the KQL query code with no explanations, comments, or additional text. "
    "Output should be executable KQL that can be run directly in Log Analytics."
)

def _simple_translate_nl_to_kql(user_prompt: str) -> str:
    """Simple translation using direct model call with minimal system prompt."""
    system_prompt = SYSTEM_PROMPT
    
    result = run_chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        purpose="nl_to_kql_eval",
        max_tokens=2000,
        temperature=0.0
    )
    
    if result.error or not result.content:
        return f"// Error: Could not translate question to KQL: {user_prompt}\n// Error details: {result.error or 'Empty response'}"
    
    return result.content.strip()


def _extract_innermost_error_message(error_text: str) -> str:
    """Extract the innermost error message from nested Azure error responses."""
    if not error_text:
        return error_text
    
    import json
    import re
    
    # Try to find JSON-like inner error structure
    inner_error_match = re.search(r'Inner error:\s*(\{.*\})', error_text, re.DOTALL)
    if not inner_error_match:
        return error_text
    
    try:
        inner_error_json = inner_error_match.group(1)
        error_obj = json.loads(inner_error_json)
        
        # Traverse nested innererror objects to find the deepest message
        current = error_obj
        last_message = error_text
        
        while current:
            if 'message' in current:
                last_message = current['message']
            
            # Check for nested innererror
            if 'innererror' in current and isinstance(current['innererror'], dict):
                current = current['innererror']
            else:
                break
        
        return last_message
    except (json.JSONDecodeError, KeyError, TypeError):
        # If parsing fails, return original
        return error_text


def _determine_timespan(kql: str, start_iso: str = "", end_iso: str = ""):
    """Decide timespan to pass to Azure Monitor.

    - If start/end provided, return (end-start) duration.
    - If query contains ago(Xd/h/m), extract and use that duration to ensure SDK queries correct range.
    - If query contains TimeGenerated with datetime comparisons, extract the time range.
    - Otherwise, return None.
    """
    from datetime import datetime, timezone, timedelta
    import re

    if start_iso and end_iso:
        start = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
        end = datetime.fromisoformat(end_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
        return end - start

    s = (kql or "").lower()
    orig_kql = kql or ""
    
    # Extract ago() duration if present (e.g., ago(30d), ago(7h), ago(90m))
    ago_match = re.search(r'ago\s*\(\s*(\d+)\s*([dhm])\s*\)', s)
    if ago_match:
        value = int(ago_match.group(1))
        unit = ago_match.group(2)
        if unit == 'd':
            return timedelta(days=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'm':
            return timedelta(minutes=value)
    
    # Extract datetime values from TimeGenerated filters
    # Patterns: TimeGenerated >= datetime(...), TimeGenerated between (datetime(...) .. datetime(...))
    datetime_pattern = r'datetime\s*\(\s*["\']?([0-9]{4}-[0-9]{2}-[0-9]{2}(?:T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})?)?)["\']?\s*\)'
    
    # Find all datetime values in the query (case-insensitive)
    datetime_matches = re.findall(datetime_pattern, orig_kql, re.IGNORECASE)
    
    if datetime_matches:
        try:
            # Parse all datetime values
            parsed_dates = []
            for dt_str in datetime_matches:
                # Handle various formats
                dt_str_clean = dt_str.replace('Z', '+00:00')
                if 'T' not in dt_str_clean:
                    dt_str_clean += 'T00:00:00+00:00'
                elif '+' not in dt_str_clean and '-' not in dt_str_clean[-6:]:
                    dt_str_clean += '+00:00'
                dt = datetime.fromisoformat(dt_str_clean).astimezone(timezone.utc)
                parsed_dates.append(dt)
            
            if len(parsed_dates) >= 2:
                # Multiple datetime values - calculate range between min and max
                min_date = min(parsed_dates)
                max_date = max(parsed_dates)
                return max_date - min_date
            elif len(parsed_dates) == 1:
                # Single datetime value - calculate from that date to now
                dt = parsed_dates[0]
                now = datetime.now(timezone.utc)
                # If TimeGenerated >= datetime, range is from datetime to now
                # If TimeGenerated <= datetime, range is from datetime to past (use datetime as reference)
                if '>=' in orig_kql or '>' in orig_kql:
                    return now - dt
                else:
                    # For <= or <, use the datetime as the end, go back reasonable amount (30 days default)
                    return timedelta(days=30)
        except Exception:
            # If parsing fails, fall back to None
            pass
    
    # No extractable time range, return None
    return None


def _run_kql(workspace_id: str, kql: str, start_iso: str = "", end_iso: str = "", fallback_timespan_hours: int = 0) -> Dict[str, Any]:
    if LogsQueryClient is None:
        return {"error": "Azure Monitor SDK not installed", "rows": [], "columns": [], "statistics": None}
    try:
        cred = DefaultAzureCredential() if DefaultAzureCredential else None
        client = LogsQueryClient(credential=cred) if cred else None
        if client is None:
            return {"error": "No Azure credential", "rows": [], "columns": [], "statistics": None}
        timespan = _determine_timespan(kql, start_iso, end_iso)
        
        # If fallback_timespan_hours is provided and no timespan determined, use fallback
        if timespan is None and fallback_timespan_hours > 0:
            from datetime import timedelta
            timespan = timedelta(hours=fallback_timespan_hours)
        
        # Ask SDK to include execution statistics if available. If timespan is None, prefer omitting it entirely.
        try:
            if timespan is None:
                try:
                    resp = client.query_workspace(workspace_id, kql, include_statistics=True)
                except TypeError:
                    resp = client.query_workspace(workspace_id, kql)
            else:
                try:
                    resp = client.query_workspace(workspace_id, kql, timespan=timespan, include_statistics=True)
                except TypeError:
                    # Older SDKs may not support include_statistics
                    resp = client.query_workspace(workspace_id, kql, timespan=timespan)
        except TypeError as e:
            # Some SDK versions may require timespan explicitly; if not already using fallback, retry with it
            if fallback_timespan_hours == 0:
                return {"error": f"SDK requires timespan; retrying with fallback. ({e})", "rows": [], "columns": [], "statistics": None, "retry_with_timespan": True}
            # Already tried with fallback, still failed
            return {"error": f"SDK requires timespan even with fallback. ({e})", "rows": [], "columns": [], "statistics": None}
        # Status can be an enum or a plain string depending on SDK version
        status_val = getattr(resp, "status", None)
        status_name = getattr(status_val, "name", None) if status_val is not None else None
        if status_name and status_name.upper() not in ("SUCCESS", "SUCCEEDED"):
            return {"error": f"Query failed: {status_name}", "rows": [], "columns": [], "statistics": None}
        if isinstance(status_val, str) and status_val.upper() not in ("SUCCESS", "SUCCEEDED"):
            return {"error": f"Query failed: {status_val}", "rows": [], "columns": [], "statistics": None}
        tables = getattr(resp, "tables", [])
        # Try to pull statistics if present (shape varies by SDK)
        statistics = None
        try:
            statistics = getattr(resp, "statistics", None)
        except Exception:
            statistics = None
        if not tables:
            return {"error": None, "rows": [], "columns": [], "statistics": statistics}
        # Use first table only for equality checks
        tbl = tables[0]
        # Columns can be SDK objects, dicts, or plain strings depending on SDK version
        cols = []
        for col in getattr(tbl, "columns", []):
            name = None
            try:
                name = getattr(col, "name")  # SDK object with .name
            except Exception:
                name = None
            if name:
                cols.append(name)
                continue
            if isinstance(col, dict) and "name" in col:
                cols.append(str(col["name"]))
                continue
            # Fallback: stringify whatever we received
            cols.append(str(col))

        # Rows: ensure we stringify cells and handle None
        rows = []
        for row in getattr(tbl, "rows", []):
            rows.append(["" if cell is None else str(cell) for cell in row])
        return {"error": None, "rows": rows, "columns": cols, "statistics": statistics}
    except Exception as e:
        return {"error": str(e), "rows": [], "columns": [], "statistics": None}


def _normalize_kql_for_display(kql: str) -> str:
    """Normalize KQL for display in reports.

    - Remove meta comment line(s) starting with "// meta"
    - Collapse all whitespace (including newlines) to single spaces
    - Lowercase KQL keywords only (identifiers remain as-is)
    - Add single spaces around pipes, commas, and operators for readability
    - Trim leading/trailing spaces
    """
    import re
    s = kql.replace("\r\n", "\n").replace("\r", "\n")
    lines = s.split("\n")
    lines = [ln for ln in lines if not re.match(r"^\s*//\s*meta", ln, flags=re.IGNORECASE)]
    joined = " ".join(lines)
    joined = re.sub(r"\s+", " ", joined).strip()
    # Lowercase KQL keywords only
    keywords = [
        "where", "project", "summarize", "extend", "join", "order by", "take", "top",
        "union", "let", "parse", "parse_json", "make-series", "render", "bin", "by"
    ]
    kw_re = re.compile(r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b", re.IGNORECASE)
    joined = kw_re.sub(lambda m: m.group(0).lower(), joined)
    # Normalize with single spaces for readability
    # Process longer operators first to avoid matching substrings
    joined = re.sub(r"\s*,\s*", ", ", joined)
    joined = re.sub(r"\s*\|\s*", " | ", joined)
    joined = re.sub(r"\s*==\s*", " == ", joined)
    joined = re.sub(r"\s*!=\s*", " != ", joined)
    joined = re.sub(r"\s*<=\s*", " <= ", joined)
    joined = re.sub(r"\s*>=\s*", " >= ", joined)
    # Use negative lookahead/lookbehind to avoid re-processing compound operators
    joined = re.sub(r"(?<![=!<>])\s*=\s*(?!=)", " = ", joined)  # = but not part of ==, !=, <=, >=
    joined = re.sub(r"(?<!<)\s*<\s*(?!=)", " < ", joined)  # < but not part of <=
    joined = re.sub(r"(?<!>)\s*>\s*(?!=)", " > ", joined)  # > but not part of >=
    # Collapse any multiple spaces that might have been created
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined


def _normalize_kql_for_report(kql: str) -> str:
    """Normalize KQL for structural comparison.

    - Remove meta comment line(s) starting with "// meta"
    - Collapse all whitespace (including newlines) to single spaces
    - Lowercase KQL keywords only (identifiers remain as-is)
    - Normalize quotes: convert single quotes to double quotes
    - Remove all spaces around commas, pipes, and operators for comparison
    - Trim leading/trailing spaces
    """
    import re
    s = kql.replace("\r\n", "\n").replace("\r", "\n")
    lines = s.split("\n")
    lines = [ln for ln in lines if not re.match(r"^\s*//\s*meta", ln, flags=re.IGNORECASE)]
    joined = " ".join(lines)
    joined = re.sub(r"\s+", " ", joined).strip()
    
    # Normalize quotes: convert single quotes to double quotes
    joined = joined.replace("'", '"')
    
    # Lowercase KQL keywords only
    keywords = [
        "where", "project", "summarize", "extend", "join", "order by", "take", "top",
        "union", "let", "parse", "parse_json", "make-series", "render", "bin", "by"
    ]
    kw_re = re.compile(r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b", re.IGNORECASE)
    joined = kw_re.sub(lambda m: m.group(0).lower(), joined)
    # Remove "desc" since it's the default ordering mode (asc is explicit, desc is implicit)
    joined = re.sub(r"\s+desc\b", "", joined, flags=re.IGNORECASE)
    # Normalize commas, pipes, and operators (remove spaces around them for consistency)
    joined = re.sub(r"\s*,\s*", ",", joined)
    joined = re.sub(r"\s*\|\s*", "|", joined)
    joined = re.sub(r"\s*=\s*", "=", joined)
    joined = re.sub(r"\s*==\s*", "==", joined)
    joined = re.sub(r"\s*!=\s*", "!=", joined)
    joined = re.sub(r"\s*<=\s*", "<=", joined)
    joined = re.sub(r"\s*>=\s*", ">=", joined)
    joined = re.sub(r"\s*<\s*", "<", joined)
    joined = re.sub(r"\s*>\s*", ">", joined)
    return joined


def _is_generation_error(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    markers = [
        "// Error",
        "Error:",
        "Error translating NL to KQL",
        "Could not translate question to KQL",
    ]
    return any(m in s for m in markers)


def main() -> None:
    parser = argparse.ArgumentParser(description="NL2KQL Evaluation Runner (see docs/nl2kql_evaluation_prd.md)")
    parser.add_argument("--dataset", required=True, help="Path to JSONL/CSV dataset")
    parser.add_argument("--out", required=True, help="Output report folder")
    parser.add_argument("--workspace", default=os.getenv("WORKSPACE_ID", ""), help="Workspace ID (overrides dataset)")
    # Removed --start/--end; time range should come from KQL itself
    args = parser.parse_args()

    items = load_dataset(args.dataset)
    results = []
    start_ts = time.time()

    for it in items:
        workspace_id = args.workspace or it.workspace_id or ""
        # Do not inject a time range; use only what's in KQL
        start_iso = ""
        end_iso = ""

        gen_kql = _simple_translate_nl_to_kql(it.prompt)

        # If generation failed, record error and skip execution
        if _is_generation_error(gen_kql):
            results.append({
                "id": it.id,
                "prompt": it.prompt,
                "generated_kql": "",
                "expected_kql": it.expected_kql,
                "score": "NA",
                "metrics": {
                    "result_equality": "NA",
                    "exec_success": "NA",
                    "structural_similarity": "NA",
                    "cost_efficiency": "NA",
                    "robustness": "NA",
                    "schema_match_score": "NA",
                    "rows_match_score": "NA",
                },
                "struct_diff": None,
                "schema_match": {"na": True},
                "rows_match": {"na": True},
                "exec_error": None,
                "query_generation_error": gen_kql.strip(),
                "has_results": False,
                "gen_statistics": None,
                "exp_statistics": None,
            })
            continue

        # Perform structural comparison BEFORE execution for optimization
        rep_gen = _normalize_kql_for_report(gen_kql)
        rep_exp = _normalize_kql_for_report(it.expected_kql)
        
        # Optimization: if normalized KQL are identical (structural similarity = 1.0), skip execution and return perfect score
        if rep_exp == rep_gen:
            results.append({
                "id": it.id,
                "prompt": it.prompt,
                "generated_kql": _normalize_kql_for_display(gen_kql),
                "expected_kql": it.expected_kql,
                "score": 1.0,
                "metrics": {
                    "result_equality": 1.0,
                    "exec_success": 1.0,
                    "structural_similarity": 1.0,
                    "schema_match_score": 1.0,
                    "rows_match_score": 1.0,
                    "llm_graded_similarity": 1.0,
                },
                "struct_diff": {"perfect_match": True},
                "schema_match": {"identical": True},
                "rows_match": {"identical": True},
                "exec_error": None,
                "has_results": True,
                "gen_statistics": None,
                "llm_graded_similarity_score": 1.0,
            })
            continue

        # Execute generated KQL first; if execution fails, mark evaluation as NA and skip further checks
        exec_out = {"error": "workspace missing", "rows": [], "columns": []}
        if workspace_id:
            exec_out = _run_kql(workspace_id, gen_kql, start_iso, end_iso)
            
            # If execution failed due to missing timespan, retry with 24-hour fallback
            if exec_out.get("retry_with_timespan"):
                exec_out = _run_kql(workspace_id, gen_kql, start_iso, end_iso, fallback_timespan_hours=24)

        if exec_out.get("error"):
            # Determine if this is an authentication error
            exec_error = _extract_innermost_error_message(exec_out.get("error", ""))
            is_auth_error = any(keyword in exec_error.lower() for keyword in [
                "authentication", "credential", "defaultazurecredential", "failed to retrieve a token",
                "401", "unauthorized", "access denied"
            ])
            
            # Determine critical issue category and details
            if is_auth_error:
                critical_issue = f"authentication ({exec_error})"
            else:
                critical_issue = f"Invalid syntax ({exec_error})"
            
            # Short-circuit: do not perform structural or result comparisons
            # Score is NA for auth errors, 0 for other execution failures
            # Schema/rows match are NA for all execution failures
            results.append({
                "id": it.id,
                "prompt": it.prompt,
                "generated_kql": _normalize_kql_for_display(gen_kql),
                "expected_kql": it.expected_kql,
                "score": "NA" if is_auth_error else 0.0,
                "metrics": {
                    "result_equality": "NA" if is_auth_error else 0.0,
                    "exec_success": "NA" if is_auth_error else 0.0,
                    "structural_similarity": "NA" if is_auth_error else 0.0,
                    "cost_efficiency": "NA",
                    "robustness": "NA",
                    "schema_match_score": "NA",
                    "rows_match_score": "NA",
                },
                "struct_diff": None,
                "schema_match": {"na": True},
                "rows_match": {"na": True},
                "exec_error": exec_error,
                "critical_issue": critical_issue,
                "has_results": False,
                "gen_statistics": exec_out.get("statistics"),
                "exp_statistics": None,
            })
            continue

        # Calculate structural similarity using semantic KQL comparison
        # Normalize queries before comparison to eliminate whitespace differences
        normalized_expected = _normalize_kql_for_report(it.expected_kql)
        normalized_generated = _normalize_kql_for_report(gen_kql)
        semantic_comparison = compare_kql_semantic(normalized_expected, normalized_generated)
        structural_similarity = semantic_comparison['similarity']
        struct_diff = semantic_comparison['details']

        # If different tables are used, score is 0 and skip all evaluations
        if not struct_diff.get('table_match', True):
            results.append({
                "id": it.id,
                "prompt": it.prompt,
                "generated_kql": _normalize_kql_for_display(gen_kql),
                "expected_kql": it.expected_kql,
                "score": 0.0,
                "metrics": {
                    "result_equality": 0.0,
                    "exec_success": 0.0,
                    "structural_similarity": 0.0,
                    "schema_match_score": 0.0,
                    "rows_match_score": 0.0,
                    "llm_graded_similarity": 0.0,
                },
                "struct_diff": struct_diff,
                "schema_match": {"different_tables": True},
                "rows_match": {"different_tables": True},
                "exec_error": None,
                "has_results": False,
                "gen_statistics": None,
                "llm_graded_similarity_score": 0.0,
            })
            continue

        # LLM-graded similarity: ask LLM to compare queries and grade 0-1, ignoring column name differences
        llm_graded_similarity = grade_kql_similarity(it.expected_kql, gen_kql)
        if llm_graded_similarity is None:
            llm_graded_similarity = "NA"

        # Expected snapshot and schema from executing expected_kql
        schema_match = None
        rows_match = None
        exp_statistics = None
        # Determine if strict row order is required based on prompt content:
        # Check if prompt implies ordering (sort, order, top, bottom, first, last, etc.)
        # If prompt doesn't imply ordering, use order-insensitive comparison regardless of query clauses
        strict_rows = False
        import re as _re
        # Check if the prompt mentions ordering concepts
        prompt_lower = it.prompt.lower()
        order_keywords = ['sort', 'order', 'top', 'bottom', 'first', 'last', 'oldest', 'newest', 'earliest', 'latest', 'most recent', 'least recent']
        prompt_implies_order = any(keyword in prompt_lower for keyword in order_keywords)
        
        # Only enforce strict order if the prompt implies ordering
        if prompt_implies_order:
            order_pat = _re.compile(r"\b(order|sort)\s+by\b", _re.IGNORECASE)
            if order_pat.search(gen_kql) or order_pat.search(it.expected_kql or ""):
                strict_rows = True

        # Load expected results from snapshot file or execute expected query
        if it.expected_output_path:
            exp_cols, exp_rows = load_csv_snapshot(it.expected_output_path)
        else:
            # Execute expected_kql to derive expected schema and rows (no snapshot)
            exp_exec_out = _run_kql(workspace_id, it.expected_kql, start_iso, end_iso)
            
            # If execution failed due to missing timespan, retry with 24-hour fallback
            if exp_exec_out.get("retry_with_timespan"):
                exp_exec_out = _run_kql(workspace_id, it.expected_kql, start_iso, end_iso, fallback_timespan_hours=24)
            
            exp_cols = exp_exec_out.get("columns") or []
            exp_rows = exp_exec_out.get("rows") or []
        
        schema_match = compare_schema(exp_cols, exec_out["columns"], strict_order=False)
        rows_match = compare_rows(exp_rows, exec_out["rows"], strict_order=strict_rows, tolerances=it.tolerances)

        # Must-pass gates
        gates_ok = True
        # Time filter presence gate (basic heuristic, case-insensitive)
        import re as _re
        has_where = _re.search(r"\|\s*where", gen_kql, flags=_re.IGNORECASE) is not None
        has_timegenerated = _re.search(r"timegenerated", gen_kql, flags=_re.IGNORECASE) is not None
        if not has_where and not has_timegenerated:
            # allow when dataset explicitly disables
            gates_ok = gates_ok and True

        # Calibrated schema match score with context-aware penalties
        schema_details = schema_match.get("details") or {}
        base_schema_score = float(schema_details.get("overlap_ratio", 0.0))
        extras = [str(x) for x in (schema_details.get("extra") or [])]
        missing = [str(x) for x in (schema_details.get("missing") or [])]
        total_cols = max(1, len(exec_out.get("columns") or []))

        def _is_critical_field(field_name: str, prompt: str, expected_query: str) -> bool:
            """Determine if a field is critical based on prompt and query context."""
            field_lower = field_name.lower()
            prompt_lower = prompt.lower()
            query_lower = expected_query.lower()
            
            # Check if field is explicitly mentioned in project clause (these are explicitly requested)
            project_match = _re.search(r'\|\s*project\s+([^|]+)', expected_query, flags=_re.IGNORECASE)
            if project_match:
                project_clause = project_match.group(1).lower()
                # Check if this specific field appears in the project
                project_fields = [f.strip().split('=')[0].strip() for f in project_clause.split(',')]
                if any(field_lower == pf.lower() or field_lower in pf.lower() for pf in project_fields):
                    return True
            
            # Check if field is mentioned in the prompt (by name or close match)
            # Remove common suffixes for matching
            field_base = field_lower.replace('name', '').replace('id', '').replace('uid', '')
            if field_lower in prompt_lower or field_base in prompt_lower:
                return True
            
            # Check if this is an aggregation or simple list query
            aggregation_keywords = ['aggregate', 'summarize', 'count', 'sum', 'average', 'total', 'max', 'min', 'most', 'least', 'highest', 'lowest']
            is_aggregation = any(keyword in prompt_lower for keyword in aggregation_keywords) or 'summarize' in query_lower
            
            simple_list_keywords = ['list', 'show all', 'display all', 'get all', 'find all']
            is_simple_list = any(keyword in prompt_lower for keyword in simple_list_keywords) and not any(keyword in prompt_lower for keyword in ['time', 'when', 'date', 'recent', 'old', 'new', 'first', 'last', 'order', 'sort'])
            
            # If prompt asks for a "list" or "show" items, identifier fields are critical
            list_indicators = ['list', 'show', 'find', 'get', 'display', 'identify', 'which']
            asks_for_list = any(indicator in prompt_lower for indicator in list_indicators)
            
            if asks_for_list:
                # Identifier patterns - these are critical when listing items
                identifier_patterns = ['id', 'uid', 'name']
                if any(pattern in field_lower for pattern in identifier_patterns):
                    # But only if it's a reasonable identifier (not just any field with 'name' in it)
                    # Check if it's a primary identifier for the entity being queried
                    entity_keywords = ['pod', 'container', 'node', 'namespace', 'service', 'event']
                    for entity in entity_keywords:
                        if entity in prompt_lower:
                            # If field contains both entity and identifier pattern, it's critical
                            if entity in field_lower and any(p in field_lower for p in identifier_patterns):
                                return True
                            # Or if it's just the identifier pattern itself (like "Name", "ID")
                            if field_lower in identifier_patterns:
                                return True
            
            # TimeGenerated and timestamp fields are generally critical
            # EXCEPT when it's an aggregation or simple list without time context
            if 'timegenerated' in field_lower or 'timestamp' in field_lower:
                # Not critical if it's an aggregation or simple list
                if is_aggregation or is_simple_list:
                    return False
                # Otherwise, timestamp is critical
                return True
            
            return False

        def _extract_group_by_cols(kql_text: str) -> set[str]:
            import re
            s = (kql_text or "")
            m = re.search(r"\bsummarize\b.*?\bby\b([^|]+)", s, flags=re.IGNORECASE | re.DOTALL)
            if not m:
                return set()
            by_clause = m.group(1)
            cols = set()
            for part in by_clause.split(","):
                token = part.strip()
                # try to extract a plausible column identifier (alphanum, underscore, dot) possibly inside functions
                idm = re.search(r"([A-Za-z_][A-Za-z0-9_\.]*)", token)
                if idm:
                    cols.add(idm.group(1).lower())
            return cols

        group_by_cols = _extract_group_by_cols(gen_kql)
        
        # Calculate penalty for extra fields (minor impact)
        extra_penalty = 0.0
        for e in extras:
            if e.lower() in group_by_cols:
                # Extra group-by columns are slightly more problematic
                extra_penalty += 0.02
            else:
                # Extra non-group-by columns are minimal impact
                extra_penalty += 0.01
        
        # Calculate penalty for missing fields (context-aware)
        missing_penalty = 0.0
        for m in missing:
            if _is_critical_field(m, it.prompt, it.expected_kql):
                # Missing critical field: major penalty
                missing_penalty += 0.4
            else:
                # Missing non-critical field: minor penalty
                missing_penalty += 0.05
        
        # Normalize penalties by total columns to keep them reasonable
        normalized_extra_penalty = extra_penalty / total_cols if total_cols > 0 else 0.0
        normalized_missing_penalty = missing_penalty / max(1, len(schema_details.get("expected", [])))
        
        # Total penalty is combination of both
        total_penalty = normalized_extra_penalty + normalized_missing_penalty
        schema_score = max(0.0, min(1.0, base_schema_score - total_penalty))

        # Calibrated rows match score from comparator details
        rows_details = rows_match.get("details") or {}
        rows_score = float(rows_details.get("overlap_ratio", 0.0))

        # Execution-success metric and has_results flag
        exec_success = 1.0  # reached only if no exec error
        has_results = bool(exec_out.get("rows"))

        metrics = {
            "result_equality": 1.0 if (schema_match.get("match") and rows_match.get("match")) else 0.0,
            "exec_success": exec_success,
            "structural_similarity": structural_similarity,
            "schema_match_score": schema_score,
            "rows_match_score": rows_score,
            "llm_graded_similarity": llm_graded_similarity,
        }
        # Build numeric-only copy for scoring; treat non-numeric as 0.0
        scoring_metrics = {k: (v if isinstance(v, (int, float)) else 0.0) for k, v in metrics.items()}
        # If llm_graded_similarity is NA, zero its weight so it doesn't affect the score
        weights_used = dict(DEFAULT_WEIGHTS)
        if not isinstance(metrics.get("llm_graded_similarity"), (int, float)):
            weights_used["llm_graded_similarity"] = 0.0
        
        # Normalize weights to sum to 1.0 after zeroing NA metrics
        total_weight = sum(weights_used.values())
        if total_weight > 0:
            weights_used = {k: v / total_weight for k, v in weights_used.items()}
        
        score = score_test(scoring_metrics, weights_used)
        if not gates_ok:
            score = 0.0

        results.append({
            "id": it.id,
            "prompt": it.prompt,
            "generated_kql": _normalize_kql_for_display(gen_kql),
            "expected_kql": it.expected_kql,
            "score": score,
            "metrics": metrics,
            "struct_diff": struct_diff,
            "schema_match": schema_match,
            "rows_match": rows_match,
            "exec_error": _extract_innermost_error_message(exec_out.get("error", "")),
            "has_results": has_results,
            "gen_statistics": exec_out.get("statistics"),
            "llm_graded_similarity_score": llm_graded_similarity if isinstance(llm_graded_similarity, (int, float)) else None,
            "critical_issue": None,
        })

    # Get model details
    cfg = load_config()
    model_details = {
        "deployment": cfg.deployment if cfg else "unknown",
        "api_version": cfg.api_version if cfg else "unknown",
        "system_prompt": SYSTEM_PROMPT
    }

    # Calculate summary statistics
    total_count = len(results)
    valid_queries = [r for r in results if not r.get("critical_issue")]
    invalid_queries = [r for r in results if r.get("critical_issue")]
    
    valid_count = len(valid_queries)
    invalid_count = len(invalid_queries)
    valid_rate = (valid_count / total_count * 100) if total_count > 0 else 0
    invalid_rate = (invalid_count / total_count * 100) if total_count > 0 else 0
    
    # For invalid queries, count auth vs syntax errors
    auth_errors = [r for r in invalid_queries if r.get("critical_issue", "").startswith("authentication")]
    syntax_errors = [r for r in invalid_queries if r.get("critical_issue", "").startswith("Invalid syntax")]
    
    auth_count = len(auth_errors)
    syntax_count = len(syntax_errors)
    auth_rate = (auth_count / invalid_count * 100) if invalid_count > 0 else 0
    syntax_rate = (syntax_count / invalid_count * 100) if invalid_count > 0 else 0
    
    # For valid queries, calculate average scores
    def safe_avg(values):
        numeric_values = [v for v in values if isinstance(v, (int, float)) and v != "NA"]
        return round(sum(numeric_values) / len(numeric_values), 3) if numeric_values else "NA"
    
    avg_structural = safe_avg([r.get("metrics", {}).get("structural_similarity", "NA") for r in valid_queries])
    avg_schema = safe_avg([r.get("metrics", {}).get("schema_match_score", "NA") for r in valid_queries])
    avg_rows = safe_avg([r.get("metrics", {}).get("rows_match_score", "NA") for r in valid_queries])
    avg_llm_grade = safe_avg([r.get("metrics", {}).get("llm_graded_similarity", "NA") for r in valid_queries])
    
    summary = {
        "total_count": total_count,
        "valid_queries": {
            "count": valid_count,
            "rate": round(valid_rate, 2),
            "avg_structural_similarity": avg_structural,
            "avg_schema_match": avg_schema,
            "avg_rows_match": avg_rows,
            "avg_llm_grade": avg_llm_grade
        },
        "invalid_queries": {
            "count": invalid_count,
            "rate": round(invalid_rate, 2),
            "authentication_errors": {
                "count": auth_count,
                "rate": round(auth_rate, 2)
            },
            "syntax_errors": {
                "count": syntax_count,
                "rate": round(syntax_rate, 2)
            }
        }
    }

    report = {
        "model": model_details,
        "dataset": args.dataset,
        "count": len(items),
        "elapsed_sec": round(time.time() - start_ts, 3),
        "summary": summary,
        "results": results
    }
    out_path = os.path.join(args.out, "report.json")
    write_json(out_path, report)
    print(f"Wrote report to {out_path}")
    
    # Generate HTML report
    html_path = os.path.join(args.out, "report.html")
    _write_html_report(report, html_path)
    print(f"Wrote HTML report to {html_path}")


def _write_html_report(report: dict, output_path: str):
    """Generate an HTML report with a table view of evaluation results."""
    results = report.get("results", [])
    
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>KQL Evaluation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .metadata {{
            color: #666;
            font-size: 14px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        th {{
            background-color: #0078d4;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
            vertical-align: top;
        }}
        tr:hover {{
            background-color: #f9f9f9;
        }}
        .prompt-cell {{
            max-width: 300px;
            font-weight: 500;
            color: #333;
        }}
        .kql-cell {{
            max-width: 400px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            white-space: pre-wrap;
            word-break: break-word;
            background-color: #f8f8f8;
            padding: 8px;
            border-radius: 4px;
        }}
        .score-cell {{
            text-align: center;
            font-weight: 600;
            font-size: 16px;
        }}
        .metric-cell {{
            text-align: center;
            font-size: 14px;
        }}
        .status-success {{
            color: #107c10;
        }}
        .status-valid {{
            color: #6c6c6c;
        }}
        .status-failed {{
            color: #d13438;
        }}
        .status-na {{
            color: #8a8a8a;
        }}
        .score-perfect {{
            color: #107c10;
            background-color: #e6f7e6;
            padding: 4px 8px;
            border-radius: 4px;
        }}
        .score-good {{
            color: #7a7a00;
            background-color: #ffffcc;
            padding: 4px 8px;
            border-radius: 4px;
        }}
        .score-poor {{
            color: #d13438;
            background-color: #ffe6e6;
            padding: 4px 8px;
            border-radius: 4px;
        }}
        .score-na {{
            color: #8a8a8a;
        }}
        .error-cell {{
            color: #d13438;
            font-size: 12px;
            font-style: italic;
            max-width: 300px;
        }}
        .system-prompt {{
            background-color: #f8f8f8;
            border-left: 4px solid #0078d4;
            padding: 12px;
            margin-top: 15px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        .summary {{
            background-color: #f0f8ff;
            border: 2px solid #0078d4;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
        }}
        .summary h2 {{
            margin-top: 0;
            color: #0078d4;
            font-size: 20px;
        }}
        .summary-section {{
            margin: 15px 0;
        }}
        .summary-section h3 {{
            color: #005a9e;
            font-size: 16px;
            margin-bottom: 8px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 8px;
        }}
        .summary-item {{
            padding: 8px;
            background-color: white;
            border-radius: 4px;
            border: 1px solid #d1d1d1;
        }}
        .summary-label {{
            font-weight: bold;
            color: #333;
            font-size: 13px;
        }}
        .summary-value {{
            font-size: 18px;
            color: #0078d4;
            margin-top: 4px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>KQL Evaluation Report</h1>
        <div class="metadata">
            <strong>Model Deployment:</strong> {deployment}<br>
            <strong>API Version:</strong> {api_version}<br>
            <strong>Dataset:</strong> {dataset}<br>
            <strong>Total Tests:</strong> {count}<br>
            <strong>Elapsed Time:</strong> {elapsed_sec}s
        </div>
        <div class="system-prompt">
            <strong>System Prompt:</strong><br>
            {system_prompt}
        </div>
    </div>
    
    <div class="summary">
        <h2>Summary Statistics</h2>
        
        <div class="summary-section">
            <h3>Query Execution Results</h3>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-label">Valid Queries</div>
                    <div class="summary-value">{valid_count} ({valid_rate}%)</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Invalid Queries</div>
                    <div class="summary-value">{invalid_count} ({invalid_rate}%)</div>
                </div>
            </div>
        </div>
        
        <div class="summary-section">
            <h3>Invalid Queries Breakdown</h3>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-label">Authentication Errors</div>
                    <div class="summary-value">{auth_count} ({auth_rate}%)</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Syntax Errors</div>
                    <div class="summary-value">{syntax_count} ({syntax_rate}%)</div>
                </div>
            </div>
        </div>
        
        <div class="summary-section">
            <h3>Valid Queries - Average Scores</h3>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-label">Structural Similarity</div>
                    <div class="summary-value">{avg_structural}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Schema Match</div>
                    <div class="summary-value">{avg_schema}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Rows Match</div>
                    <div class="summary-value">{avg_rows}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">LLM Grade</div>
                    <div class="summary-value">{avg_llm_grade}</div>
                </div>
            </div>
        </div>
    </div>
    
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Prompt</th>
                <th>Expected KQL</th>
                <th>Generated KQL</th>
                <th>Expected Rows</th>
                <th>Generated Rows</th>
                <th>Exec Status</th>
                <th>Critical Issue</th>
                <th>Overall Score</th>
                <th>Structural Similarity</th>
                <th>Schema Match</th>
                <th>Rows Match</th>
                <th>LLM Grade</th>
            </tr>
        </thead>
        <tbody>
""".format(
        deployment=report.get("model", {}).get("deployment", "unknown"),
        api_version=report.get("model", {}).get("api_version", "unknown"),
        system_prompt=_escape_html(report.get("model", {}).get("system_prompt", "")),
        dataset=report.get("dataset", ""),
        count=report.get("count", 0),
        elapsed_sec=report.get("elapsed_sec", 0),
        valid_count=report.get("summary", {}).get("valid_queries", {}).get("count", 0),
        valid_rate=report.get("summary", {}).get("valid_queries", {}).get("rate", 0),
        invalid_count=report.get("summary", {}).get("invalid_queries", {}).get("count", 0),
        invalid_rate=report.get("summary", {}).get("invalid_queries", {}).get("rate", 0),
        auth_count=report.get("summary", {}).get("invalid_queries", {}).get("authentication_errors", {}).get("count", 0),
        auth_rate=report.get("summary", {}).get("invalid_queries", {}).get("authentication_errors", {}).get("rate", 0),
        syntax_count=report.get("summary", {}).get("invalid_queries", {}).get("syntax_errors", {}).get("count", 0),
        syntax_rate=report.get("summary", {}).get("invalid_queries", {}).get("syntax_errors", {}).get("rate", 0),
        avg_structural=report.get("summary", {}).get("valid_queries", {}).get("avg_structural_similarity", "NA"),
        avg_schema=report.get("summary", {}).get("valid_queries", {}).get("avg_schema_match", "NA"),
        avg_rows=report.get("summary", {}).get("valid_queries", {}).get("avg_rows_match", "NA"),
        avg_llm_grade=report.get("summary", {}).get("valid_queries", {}).get("avg_llm_grade", "NA")
    )
    
    for result in results:
        test_id = result.get("id", "")
        prompt = result.get("prompt", "")
        expected_kql = result.get("expected_kql", "")
        generated_kql = result.get("generated_kql", "")
        
        # Get metrics
        metrics = result.get("metrics", {})
        exec_success = metrics.get("exec_success", "NA")
        overall_score = result.get("score", "NA")
        structural_sim = metrics.get("structural_similarity", "NA")
        schema_match = metrics.get("schema_match_score", "NA")
        rows_match = metrics.get("rows_match_score", "NA")
        llm_grade = metrics.get("llm_graded_similarity", "NA")
        
        # Execution status
        if result.get("exec_error"):
            exec_status = f'<span class="status-failed">Failed</span><br><span class="error-cell">{_escape_html(result.get("exec_error", ""))}</span>'
        elif exec_success == 1.0:
            exec_status = '<span class="status-valid">Valid Query</span>'
        elif exec_success == "NA":
            exec_status = '<span class="status-na">N/A</span>'
        else:
            exec_status = f'<span class="status-failed">{exec_success}</span>'
        
        # Critical issue
        critical_issue = result.get("critical_issue")
        if critical_issue:
            critical_issue_html = f'<span class="error-cell">{_escape_html(critical_issue)}</span>'
        else:
            critical_issue_html = '<span class="status-valid">-</span>'
        
        # Format scores with color coding
        def format_score(score):
            if score == "NA":
                return '<span class="score-na">N/A</span>'
            try:
                score_val = float(score)
                if score_val >= 0.95:
                    return f'<span class="score-perfect">{score_val:.3f}</span>'
                elif score_val >= 0.7:
                    return f'<span class="score-good">{score_val:.3f}</span>'
                else:
                    return f'<span class="score-poor">{score_val:.3f}</span>'
            except (ValueError, TypeError):
                return f'<span class="score-na">{score}</span>'
        
        # Get row counts from rows_match details
        rows_match_data = result.get("rows_match", {})
        rows_details = rows_match_data.get("details", {})
        expected_row_count = rows_details.get("expected_count", "N/A")
        actual_row_count = rows_details.get("actual_count", "N/A")
        
        html += f"""
            <tr>
                <td>{_escape_html(test_id)}</td>
                <td class="prompt-cell">{_escape_html(prompt)}</td>
                <td class="kql-cell">{_escape_html(expected_kql)}</td>
                <td class="kql-cell">{_escape_html(generated_kql)}</td>
                <td class="metric-cell">{expected_row_count}</td>
                <td class="metric-cell">{actual_row_count}</td>
                <td class="metric-cell">{exec_status}</td>
                <td class="metric-cell">{critical_issue_html}</td>
                <td class="score-cell">{format_score(overall_score)}</td>
                <td class="metric-cell">{format_score(structural_sim)}</td>
                <td class="metric-cell">{format_score(schema_match)}</td>
                <td class="metric-cell">{format_score(rows_match)}</td>
                <td class="metric-cell">{format_score(llm_grade)}</td>
            </tr>
"""
    
    html += """
        </tbody>
    </table>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def _escape_html(text):
    """Escape HTML special characters."""
    if not isinstance(text, str):
        text = str(text)
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))



if __name__ == "__main__":
    main()
