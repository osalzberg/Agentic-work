"""Deprecated runner adapter

This module previously contained a large legacy evaluation pipeline. It is now
deprecated in favor of `utils.batch_runner` (orchestration) and
`utils.report_builder` (exporting). The file remains as a lightweight
compatibility wrapper to preserve CLI usage patterns.

Usage:
  python -m query_evaluations.runner --dataset <path> --out <outdir>

This wrapper performs dataset loading and delegates execution + reporting to
the canonical modules.
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

from query_evaluations.dataset import load_dataset
from utils.batch_runner import run_batch
from utils.report_builder import build_json, write_html_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Deprecated runner wrapper")
    parser.add_argument("--dataset", required=True, help="Path to JSONL/CSV dataset")
    parser.add_argument("--out", required=True, help="Output report folder")
    parser.add_argument(
        "--workspace", default=os.getenv("WORKSPACE_ID", ""), help="Workspace ID"
    )
    args = parser.parse_args()

    items = load_dataset(args.dataset)
    prompts: List[Dict[str, Any]] = []
    for it in items:
        prompts.append(
            {
                "id": getattr(it, "id", None) or it.get("id"),
                "prompt": getattr(it, "prompt", None) or it.get("prompt") or getattr(it, "nl_prompt", None) or it.get("nl_prompt", ""),
                "expected_query": getattr(it, "expected_kql", None) or it.get("expected_kql") or it.get("expected_query"),
            }
        )

    # Delegate to canonical runner
    results = run_batch(prompts, agent=None, execute=True)

    # Build and write canonical reports
    os.makedirs(args.out, exist_ok=True)
    json_str = build_json(results, {"model": {}})
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        f.write(json_str)

    write_html_report({"results": results, "model": {}, "dataset": args.dataset, "count": len(items)}, os.path.join(args.out, "report.html"))
    print(f"Wrote report to {args.out}")


if __name__ == "__main__":
    main()
from __future__ import annotations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NL2KQL Evaluation Runner (delegates to utils.batch_runner)"
    )
    parser.add_argument("--dataset", required=True, help="Path to JSONL/CSV dataset")
    parser.add_argument("--out", required=True, help="Output report folder")
    parser.add_argument(
        "--workspace",
        default=os.getenv("WORKSPACE_ID", ""),
        help="Workspace ID (overrides dataset)",
    )
    args = parser.parse_args()

    items = load_dataset(args.dataset)

    # Build prompts list expected by utils.batch_runner.run_batch
    prompts = []
    for it in items:
        prompts.append(
            {
                "id": getattr(it, "id", None) or it.get("id"),
                "prompt": getattr(it, "prompt", None) or it.get("prompt") or getattr(it, "nl_prompt", None) or it.get("nl_prompt", ""),
                "expected_query": getattr(it, "expected_kql", None) or it.get("expected_kql") or it.get("expected_query"),
            }
        )

    # Use canonical runner from utils
    try:
        from utils.batch_runner import run_batch as canonical_run_batch

        results = canonical_run_batch(prompts, agent=None, execute=True)
        os.makedirs(args.out, exist_ok=True)
        write_json(results, os.path.join(args.out, "results.json"))
        print(f"Wrote results to: {args.out}")
        return
    except Exception:
        # Fallback: simple legacy behavior (generate queries only, no execution)
        results = []
        for it in items:
            gen_kql = _simple_translate_nl_to_kql(getattr(it, "prompt", None) or it.get("prompt") or "")
            results.append({"id": getattr(it, "id", None) or it.get("id"), "prompt": getattr(it, "prompt", None) or it.get("prompt"), "generated_kql": gen_kql, "expected_kql": getattr(it, "expected_kql", None) or it.get("expected_kql")})
        os.makedirs(args.out, exist_ok=True)
        write_json(results, os.path.join(args.out, "results.json"))
        print(f"Wrote fallback results to: {args.out}")
        "extend",
        "join",
        "order by",
        "take",
        "top",
        "union",
        "let",
        "parse",
        "parse_json",
        "make-series",
        "render",
        "bin",
        "by",
    ]
    kw_re = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b", re.IGNORECASE
    )
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
    joined = re.sub(
        r"(?<![=!<>])\s*=\s*(?!=)", " = ", joined
    )  # = but not part of ==, !=, <=, >=
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
    lines = [
        ln for ln in lines if not re.match(r"^\s*//\s*meta", ln, flags=re.IGNORECASE)
    ]
    joined = " ".join(lines)
    joined = re.sub(r"\s+", " ", joined).strip()

    # Normalize quotes: convert single quotes to double quotes
    joined = joined.replace("'", '"')

    # Lowercase KQL keywords only
    keywords = [
        "where",
        "project",
        "summarize",
        "extend",
        "join",
        "order by",
        "take",
        "top",
        "union",
        "let",
        "parse",
        "parse_json",
        "make-series",
        "render",
        "bin",
        "by",
    ]
    kw_re = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b", re.IGNORECASE
    )
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
    parser = argparse.ArgumentParser(
        description="NL2KQL Evaluation Runner (see docs/nl2kql_evaluation_prd.md)"
    )
    parser.add_argument("--dataset", required=True, help="Path to JSONL/CSV dataset")
    parser.add_argument("--out", required=True, help="Output report folder")
    parser.add_argument(
        "--workspace",
        default=os.getenv("WORKSPACE_ID", ""),
        help="Workspace ID (overrides dataset)",
    )
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
            results.append(
                {
                    # Prefer canonical orchestration using utils.batch_runner if available
                    try:
                        from utils.batch_runner import \
                            run_batch as canonical_run_batch

                        prompts = []
                        for it in items:
                            prompts.append(
                                {
                                    "id": getattr(it, "id", None) or it.get("id"),
                                    "prompt": getattr(it, "prompt", None) or it.get("prompt") or getattr(it, "nl_prompt", None) or it.get("nl_prompt", ""),
                                    "expected_query": getattr(it, "expected_kql", None) or it.get("expected_kql") or it.get("expected_query"),
                                }
                            )

                        results = canonical_run_batch(prompts, agent=None, execute=args.execute)

                        # write reports
                        os.makedirs(args.out, exist_ok=True)
                        write_json(results, os.path.join(args.out, "results.json"))
                        print(f"Wrote results to: {args.out}")
                        return
                    except Exception:
                        # Fall back to legacy runner behavior if canonical runner not available
                        pass

                    # If canonical runner not used, fall back to existing legacy processing loop
                    # (retain previous detailed logic)
                    # (legacy loop preserved above in file for reference)
                    print("Canonical runner not available; running legacy dataset processing is not implemented in this adapter.")
            results.append(
                {
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
                    # expected rows are not available at this point (expected query not executed);
                    # avoid NameError by returning None
                    "expected_rows_count": None,
                    "returned_rows_count": len(exec_out.get("rows") or []),
                    "has_results": False,
                    "gen_statistics": None,
                    "gen_statistics": exec_out.get("statistics"),
                    "exp_statistics": None,
                    "llm_graded_similarity_score": 0.0,
                }
            )
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
        order_keywords = [
            "sort",
            "order",
            "top",
            "bottom",
            "first",
            "last",
            "oldest",
            "newest",
            "earliest",
            "latest",
            "most recent",
            "least recent",
        ]
        prompt_implies_order = any(
            keyword in prompt_lower for keyword in order_keywords
        )

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
            try:
                exp_exec_out = kql_exec.execute_kql_query(
                    it.expected_kql, connection={"workspace_id": workspace_id}
                )
            except Exception:
                exp_exec_out = {"rows": [], "columns": [], "error": "execution_failed"}

            exp_cols = exp_exec_out.get("columns") or []
            exp_rows = exp_exec_out.get("rows") or []

        schema_match = compare_schema(exp_cols, exec_out["columns"], strict_order=False)
        # Set column headers for row comparison (used by fuzzy alignment)
        import query_evaluations.comparator as comparator

        comparator.compare_rows.expected_cols = exp_cols
        comparator.compare_rows.actual_cols = exec_out["columns"]
        rows_match = comparator.compare_rows(
            exp_rows,
            exec_out["rows"],
            strict_order=strict_rows,
            tolerances=it.tolerances,
        )

        # Must-pass gates
        gates_ok = True
        # Time filter presence gate (basic heuristic, case-insensitive)
        import re as _re

        has_where = _re.search(r"\|\s*where", gen_kql, flags=_re.IGNORECASE) is not None
        has_timegenerated = (
            _re.search(r"timegenerated", gen_kql, flags=_re.IGNORECASE) is not None
        )
        if not has_where and not has_timegenerated:
            # allow when dataset explicitly disables
            gates_ok = gates_ok and True

        # Calibrated schema match score with context-aware penalties
        schema_details = schema_match.get("details") or {}
        base_schema_score = float(schema_details.get("overlap_ratio", 0.0))
        extras = [str(x) for x in (schema_details.get("extra") or [])]
        missing = [str(x) for x in (schema_details.get("missing") or [])]
        total_cols = max(1, len(exec_out.get("columns") or []))

        def _is_critical_field(
            field_name: str, prompt: str, expected_query: str
        ) -> bool:
            """Determine if a field is critical based on prompt and query context."""
            field_lower = field_name.lower()
            prompt_lower = prompt.lower()
            # Only use prompt for all logic below
            # Check if field is mentioned in the prompt (by name or close match)
            field_base = (
                field_lower.replace("name", "").replace("id", "").replace("uid", "")
            )
            if field_lower in prompt_lower or field_base in prompt_lower:
                return True

            # Check if this is an aggregation or simple list query
            aggregation_keywords = [
                "aggregate",
                "summarize",
                "count",
                "sum",
                "average",
                "total",
                "max",
                "min",
                "most",
                "least",
                "highest",
                "lowest",
            ]
            is_aggregation = any(
                keyword in prompt_lower for keyword in aggregation_keywords
            )

            simple_list_keywords = [
                "list",
                "show all",
                "display all",
                "get all",
                "find all",
            ]
            is_simple_list = any(
                keyword in prompt_lower for keyword in simple_list_keywords
            ) and not any(
                keyword in prompt_lower
                for keyword in [
                    "time",
                    "when",
                    "date",
                    "recent",
                    "old",
                    "new",
                    "first",
                    "last",
                    "order",
                    "sort",
                ]
            )

            # If prompt asks for a "list" or "show" items, identifier fields are critical
            list_indicators = [
                "list",
                "show",
                "find",
                "get",
                "display",
                "identify",
                "which",
            ]
            asks_for_list = any(
                indicator in prompt_lower for indicator in list_indicators
            )

            if asks_for_list:
                # Identifier patterns - these are critical when listing items
                identifier_patterns = ["id", "uid", "name"]
                if any(pattern in field_lower for pattern in identifier_patterns):
                    # But only if it's a reasonable identifier (not just any field with 'name' in it)
                    # Check if it's a primary identifier for the entity being queried
                    entity_keywords = [
                        "pod",
                        "container",
                        "node",
                        "namespace",
                        "service",
                        "event",
                    ]
                    for entity in entity_keywords:
                        if entity in prompt_lower:
                            # If field contains both entity and identifier pattern, it's critical
                            if entity in field_lower and any(
                                p in field_lower for p in identifier_patterns
                            ):
                                return True
                            # Or if it's just the identifier pattern itself (like "Name", "ID")
                            if field_lower in identifier_patterns:
                                return True

            # TimeGenerated and timestamp fields are generally critical
            # EXCEPT when it's an aggregation or simple list without time context
            if "timegenerated" in field_lower or "timestamp" in field_lower:
                if is_aggregation or is_simple_list:
                    return False
                return True

            return False

        def _extract_group_by_cols(kql_text: str) -> set[str]:
            import re

            s = kql_text or ""
            m = re.search(
                r"\bsummarize\b.*?\bby\b([^|]+)", s, flags=re.IGNORECASE | re.DOTALL
            )
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
        normalized_missing_penalty = missing_penalty / max(
            1, len(schema_details.get("expected", []))
        )

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
            "result_equality": (
                1.0 if (schema_match.get("match") and rows_match.get("match")) else 0.0
            ),
            "exec_success": exec_success,
            "structural_similarity": structural_similarity,
            "schema_match_score": schema_score,
            "rows_match_score": rows_score,
            "llm_graded_similarity": llm_graded_similarity,
        }
        # Use canonical scoring surface
        try:
            canonical_score = calculate_total_score(
                gen_kql,
                it.expected_kql,
                exec_out.get("columns") or [],
                exp_cols or [],
                exec_out.get("rows") or [],
                exp_rows or [],
                it.prompt,
            )
        except Exception:
            canonical_score = {
                "total_score": 0.0,
                "is_successful": False,
                "threshold": 0.9,
                "weights": {"results_match": 0.5, "query_similarity": 0.5},
                "components": {},
                "query_similarity": 0.0,
                "results_match": 0.0,
            }

        normalized_score = canonical_score

        results.append(
            {
                "id": it.id,
                "prompt": it.prompt,
                "generated_kql": _normalize_kql_for_display(gen_kql),
                "expected_kql": it.expected_kql,
                "score": normalized_score,
                "metrics": metrics,
                "struct_diff": struct_diff,
                "schema_match": schema_match,
                "rows_match": rows_match,
                "exec_error": _extract_innermost_error_message(
                    exec_out.get("error", "")
                ),
                "expected_rows_count": (
                    len(exp_rows) if isinstance(exp_rows, list) else None
                ),
                "returned_rows_count": len(exec_out.get("rows") or []),
                "has_results": has_results,
                "gen_statistics": exec_out.get("statistics"),
                "exp_statistics": (
                    exp_exec_out.get("statistics")
                    if "exp_exec_out" in locals() and isinstance(exp_exec_out, dict)
                    else None
                ),
                "llm_graded_similarity_score": (
                    llm_graded_similarity
                    if isinstance(llm_graded_similarity, (int, float))
                    else None
                ),
                "critical_issue": None,
            }
        )

    # Get model details
    cfg = load_config()
    model_details = {
        "deployment": cfg.deployment if cfg else "unknown",
        "api_version": cfg.api_version if cfg else "unknown",
        "system_prompt": SYSTEM_PROMPT,
    }

    # Calculate summary statistics
        if calculate_total_score is not None: 
    valid_queries = [r for r in results if not r.get("critical_issue")]
    invalid_queries = [r for r in results if r.get("critical_issue")]

    valid_count = len(valid_queries)
    invalid_count = len(invalid_queries)
    valid_rate = (valid_count / total_count * 100) if total_count > 0 else 0
    invalid_rate = (invalid_count / total_count * 100) if total_count > 0 else 0

    # For invalid queries, count auth vs syntax errors
    auth_errors = [
        r
        for r in invalid_queries
        if r.get("critical_issue", "").startswith("authentication")
    ]
    syntax_errors = [
        r
        for r in invalid_queries
        if r.get("critical_issue", "").startswith("Invalid syntax")
    ]

    auth_count = len(auth_errors)
    syntax_count = len(syntax_errors)
    auth_rate = (auth_count / invalid_count * 100) if invalid_count > 0 else 0
    syntax_rate = (syntax_count / invalid_count * 100) if invalid_count > 0 else 0

    # For valid queries, calculate average scores
    def safe_avg(values):
        numeric_values = [
            v for v in values if isinstance(v, (int, float)) and v != "NA"
        ]
        return (
            round(sum(numeric_values) / len(numeric_values), 3)
            if numeric_values
            else "NA"
        )

    avg_structural = safe_avg(
        [r.get("metrics", {}).get("structural_similarity", "NA") for r in valid_queries]
    )
    avg_schema = safe_avg(
        [r.get("metrics", {}).get("schema_match_score", "NA") for r in valid_queries]
    )
    avg_rows = safe_avg(
        [r.get("metrics", {}).get("rows_match_score", "NA") for r in valid_queries]
    )
    avg_llm_grade = safe_avg(
        [r.get("metrics", {}).get("llm_graded_similarity", "NA") for r in valid_queries]
    )

    summary = {
        "total_count": total_count,
        "valid_queries": {
            "count": valid_count,
            "rate": round(valid_rate, 2),
            "avg_structural_similarity": avg_structural,
            "avg_schema_match": avg_schema,
            "avg_rows_match": avg_rows,
            "avg_llm_grade": avg_llm_grade,
        },
        "invalid_queries": {
            "count": invalid_count,
            "rate": round(invalid_rate, 2),
            "authentication_errors": {"count": auth_count, "rate": round(auth_rate, 2)},
            "syntax_errors": {"count": syntax_count, "rate": round(syntax_rate, 2)},
        },
    }

    report = {
        "model": model_details,
        "dataset": args.dataset,
        "count": len(items),
        "summary": summary,
        "results": results,
    }
    out_path = os.path.join(args.out, "report.json")
    # Use canonical JSON builder
    from utils.report_builder import build_json, write_html_report

    os.makedirs(args.out, exist_ok=True)
    json_str = build_json(report.get("results", []), {"model": report.get("model", {})})
    with open(out_path, "w", encoding="utf-8") as jf:
        jf.write(json_str)
    print(f"Wrote report to {out_path}")

    # Generate HTML report using canonical writer
    html_path = os.path.join(args.out, "report.html")
    write_html_report(report, html_path)
    print(f"Wrote HTML report to {html_path}")





if __name__ == "__main__":
    main()
