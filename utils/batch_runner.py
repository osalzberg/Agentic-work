import time
import asyncio
from typing import Any, Dict, List, Optional

import utils.kql_exec as kql_exec
from query_evaluations.comparator import compare_rows, compare_schema


def _normalize_kql(kql_text: Optional[str]) -> Optional[str]:
    """Normalize a KQL string for execution by unescaping common escape sequences.

    Currently this converts \" -> " and \' -> '. Keep this function small and
    conservative to avoid changing intended queries; adjust if you see more
    patterns in the wild.
    """
    if not isinstance(kql_text, str):
        return kql_text
    try:
        s = kql_text.replace('\\"', '"').replace("\\'", "'")
        return s
    except Exception:
        return kql_text


def generate_and_evaluate_query(
    case: Dict[str, Any],
    *,
    agent: Any,
    execute: bool = False,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Shared helper that evaluates a single prompt case.

    This implements the canonical pipeline used by `/api/generate-kql`
    and `/api/generate_and_evaluate_query`: generate -> execute -> compare -> score.
    It respects optional `model` and `system_prompt` overrides and
    returns a canonical result dict compatible with `run_batch` outputs.
    """
    # If agent supports model/system overrides via context functions, apply them
    # We intentionally do a best-effort import so this helper remains decoupled
    try:
        from azure_openai_utils import set_model_override, clear_model_override
    except Exception:
        set_model_override = None
        clear_model_override = None
    try:
        from prompt_builder import (
            set_system_prompt_override,
            clear_system_prompt_override,
        )
    except Exception:
        set_system_prompt_override = None
        clear_system_prompt_override = None

    # Apply overrides if provided (centralized here so generation+scoring see them)
    if set_model_override and model:
        try:
            set_model_override(model)
        except Exception:
            pass
    if set_system_prompt_override and system_prompt:
        try:
            set_system_prompt_override(system_prompt)
        except Exception:
            pass

    row: Dict[str, Any] = {"prompt_id": case.get("id"), "prompt": case.get("prompt")}
    try:

        # Expected query passthrough (we now execute the expected query first)
        row["expected_query"] = case.get("expected_query")
        row["expected_rows_count"] = case.get("expected_rows_count") or (
            len(case.get("expected_rows", [])) if case.get("expected_rows") else None
        )

        # Prepare expected placeholders
        row["expected_rows"] = []
        row["expected_exec_stats"] = {}

        # Generation (use shared helper to match /api/generate-kql behavior)
        from utils.generator import generate_kql

        gen_start = time.time()
        # If caller supplied a generated_query (e.g., via API), honor it and skip regeneration.
        if case.get("generated_query") is not None:
            gen_query = case.get("generated_query")
            gen_elapsed = 0.0
        else:
            gen_query = generate_kql(agent, case.get("prompt", ""), model=model, system_prompt=system_prompt)
            gen_elapsed = time.time() - gen_start
        # Preserve the raw generator output for diagnostics (may be str or dict)
        row["generated_query"] = gen_query
        row["gen_exec_stats"] = {"elapsed_sec": gen_elapsed}

        # Normalize generator output to a KQL string for execution. Accept
        # structured outputs from agents that return {kql_query, kql, query, type, ...}
        gen_kql_str = None
        if isinstance(gen_query, str):
            gen_kql_str = gen_query.strip() or None
        elif isinstance(gen_query, dict):
            gen_kql_str = (
                gen_query.get("kql_query")
                or gen_query.get("kql")
                or gen_query.get("query")
            )
            if isinstance(gen_kql_str, str):
                gen_kql_str = gen_kql_str.strip() or None

        # If the generator explicitly returned an error-type payload, do not execute
        if isinstance(gen_query, dict) and (gen_query.get("type") or "").lower() == "query_error":
            row["returned_rows"] = []
            row["returned_rows_count"] = 0
            row["returned_exec_stats"] = {"error": gen_query.get("error") or gen_query.get("message") or "generation_error", "elapsed_sec": gen_elapsed}
            row["status"] = "generation_failed"
            return row

        if not gen_kql_str and execute:
            row["returned_rows"] = []
            row["returned_rows_count"] = 0
            row["returned_exec_stats"] = {"error": "generated_query_not_string", "elapsed_sec": gen_elapsed}
            row["status"] = "generation_failed"
            # Short-circuit: do not attempt execution or expected query comparison
            return row

        # Expected passthrough (preserve counts provided by the case)
        row["expected_query"] = case.get("expected_query")
        row["expected_rows_count"] = case.get("expected_rows_count") or (
            len(case.get("expected_rows", [])) if case.get("expected_rows") else None
        )

        # Execution: run the generated query first (server-side). If generated
        # execution fails, do not run the expected query. Only if the generated
        # execution succeeds do we proceed to run the expected query.
        if execute:
            # Execute generated query on server
            gen_exec = kql_exec.execute_kql_query(_normalize_kql(gen_kql_str))
            gen_tables = gen_exec.get("tables", [])
            if gen_tables and isinstance(gen_tables, list):
                first = gen_tables[0] if gen_tables else {"columns": [], "rows": []}
                row["returned_rows"] = [
                    {c: r[idx] for idx, c in enumerate(first.get("columns", [])) if idx < len(r)}
                    for r in first.get("rows", [])
                ]
                row["returned_rows_count"] = gen_exec.get("returned_rows_count", len(row.get("returned_rows", [])))
            else:
                row["returned_rows"] = gen_exec.get("rows", [])
                row["returned_rows_count"] = gen_exec.get("returned_rows_count", len(row.get("returned_rows", [])))
            row["returned_exec_stats"] = gen_exec.get("exec_stats", {})

            # If generated execution failed, short-circuit and return the failure
            try:
                if row["returned_exec_stats"].get("error"):
                    row["status"] = "execution_failed"
                    # For backward compatibility surface generated exec stats in returned_exec_stats
                    # and avoid running expected query.
                    # Strip full row lists to avoid leaking data in outward-facing response
                    if "returned_rows" in row:
                        del row["returned_rows"]
                    if "expected_rows" in row:
                        del row["expected_rows"]
                    return row
            except Exception:
                row["status"] = "execution_failed"
                return row

            # Generated succeeded; now run expected (if provided)
            if row.get("expected_query"):
                exp_q = _normalize_kql(row.get("expected_query"))
                exp_exec = kql_exec.execute_kql_query(exp_q)
                exp_tables = exp_exec.get("tables", [])
                if exp_tables and isinstance(exp_tables, list):
                    first = exp_tables[0] if exp_tables else {"columns": [], "rows": []}
                    row["expected_rows"] = [
                        {c: r[idx] for idx, c in enumerate(first.get("columns", [])) if idx < len(r)}
                        for r in first.get("rows", [])
                    ]
                    row["expected_rows_count"] = exp_exec.get("returned_rows_count", len(row.get("expected_rows", [])))
                else:
                    row["expected_rows"] = exp_exec.get("rows", [])
                    row["expected_rows_count"] = exp_exec.get("returned_rows_count", len(row.get("expected_rows", [])))
                row["expected_exec_stats"] = exp_exec.get("exec_stats", {})

                # If expected execution failed, short-circuit and return expected failure
                try:
                    if row["expected_exec_stats"].get("error"):
                        row["status"] = "expected_execution_failed"
                        # Strip full row lists to avoid leaking data
                        if "returned_rows" in row:
                            del row["returned_rows"]
                        if "expected_rows" in row:
                            del row["expected_rows"]
                        return row
                except Exception:
                    row["status"] = "expected_execution_failed"
                    return row
        else:
            row["returned_rows"] = []
            row["returned_rows_count"] = 0
            row["returned_exec_stats"] = {}
            row["expected_rows"] = []
            row["expected_exec_stats"] = {}

        # Comparator
        expected_cols = []
        actual_cols = []
        if (
            row.get("expected_rows")
            and isinstance(row.get("expected_rows"), list)
            and row.get("expected_rows")
            and isinstance(row.get("expected_rows")[0], dict)
        ):
            expected_cols = list(row.get("expected_rows")[0].keys())
        if (
            row.get("returned_rows")
            and isinstance(row.get("returned_rows"), list)
            and row.get("returned_rows")
            and isinstance(row.get("returned_rows")[0], dict)
        ):
            actual_cols = list(row.get("returned_rows")[0].keys())

        schema_result = (
            compare_schema(expected_cols, actual_cols, strict_order=False)
            if expected_cols or actual_cols
            else {"match": False, "details": {}}
        )

        def _rows_to_lists(rows_list):
            if not rows_list:
                return []
            if isinstance(rows_list[0], dict):
                cols = list(rows_list[0].keys())
                return [[str(r.get(c)) for c in cols] for r in rows_list]
            return rows_list

        exp_rows_list = _rows_to_lists(row.get("expected_rows") or [])
        ret_rows_list = _rows_to_lists(row.get("returned_rows") or [])

        rows_result = compare_rows(exp_rows_list, ret_rows_list, strict_order=False)

        comparator = {
            "schema_match": round(float(schema_result.get("details", {}).get("overlap_ratio", 0.0)), 3),
            "rows_match": round(float(rows_result.get("details", {}).get("overlap_ratio", 0.0)), 3),
            "details": {**schema_result.get("details", {}), **rows_result.get("details", {})},
        }
        row["comparator"] = comparator

        # Build list-of-dict results for LLM scoring and for client display
        # Keep full results for scoring internally, but do not return full rows
        # in the outward-facing `row` dict to avoid leaking or sending large
        # result sets to clients. Instead, extract columns/counts for scoring
        # and preserve exec stats. We'll strip full row contents before return.
        gen_results = row.get("returned_rows") or []
        exp_results = row.get("expected_rows") or []
        gen_columns = []
        exp_columns = []
        if gen_results and isinstance(gen_results, list) and isinstance(gen_results[0], dict):
            gen_columns = list(gen_results[0].keys())
        if exp_results and isinstance(exp_results, list) and isinstance(exp_results[0], dict):
            exp_columns = list(exp_results[0].keys())

        # Calculate LLM-graded + deterministic score using existing scorer
        try:
            from query_scorer import calculate_total_score

            # Score should be based on the normalized KQL string (if any).
            # Pass the raw generator output separately inside diagnostics if needed.
            llm_score = calculate_total_score(
                generated_kql=gen_kql_str or "",
                expected_kql=row.get("expected_query") or "",
                generated_columns=gen_columns,
                expected_columns=exp_columns,
                generated_results=gen_results,
                expected_results=exp_results,
                prompt=case.get("prompt", ""),
                generated_exec_stats=row.get("returned_exec_stats") or {},
                expected_exec_stats=row.get("expected_exec_stats") or {},
                generated_query_raw=row.get("generated_query"),
            )
            # Keep the scorer's canonical object directly
            row["score"] = llm_score
        except Exception:
            # Fallback: deterministic comparator-derived score
            score = case.get("score") or {}
            try:
                qsim = score.get("query_similarity") or score.get("query_sim") or 0.0
            except Exception:
                qsim = 0.0
            canon = {
                "query_similarity": float(qsim),
                "results_match": {"value": float((score.get("results_match") if isinstance(score.get("results_match"), (int, float)) else (score.get("results_match", {}).get("value") if isinstance(score.get("results_match"), dict) else 0.0))) if score else 0.0}
            }
            try:
                rm_val = float(comparator.get("rows_match", canon.get("results_match", {}).get("value", 0.0)))
            except Exception:
                rm_val = canon.get("results_match", {}).get("value", 0.0) if isinstance(canon.get("results_match"), dict) else canon.get("results_match", 0.0)
            canon["results_match"] = {"value": round(float(rm_val), 3), "components": {"schema_match": comparator.get("schema_match"), "rows_match": comparator.get("rows_match")}}
            row["score"] = canon

        # warnings
        row["zero_rows_warning"] = False
        try:
            if row["score"]["results_match"]["components"]["rows_match"] == 1.0 and (row.get("returned_rows_count") in (None, 0)):
                row["zero_rows_warning"] = True
        except Exception:
            row["zero_rows_warning"] = False

        # Before returning, strip full row contents to only keep counts and
        # execution stats. This ensures API consumers receive only the number
        # of rows and exec metadata, not the full data payload.
        try:
            # preserve counts and exec stats
            returned_rows_count = row.get("returned_rows_count")
            expected_rows_count = row.get("expected_rows_count")
            returned_exec_stats = row.get("returned_exec_stats")
            expected_exec_stats = row.get("expected_exec_stats")
            # remove potentially large fields
            if "returned_rows" in row:
                del row["returned_rows"]
            if "expected_rows" in row:
                del row["expected_rows"]
            # restore counts and exec stats (they may have been present)
            row["returned_rows_count"] = returned_rows_count
            row["expected_rows_count"] = expected_rows_count
            row["returned_exec_stats"] = returned_exec_stats
            row["expected_exec_stats"] = expected_exec_stats
        except Exception:
            # if any unexpected issue, fall back to original row
            pass

        # Only mark as success when no prior failure status was set
        if not row.get("status") or row.get("status") == "success":
            row["status"] = "success"
        return row

    except Exception as ex:
        return {"prompt_id": case.get("id"), "prompt": case.get("prompt"), "error": str(ex), "status": "failed"}
    finally:
        # Clear overrides if we set them
        if clear_model_override and model:
            try:
                clear_model_override()
            except Exception:
                pass
        if clear_system_prompt_override and system_prompt:
            try:
                clear_system_prompt_override()
            except Exception:
                pass


def _canonicalize_score(score: Dict[str, Any]) -> Dict[str, Any]:
    # ensure pattern 1 shape: top-level numeric query_similarity and results_match
    qsim = score.get("query_similarity") or score.get("query_sim") or 0.0
    results_match = score.get("results_match") or {}
    components = results_match.get("components") or {}
    schema_match = (
        components.get("schema_match") or results_match.get("schema_match") or 0.0
    )
    rows_match = (
        components.get("rows_match")
        or results_match.get("rows_match")
        or results_match.get("rows")
        or 0.0
    )
    return {
        "query_similarity": float(qsim),
        "results_match": {
            "value": (
                float(results_match.get("value", 0.0))
                if isinstance(results_match, dict)
                else float(results_match or 0.0)
            ),
            "components": {
                "schema_match": float(schema_match),
                "rows_match": float(rows_match),
            },
        },
    }


def run_batch(
    prompts: List[Dict[str, Any]],
    *,
    agent: Optional[Any] = None,
    execute: bool = False,
    stop_on_critical_error: bool = True,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run a batch of prompts through the generator, optional execution, and evaluation.

    Returns a list of canonical result dicts suitable for report building. Each
    result will contain keys:
      - prompt_id, prompt_text
      - gen_query, gen_exec_stats
      - expected_query, expected_rows_count
      - returned_rows_count, returned_rows, returned_exec_stats
      - comparator results (schema_match, rows_match, diagnostics)
      - score (canonicalized pattern 1)

    The function intentionally keeps the implementation small and delegates
    generation to `agent` if provided.
    """
    results: List[Dict[str, Any]] = []
    # Require a real agent to be provided. Previously a lightweight fallback
    # was used here, but callers should explicitly supply an `agent` that
    # implements `process_natural_language(prompt)` to avoid surprising
    # behavior in batch runs.
    if agent is None:
        raise ValueError(
            "run_batch requires a valid 'agent' instance with method process_natural_language(prompt)."
        )

    for i, case in enumerate(prompts):
        try:
            res = generate_and_evaluate_query(case, agent=agent, execute=execute, model=model, system_prompt=system_prompt)
            results.append(res)
        except Exception as ex:
            err = {"prompt_id": case.get("id", i), "prompt": case.get("prompt"), "error": str(ex), "status": "failed"}
            results.append(err)
            if stop_on_critical_error:
                break

    return results
