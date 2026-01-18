import time
import asyncio
from typing import Any, Dict, List, Optional

import utils.kql_exec as kql_exec
from query_evaluations.comparator import compare_rows, compare_schema


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
        # Generation (use shared helper to match /api/generate-kql behavior)
        from utils.generator import generate_kql

        gen_start = time.time()
        gen_query = generate_kql(agent, case.get("prompt", ""), model=model, system_prompt=system_prompt)
        gen_elapsed = time.time() - gen_start
        row["generated_query"] = gen_query
        row["gen_exec_stats"] = {"elapsed_sec": gen_elapsed}

        # Expected passthrough
        row["expected_query"] = case.get("expected_query")
        row["expected_rows_count"] = case.get("expected_rows_count") or (
            len(case.get("expected_rows", [])) if case.get("expected_rows") else None
        )

        # Execution
        if execute:
            gen_exec = kql_exec.execute_kql_query(gen_query)
            # `execute_kql_query` returns `tables`; try to extract rows in a compatible way
            gen_tables = gen_exec.get("tables", [])
            if gen_tables and isinstance(gen_tables, list):
                # take first table
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

            # If execution reported an error in exec_stats or returned an error-like structure,
            # mark the row status accordingly so callers know execution failed.
            gen_err_field = None
            try:
                gen_err_field = row["returned_exec_stats"].get("error")
            except Exception:
                gen_err_field = None
            if gen_err_field:
                row["status"] = "execution_failed"

            if row.get("expected_query"):
                exp_exec = kql_exec.execute_kql_query(row.get("expected_query"))
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
                # Mark expected execution failure if present
                try:
                    if row["expected_exec_stats"].get("error"):
                        row["status"] = "execution_failed"
                except Exception:
                    pass
            else:
                row["expected_rows"] = []
                row["expected_rows_count"] = row.get("expected_rows_count")
                row["expected_exec_stats"] = {}
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

            llm_score = calculate_total_score(
                generated_kql=gen_query or "",
                expected_kql=row.get("expected_query") or "",
                generated_columns=gen_columns,
                expected_columns=exp_columns,
                generated_results=gen_results,
                expected_results=exp_results,
                prompt=case.get("prompt", ""),
                generated_exec_stats=row.get("returned_exec_stats") or {},
                expected_exec_stats=row.get("expected_exec_stats") or {},
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
