import time
import asyncio
from typing import Any, Dict, List, Optional

import utils.kql_exec as kql_exec
from query_evaluations.comparator import compare_rows, compare_schema


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
    # Provide a lightweight fallback agent when none is supplied.
    # Avoid instantiating logs_agent.KQLAgent without a workspace_id.
    if agent is None:

        class _FallbackAgent:
            def process_natural_language(self, prompt: str) -> str:
                return f"KQL: {prompt}"

        agent = _FallbackAgent()

    for i, case in enumerate(prompts):
        row = {"prompt_id": case.get("id", i), "prompt": case.get("prompt")}
        try:
            gen_start = time.time()
            gen_query = agent.process_natural_language(case.get("prompt", ""))
            # If agent.process_natural_language is async it may return a coroutine.
            # Execute it synchronously here to ensure `gen_query` is a string.
            try:
                if asyncio.iscoroutine(gen_query):
                    gen_query = asyncio.run(gen_query)
            except Exception:
                try:
                    gen_query = str(gen_query)
                except Exception:
                    gen_query = ""
            gen_elapsed = time.time() - gen_start
            row["generated_query"] = gen_query
            row["gen_exec_stats"] = {"elapsed_sec": gen_elapsed}

            # Expected data pass-through
            row["expected_query"] = case.get("expected_query")
            row["expected_rows_count"] = case.get("expected_rows_count") or (
                len(case.get("expected_rows", []))
                if case.get("expected_rows")
                else None
            )

            # Optional execution of generated and expected queries
            if execute:
                gen_exec = kql_exec.execute_kql_query(gen_query)
                # normalize exec_kql response from execute_kql_query
                row["returned_rows"] = gen_exec.get("rows", [])
                row["returned_rows_count"] = gen_exec.get(
                    "returned_rows_count", len(row.get("returned_rows", []))
                )
                row["returned_exec_stats"] = gen_exec.get("exec_stats", {})
                # execute expected query if provided
                if row.get("expected_query"):
                    exp_exec = kql_exec.execute_kql_query(row.get("expected_query"))
                    row["expected_rows"] = exp_exec.get("rows", [])
                    row["expected_rows_count"] = exp_exec.get(
                        "returned_rows_count", len(row.get("expected_rows", []))
                    )
                    row["expected_exec_stats"] = exp_exec.get("exec_stats", {})
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

            # Comparator: use canonical comparator implementation
            try:
                # Schema comparator
                expected_cols = []
                actual_cols = []
                # If rows exist and are list-of-dicts, infer columns
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

                # Prepare rows as list-of-lists for row comparator
                def _rows_to_lists(rows_list):
                    if not rows_list:
                        return []
                    if isinstance(rows_list[0], dict):
                        cols = list(rows_list[0].keys())
                        return [[str(r.get(c)) for c in cols] for r in rows_list]
                    return rows_list

                exp_rows_list = _rows_to_lists(row.get("expected_rows") or [])
                ret_rows_list = _rows_to_lists(row.get("returned_rows") or [])

                rows_result = compare_rows(
                    exp_rows_list, ret_rows_list, strict_order=False
                )

                comparator = {
                    "schema_match": round(
                        float(
                            schema_result.get("details", {}).get("overlap_ratio", 0.0)
                        ),
                        3,
                    ),
                    "rows_match": round(
                        float(rows_result.get("details", {}).get("overlap_ratio", 0.0)),
                        3,
                    ),
                    "details": {
                        **schema_result.get("details", {}),
                        **rows_result.get("details", {}),
                    },
                }
            except Exception:
                comparator = {"schema_match": 0.0, "rows_match": 0.0, "details": {}}

            row["comparator"] = comparator

            # Score canonicalization: prefer score provided in case but update deterministic results_match
            score = case.get("score") or {}
            canon = _canonicalize_score(score)
            # overwrite deterministic results_match with comparator rows/schema
            try:
                rm_val = float(
                    comparator.get(
                        "rows_match", canon.get("results_match", {}).get("value", 0.0)
                    )
                )
            except Exception:
                rm_val = (
                    canon.get("results_match", {}).get("value", 0.0)
                    if isinstance(canon.get("results_match"), dict)
                    else canon.get("results_match", 0.0)
                )
            # Build Pattern 1 results_match structure
            canon["results_match"] = {
                "value": round(float(rm_val), 3),
                "components": {
                    "schema_match": comparator.get("schema_match"),
                    "rows_match": comparator.get("rows_match"),
                },
            }
            row["score"] = canon

            # zero rows warning
            row["zero_rows_warning"] = False
            try:
                if row["score"]["results_match"]["components"][
                    "rows_match"
                ] == 1.0 and (row.get("returned_rows_count") in (None, 0)):
                    row["zero_rows_warning"] = True
            except Exception:
                row["zero_rows_warning"] = False

            # Status
            row["status"] = "success"
            results.append(row)

        except Exception as ex:
            # error handling per-row
            err = {
                "prompt_id": case.get("id", i),
                "prompt": case.get("prompt"),
                "error": str(ex),
                "status": "failed",
            }
            results.append(err)
            if stop_on_critical_error:
                break

    return results
