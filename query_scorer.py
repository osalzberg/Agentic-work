"""
Query Scorer - Evaluates generated KQL queries against expected queries
Uses the mature query_evaluations system for consistent scoring.
"""

import os
import sys
from typing import Any, Dict, List, Tuple

# Add query_evaluations to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "query_evaluations"))

from query_evaluations.comparator import (compare_rows,  # type: ignore
                                          compare_schema)
from query_evaluations.explainer import grade_kql_similarity  # type: ignore
from query_evaluations.kql_parser import compare_kql_semantic  # type: ignore


def calculate_llm_graded_score(
    generated_kql: str, expected_kql: str, prompt: str
) -> Tuple[float, Dict]:
    """
    Use LLM to grade the generated query against expected query.
    Returns score 0-1.
    Uses the same Azure OpenAI deployment as query generation.
    """
    if not expected_kql or not generated_kql:
        return 0.0, {"reason": "Missing query for comparison"}

    try:
        # Get the actual model deployment being used
        from azure_openai_utils import load_config

        config = load_config()
        actual_model = config.deployment if config else "unknown"

        # Use query_evaluations LLM grading (uses same OpenAI config as query generation)
        score = grade_kql_similarity(expected_kql, generated_kql)

        if score is None:
            return 0.5, {
                "error": "LLM grading failed",
                "score": 0.5,
                "reasoning": "Error during grading, assigned neutral score",
            }

        details = {"score": score, "model": actual_model}

        return score, details

    except Exception as e:
        print(f"Error in LLM grading: {e}")
        return 0.5, {
            "error": str(e),
            "score": 0.5,
            "reasoning": "Error during grading, assigned neutral score",
        }


def calculate_total_score(
    generated_kql: str,
    expected_kql: str,
    generated_columns: List[str],
    expected_columns: List[str],
    generated_results: List[Dict],
    expected_results: List[Dict],
    prompt: str,
) -> Dict[str, Any]:
    """
    Calculate total score using query_evaluations scoring system.
    Uses the same Azure OpenAI deployment as query generation for LLM grading.

    Weights:
    - results_match: 0.55 (combines schema and rows match)
    - structural_similarity: 0.15
    - llm_graded_similarity: 0.30

    Note: Execution success is a gate (if execution fails, categorized as "Failed" with NA scores)
    Success threshold: weighted score >= 0.9
    """

    # Compute results/schema matching using deterministic comparators
    try:
        schema_result = compare_schema(
            expected_columns or [], generated_columns or [], strict_order=False
        )
    except Exception as e:
        schema_result = {"details": {"overlap_ratio": 0.0}, "match": False}

    # Build row matrices (lists of lists) from dict rows using provided column orders
    def _rows_from_dicts(rows: List[Dict], cols: List[str]) -> List[List[str]]:
        out: List[List[str]] = []
        if not rows:
            return out
        for r in rows:
            row_vals = [
                str(r.get(c, "")) if isinstance(r, dict) else str(r) for c in cols
            ]
            out.append(row_vals)
        return out

    expected_row_matrix = _rows_from_dicts(
        expected_results or [], expected_columns or []
    )
    generated_row_matrix = _rows_from_dicts(
        generated_results or [], generated_columns or []
    )

    # attach column context so compare_rows can do fuzzy alignment
    try:
        compare_rows.expected_cols = expected_columns or []
        compare_rows.actual_cols = generated_columns or []
        rows_result = compare_rows(
            expected_row_matrix, generated_row_matrix, strict_order=False
        )
    except Exception as e:
        rows_result = {"details": {"overlap_ratio": 0.0}, "match": False}

    # Extract deterministic metrics (0..1)
    schema_overlap = float(
        schema_result.get("details", {}).get("overlap_ratio", 0.0) or 0.0
    )
    rows_overlap = float(
        rows_result.get("details", {}).get("overlap_ratio", 0.0) or 0.0
    )

    # Combine schema + rows into a results_match metric (simple average)
    results_match_score = (schema_overlap + rows_overlap) / 2.0

    # LLM graded similarity applies only to the query semantic similarity
    query_similarity, llm_details = calculate_llm_graded_score(
        generated_kql, expected_kql, prompt
    )
    if query_similarity is None:
        query_similarity = 0.5

    # Weights: split equally between deterministic results_match and LLM query similarity
    results_weight = 0.5
    query_weight = 0.5

    # Compose total weighted score
    total_score = (results_match_score * results_weight) + (
        float(query_similarity) * query_weight
    )
    is_successful = total_score >= 0.9

    # Build canonical score object with normalized components (Pattern 1)
    components = {
        "query_similarity": {
            "score": round(float(query_similarity), 3),
            "weight": query_weight,
            "weighted_score": round(float(query_similarity) * query_weight, 3),
        },
        "results_match": {
            "score": round(results_match_score, 3),
            "weight": results_weight,
            "weighted_score": round(results_match_score * results_weight, 3),
            "components": {
                "schema_match": {
                    "score": round(schema_overlap, 3),
                    "explanation": "Numeric schema overlap (details omitted)",
                },
                "rows_match": {
                    "score": round(rows_overlap, 3),
                    "explanation": "Numeric rows overlap (details omitted)",
                },
            },
        },
    }

    # Return a compact score object; do not include raw comparator dicts (schema_result/rows_result)
    return {
        "total_score": round(float(total_score), 3),
        "is_successful": is_successful,
        "threshold": 0.9,
        "weights": {"results_match": results_weight, "query_similarity": query_weight},
        "components": components,
        "query_similarity": float(query_similarity),
        "results_match": float(results_match_score),
    }
