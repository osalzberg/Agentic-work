"""
Query Scorer - Evaluates generated KQL queries against expected queries
Uses the mature query_evaluations system for consistent scoring.
"""

from typing import Dict, List, Any, Tuple
import sys
import os

# Add query_evaluations to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'query_evaluations'))

from comparator import compare_schema, compare_rows
from kql_parser import compare_kql_semantic
from explainer import grade_kql_similarity
from scorer import score_test, DEFAULT_WEIGHTS


def calculate_llm_graded_score(generated_kql: str, expected_kql: str, prompt: str, 
                                model: str = "gpt-4o-mini") -> Tuple[float, Dict]:
    """
    Use LLM to grade the generated query against expected query.
    Returns score 0-1.
    """
    if not expected_kql or not generated_kql:
        return 0.0, {"reason": "Missing query for comparison"}
    
    try:
        # Use query_evaluations LLM grading
        score = grade_kql_similarity(expected_kql, generated_kql)
        
        if score is None:
            return 0.5, {"error": "LLM grading failed", "score": 0.5, "reasoning": "Error during grading, assigned neutral score"}
        
        details = {
            "score": score,
            "reasoning": f"LLM similarity score based on semantic equivalence (ignoring column name differences)",
            "model": model
        }
        
        return score, details
        
    except Exception as e:
        print(f"Error in LLM grading: {e}")
        return 0.5, {"error": str(e), "score": 0.5, "reasoning": "Error during grading, assigned neutral score"}


def calculate_total_score(
    generated_kql: str,
    expected_kql: str,
    generated_columns: List[str],
    expected_columns: List[str],
    generated_results: List[Dict],
    expected_results: List[Dict],
    prompt: str,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Calculate total score using query_evaluations scoring system.
    
    Uses the same weights and logic as query_evaluations:
    - result_equality: 0.05 (binary pass/fail)
    - exec_success: 0.15 (assumed True if we got here)
    - schema_match_score: 0.25
    - rows_match_score: 0.25
    - structural_similarity: 0.20
    - llm_graded_similarity: 0.10
    
    Success threshold: weighted score >= 0.9
    """
    
    # 1. Schema match using query_evaluations comparator
    schema_match = compare_schema(expected_columns, generated_columns, strict_order=False)
    schema_score = schema_match.get("details", {}).get("overlap_ratio", 0.0)
    
    # 2. Rows match using query_evaluations comparator (order-insensitive)
    # Convert dict results to list-of-lists format
    def dict_to_rows(results: List[Dict], columns: List[str]) -> List[List[str]]:
        rows = []
        for row_dict in results:
            row = [str(row_dict.get(col, "")) for col in columns]
            rows.append(row)
        return rows
    
    expected_rows = dict_to_rows(expected_results, expected_columns) if expected_results else []
    generated_rows = dict_to_rows(generated_results, generated_columns) if generated_results else []
    
    rows_match = compare_rows(expected_rows, generated_rows, strict_order=False)
    rows_score = rows_match.get("details", {}).get("overlap_ratio", 0.0)
    
    # 3. Structural similarity using query_evaluations kql_parser
    def normalize_kql(kql: str) -> str:
        """Simple normalization for comparison."""
        import re
        s = kql.replace("\r\n", "\n").replace("\r", "\n")
        lines = [ln for ln in s.split("\n") if not re.match(r"^\s*//", ln)]
        return " ".join(lines).strip()
    
    normalized_expected = normalize_kql(expected_kql)
    normalized_generated = normalize_kql(generated_kql)
    
    semantic_comparison = compare_kql_semantic(normalized_expected, normalized_generated)
    structural_score = semantic_comparison.get('similarity', 0.0)
    
    # 4. LLM grading using query_evaluations explainer
    llm_score, llm_details = calculate_llm_graded_score(generated_kql, expected_kql, prompt, model)
    
    # 5. Result equality (binary check)
    result_equality = 1.0 if (schema_match.get("match") and rows_match.get("match")) else 0.0
    
    # 6. Execution success (assumed True if we have results)
    exec_success = 1.0
    
    # Build metrics dict for scoring
    metrics = {
        "result_equality": result_equality,
        "exec_success": exec_success,
        "schema_match_score": schema_score,
        "rows_match_score": rows_score,
        "structural_similarity": structural_score,
        "llm_graded_similarity": llm_score,
    }
    
    # Calculate weighted score using query_evaluations scorer
    total_score = score_test(metrics, DEFAULT_WEIGHTS)
    
    # Success threshold: 0.9 (consistent with requirement)
    is_successful = total_score >= 0.9
    
    return {
        "total_score": round(total_score, 3),
        "is_successful": is_successful,
        "threshold": 0.9,
        "weights": DEFAULT_WEIGHTS,
        "components": {
            "result_equality": {
                "score": round(result_equality, 3),
                "weighted_score": round(result_equality * DEFAULT_WEIGHTS["result_equality"], 3),
                "weight": DEFAULT_WEIGHTS["result_equality"],
                "details": {"binary_match": result_equality == 1.0}
            },
            "exec_success": {
                "score": round(exec_success, 3),
                "weighted_score": round(exec_success * DEFAULT_WEIGHTS["exec_success"], 3),
                "weight": DEFAULT_WEIGHTS["exec_success"],
                "details": {"query_executed": True}
            },
            "schema_match": {
                "score": round(schema_score, 3),
                "weighted_score": round(schema_score * DEFAULT_WEIGHTS["schema_match_score"], 3),
                "weight": DEFAULT_WEIGHTS["schema_match_score"],
                "details": schema_match.get("details", {})
            },
            "rows_match": {
                "score": round(rows_score, 3),
                "weighted_score": round(rows_score * DEFAULT_WEIGHTS["rows_match_score"], 3),
                "weight": DEFAULT_WEIGHTS["rows_match_score"],
                "details": rows_match.get("details", {})
            },
            "structural_similarity": {
                "score": round(structural_score, 3),
                "weighted_score": round(structural_score * DEFAULT_WEIGHTS["structural_similarity"], 3),
                "weight": DEFAULT_WEIGHTS["structural_similarity"],
                "details": semantic_comparison.get("details", {})
            },
            "llm_grading": {
                "score": round(llm_score, 3),
                "weighted_score": round(llm_score * DEFAULT_WEIGHTS["llm_graded_similarity"], 3),
                "weight": DEFAULT_WEIGHTS["llm_graded_similarity"],
                "details": llm_details
            }
        }
    }
