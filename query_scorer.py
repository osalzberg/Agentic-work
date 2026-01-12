"""
Query Scorer - Evaluates generated KQL queries against expected queries
Uses the mature query_evaluations system for consistent scoring.
"""

from typing import Dict, List, Any, Tuple
import sys
import os

# Add query_evaluations to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'query_evaluations'))

from comparator import compare_schema, compare_rows  # type: ignore
from kql_parser import compare_kql_semantic  # type: ignore
from explainer import grade_kql_similarity  # type: ignore
from scorer import score_test, DEFAULT_WEIGHTS  # type: ignore


def calculate_llm_graded_score(generated_kql: str, expected_kql: str, prompt: str) -> Tuple[float, Dict]:
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
            return 0.5, {"error": "LLM grading failed", "score": 0.5, "reasoning": "Error during grading, assigned neutral score"}
        
        details = {
            "score": score,
            "reasoning": f"LLM similarity score based on semantic equivalence (ignoring column name differences)",
            "model": actual_model
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
    prompt: str
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
    
    # Check if prompt explicitly mentions specific fields/columns
    prompt_lower = prompt.lower()
    fields_explicitly_requested = any(
        keyword in prompt_lower 
        for keyword in ['show', 'display', 'project', 'select', 'return', 'get', 'include']
    ) and any(
        field_keyword in prompt_lower
        for field_keyword in ['field', 'column', 'property', 'attribute']
    )
    
    # Check if queries involve aggregation (summarize) - different column names are acceptable
    is_aggregation = 'summarize' in expected_kql.lower() or 'summarize' in generated_kql.lower()
    
    # Check if both queries use bin() with potentially different bin sizes
    import re
    expected_bin_match = re.search(r'bin\s*\([^,]+,\s*([^)]+)\)', expected_kql, re.IGNORECASE)
    generated_bin_match = re.search(r'bin\s*\([^,]+,\s*([^)]+)\)', generated_kql, re.IGNORECASE)
    
    different_bin_sizes = False
    if expected_bin_match and generated_bin_match:
        expected_bin_size = expected_bin_match.group(1).strip()
        generated_bin_size = generated_bin_match.group(1).strip()
        if expected_bin_size != generated_bin_size:
            different_bin_sizes = True
            print(f"[SCORING] Detected different bin sizes: expected={expected_bin_size}, generated={generated_bin_size}")
    
    # 1. Schema match using query_evaluations comparator
    schema_match = compare_schema(expected_columns, generated_columns, strict_order=False)
    schema_score = schema_match.get("details", {}).get("overlap_ratio", 0.0)
    
    # If fields weren't explicitly requested, don't penalize for extra columns
    if not fields_explicitly_requested:
        missing_cols = schema_match.get("details", {}).get("missing", [])
        extra_cols = schema_match.get("details", {}).get("extra", [])
        
        if extra_cols and not missing_cols:
            # Generated query has all expected columns plus more - give full score
            schema_score = 1.0
            print(f"[SCORING] Extra columns present but fields not explicitly requested - no penalty")
        elif extra_cols and missing_cols:
            # Some expected columns missing, but has extras - only penalize for missing
            exp_count = len(expected_columns)
            matched_count = exp_count - len(missing_cols)
            schema_score = matched_count / exp_count if exp_count > 0 else 1.0
            print(f"[SCORING] Fields not explicitly requested - scoring based only on expected columns present")
    
    # For aggregation queries, if column counts match, boost schema score
    # (aggregated column names often differ but semantics are preserved)
    if is_aggregation and len(expected_columns) == len(generated_columns):
        # Give partial credit if column counts match even if names differ
        if schema_score < 0.5:
            schema_score = max(schema_score, 0.7)  # Boost to 0.7 minimum for matching column count
    
    # 2. Rows match using query_evaluations comparator (order-insensitive)
    # Convert dict results to list-of-lists format
    def dict_to_rows(results: List[Dict], columns: List[str]) -> List[List[str]]:
        rows = []
        for row_dict in results:
            row = [str(row_dict.get(col, "")) for col in columns]
            rows.append(row)
        return rows
    
    # For aggregation with different column names, align by position instead of name
    if is_aggregation and expected_results and generated_results and len(expected_columns) == len(generated_columns):
        # Extract values by position, not by column name
        expected_rows = [[str(v) for v in row_dict.values()] for row_dict in expected_results]
        generated_rows = [[str(v) for v in row_dict.values()] for row_dict in generated_results]
    elif not fields_explicitly_requested and len(generated_columns) > len(expected_columns):
        # If fields weren't explicitly requested and generated has extra columns,
        # only compare the expected columns (ignore extra columns in comparison)
        expected_rows = dict_to_rows(expected_results, expected_columns) if expected_results else []
        generated_rows = dict_to_rows(generated_results, expected_columns) if generated_results else []
        print(f"[SCORING] Comparing only expected columns ({len(expected_columns)}) since fields not explicitly requested")
    else:
        expected_rows = dict_to_rows(expected_results, expected_columns) if expected_results else []
        generated_rows = dict_to_rows(generated_results, generated_columns) if generated_results else []
    
    # If both queries use bin() with different sizes, mark rows_match as N/A
    rows_score_na = False
    if different_bin_sizes:
        rows_score = None  # Mark as N/A
        rows_score_na = True
        rows_match = {"match": False, "details": {"note": "Different bin sizes - results not comparable"}}
    else:
        # Custom row count comparison with 1% tolerance that penalizes both missing and extra rows
        expected_count = len(expected_rows)
        generated_count = len(generated_rows)
        
        # Check if row counts are within 1% tolerance
        count_tolerance = max(1, int(expected_count * 0.01))  # At least 1 row tolerance
        count_diff = abs(expected_count - generated_count)
        
        if count_diff <= count_tolerance:
            # Counts are close enough, use standard row comparison
            rows_match = compare_rows(expected_rows, generated_rows, strict_order=False)
            rows_score = rows_match.get("details", {}).get("overlap_ratio", 0.0)
        else:
            # Row counts differ significantly - penalize proportionally
            rows_match = compare_rows(expected_rows, generated_rows, strict_order=False)
            base_overlap = rows_match.get("details", {}).get("overlap_ratio", 0.0)
            
            # Penalty factor based on how far off the count is
            # If generated has more rows: penalty = expected / generated
            # If generated has fewer rows: penalty = generated / expected
            if expected_count > 0:
                count_penalty = min(expected_count, generated_count) / max(expected_count, generated_count)
            else:
                count_penalty = 0.0
            
            # Apply both overlap quality and count penalty
            rows_score = base_overlap * count_penalty
            
            rows_match["details"]["count_penalty"] = count_penalty
            rows_match["details"]["expected_count"] = expected_count
            rows_match["details"]["generated_count"] = generated_count
            rows_match["details"]["count_diff"] = count_diff
    
    # 3. Structural similarity using query_evaluations kql_parser
    def normalize_kql(kql: str) -> str:
        """Simple normalization for comparison."""
        import re
        s = kql.replace("\r\n", "\n").replace("\r", "\n")
        lines = [ln for ln in s.split("\n") if not re.match(r"^\s*//", ln)]
        return " ".join(lines).strip()
    
    normalized_expected = normalize_kql(expected_kql)
    normalized_generated = normalize_kql(generated_kql)
    
    semantic_comparison = compare_kql_semantic(normalized_expected, normalized_generated, prompt)
    structural_score = semantic_comparison.get('similarity', 0.0)
    
    # 4. LLM grading (uses same Azure OpenAI deployment as query generation)
    llm_score, llm_details = calculate_llm_graded_score(generated_kql, expected_kql, prompt)
    
    # Combine schema and rows into results_match (weighted average within the component)
    # Results match: 50% schema + 50% rows (when rows available)
    if rows_score_na:
        # If rows N/A due to different bin sizes, use only schema for results_match
        results_match_score = schema_score
        print(f"[SCORING] rows_match=N/A (different bin sizes), using schema_score only for results_match")
    else:
        # Combine schema and rows equally
        results_match_score = 0.5 * schema_score + 0.5 * rows_score
    
    # New weights: results_match (55%), structural (15%), LLM (30%)
    # Execution success is not part of the score - it's a gate
    weights = {
        "results_match": 0.55,
        "structural_similarity": 0.15,
        "llm_graded_similarity": 0.30,
    }
    
    # Build metrics dict for scoring with new structure
    metrics = {
        "results_match": results_match_score,
        "structural_similarity": structural_score,
        "llm_graded_similarity": llm_score,
    }
    
    # Calculate weighted score
    total_score = score_test(metrics, weights)
    
    # Success threshold: 0.9
    is_successful = total_score >= 0.9
    
    return {
        "total_score": round(total_score, 3),
        "is_successful": is_successful,
        "threshold": 0.9,
        "weights": weights,
        "components": {
            "results_match": {
                "score": round(results_match_score, 3),
                "weighted_score": round(results_match_score * weights["results_match"], 3),
                "weight": weights["results_match"],
                "details": {
                    "schema_score": round(schema_score, 3),
                    "rows_score": None if rows_score_na else round(rows_score, 3),
                    "rows_na": rows_score_na,
                    "schema_details": schema_match.get("details", {}),
                    "rows_details": rows_match.get("details", {})
                }
            },
            "structural_similarity": {
                "score": round(structural_score, 3),
                "weighted_score": round(structural_score * weights["structural_similarity"], 3),
                "weight": weights["structural_similarity"],
                "details": semantic_comparison.get("details", {})
            },
            "llm_grading": {
                "score": round(llm_score, 3),
                "weighted_score": round(llm_score * weights["llm_graded_similarity"], 3),
                "weight": weights["llm_graded_similarity"],
                "details": llm_details
            }
        }
    }
