from __future__ import annotations

from typing import List

from azure_openai_utils import OpenAIClient


def explain_kql(kql: str) -> str:
    """Explain KQL via LLM only.

    Uses Azure OpenAI (configured via environment) to produce a concise, neutral explanation
    focused on: source table, filters, projections, derived columns, aggregations and grouping,
    ordering, and limiting. If the LLM response is unavailable, return an empty string.
    """
    if not kql or not kql.strip():
        return ""
    try:
        client = OpenAIClient()
        system = (
            "You are a precise KQL explainer. Given a KQL query, write a concise, neutral description "
            "covering: source table, filters, selected/derived columns, aggregations + grouping, ordering, and limits. "
            "Prefer readable paraphrase over verbatim code. No code fences. Max 3 short sentences."
        )
        res = client.explain(system_prompt=system, summary=kql, max_tokens=400, temperature=0.2, top_p=0.9, allow_escalation=True)
        return (res.content or "").strip()
    except Exception:
        return ""


def grade_kql_similarity(expected_kql: str, generated_kql: str) -> float | None:
    """Grade the semantic similarity between two KQL queries using LLM.

    Asks the LLM to compare the queries and return a similarity score from 0.0 (completely different)
    to 1.0 (identical), while ignoring differences in column names created by projections, 
    aggregations, or alias assignments.
    
    Returns:
        float: Similarity score between 0.0 and 1.0, or None if grading fails
    """
    if not expected_kql or not expected_kql.strip():
        return None
    if not generated_kql or not generated_kql.strip():
        return None
    
    try:
        client = OpenAIClient()
        system = (
            "You are an expert KQL query evaluator. Compare two KQL queries and grade their semantic similarity "
            "on a scale from 0.0 (completely different) to 1.0 (identical). "
            "Ignore differences in:\n"
            "- Column names or aliases (e.g., 'Name' vs 'PodName', 'count()' vs 'LogCount')\n"
            "- Whitespace and formatting\n"
            "- Comment differences\n"
            "\n"
            "Focus on semantic equivalence:\n"
            "- Same source table(s)\n"
            "- Same filtering logic\n"
            "- Same aggregation operations (even if column names differ)\n"
            "- Same grouping logic\n"
            "- Same ordering logic\n"
            "- Same limiting/top logic\n"
            "\n"
            "Return ONLY a single decimal number between 0.0 and 1.0, nothing else."
        )
        
        prompt = f"Expected query:\n{expected_kql}\n\nGenerated query:\n{generated_kql}\n\nSimilarity score:"
        
        res = client.explain(system_prompt=system, summary=prompt, max_tokens=500, temperature=0.0, top_p=0.9, allow_escalation=True)
        content = (res.content or "").strip()
        
        # Parse the score
        try:
            score = float(content)
            # Clamp to valid range
            return max(0.0, min(1.0, score))
        except ValueError:
            print(f"[LLM Grading] Failed to parse score from response: {content}")
            return None
            
    except Exception as e:
        print(f"[LLM Grading] Exception during grading: {type(e).__name__}: {e}")
        return None
