from __future__ import annotations

from typing import Dict

# Scoring rubric mirrors PRD section 8

DEFAULT_WEIGHTS = {
    "result_equality": 0.05,
    "exec_success": 0.15,
    "schema_match_score": 0.25,
    "rows_match_score": 0.25,
    "structural_similarity": 0.20,
    "llm_graded_similarity": 0.10,
}


def score_test(metrics: Dict[str, float], weights: Dict[str, float] = DEFAULT_WEIGHTS) -> float:
    total = 0.0
    max_score = sum(weights.values())
    for k, w in weights.items():
        total += w * float(metrics.get(k, 0.0))
    # Must-pass gates can zero the score externally.
    total /= max_score
    return max(0.0, min(1.0, total))
