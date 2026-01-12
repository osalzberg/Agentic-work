# Query Scoring System

## Overview
The batch testing system now includes a comprehensive scoring mechanism that evaluates generated KQL queries against expected queries. A query is considered **successful** only if:
1. It executes without errors, AND
2. It achieves a **total score of 0.9 (90%) or higher**

## Scoring Components

The total score is calculated from 4 equally-weighted components, each contributing 25% (0.25) to the final score:

### 1. Schema Match Score (25%)
**Purpose:** Compares the column structure of query results

**Calculation:**
- **Exact match** = 1.0 (full points)
- **Missing fields** reduce the score proportionally based on how many expected columns are missing
- **Extra fields** do NOT reduce the score (superset is acceptable)

**Formula:**
```
score = 1.0 - (missing_fields_count / expected_fields_count)
```

**Example:**
- Expected columns: `['timestamp', 'level', 'message']`
- Generated columns: `['timestamp', 'level', 'message', 'source']`
- Score: 1.0 (all expected fields present, extra field 'source' doesn't reduce score)

### 2. Semantic Similarity Score (25%)
**Purpose:** Compares the KQL query structure and logic

**Components:**
- **Tables** (40% weight): Checks if the correct tables are queried
- **Filters** (30% weight): Compares the number and type of where clauses
- **Aggregations** (30% weight): Compares aggregation functions (count, sum, avg, etc.)

**Extraction Logic:**
- Tables: Extracted from query start and after pipes
- Filters: Extracted from `| where` clauses
- Aggregations: Detected by function keywords (count, sum, avg, etc.) and `| summarize` operators

### 3. Results Match Score (25%)
**Purpose:** Compares the actual data returned by both queries

**Calculation:**
- Row-by-row comparison (order-independent)
- Field order within rows doesn't matter
- Normalizes values (lowercase keys, string values) for comparison
- Counts how many expected rows are found in generated results

**Formula:**
```
score = matching_rows / total_expected_rows
```

**Note:** Only the first 100 rows are compared for performance reasons

### 4. LLM Grading Score (25%)
**Purpose:** Uses AI to evaluate query quality holistically

**Process:**
1. Sends both queries and the user prompt to the LLM
2. LLM evaluates correctness, logic, and efficiency
3. Returns a score (0-1) with reasoning

**Evaluation Criteria:**
- Correct table selection
- Appropriate filters
- Proper aggregations/transformations
- Expected result equivalence
- Query logic and efficiency

## Total Score Calculation

```
total_score = (schema_match * 0.25) + 
              (semantic_similarity * 0.25) + 
              (results_match * 0.25) + 
              (llm_grading * 0.25)
```

**Success Threshold:** total_score >= 0.9 (90%)

## UI Display

### Live Results
During batch processing, each query shows:
```
Query: <generated KQL>
Query executed (results: X)
Score: 92.5% (PASS ≥90%)
Schema: 100% | Semantic: 85% | Results: 95% | LLM: 90%
```

### Batch Summary

**Success Section:**
- Shows queries with score >= 0.9
- Displays full score breakdown
- Green border and checkmark

**Error Section:**
- Shows queries with score < 0.9 OR execution errors
- Displays score breakdown (if query executed)
- Shows LLM reasoning for why it failed
- Red border and X mark

## Report Files

Both JSON and Excel reports include:
- `Total Score`: Overall score (0-1)
- `Score Status`: PASS or FAIL
- `Schema Match Score`: Component score (0-1)
- `Semantic Similarity Score`: Component score (0-1)
- `Results Match Score`: Component score (0-1)
- `LLM Grading Score`: Component score (0-1)
- `LLM Reasoning`: Explanation from LLM grader

## Example Scenarios

### Scenario 1: Perfect Match
```
Generated: Traces | where timestamp > ago(1h) | summarize count() by level
Expected:  Traces | where timestamp > ago(1h) | summarize count() by level

Schema Match: 100% (exact columns)
Semantic: 100% (same structure)
Results: 100% (identical data)
LLM: 100% (perfect match)
Total: 100% → PASS
```

### Scenario 2: Minor Differences
```
Generated: Traces | where timestamp > ago(1h) | project timestamp, level, message
Expected:  Traces | where timestamp > ago(60m) | project timestamp, level, message

Schema Match: 100% (same columns)
Semantic: 90% (different time syntax but equivalent)
Results: 100% (same data)
LLM: 95% (recognizes equivalence)
Total: 96.25% → PASS
```

### Scenario 3: Failed Query
```
Generated: AppTraces | where timestamp > ago(1h) | summarize count()
Expected:  Traces | where timestamp > ago(1h) | summarize count() by level

Schema Match: 50% (missing 'level' column)
Semantic: 80% (wrong table, missing grouping)
Results: 30% (different data structure)
LLM: 60% (wrong approach)
Total: 55% → FAIL
```

## API Endpoint

**POST /api/score-query**

Request:
```json
{
  "generated_kql": "string",
  "expected_kql": "string",
  "prompt": "string",
  "generated_columns": ["col1", "col2"],
  "expected_columns": ["col1", "col2"],
  "generated_results": [{"col1": "val1", "col2": "val2"}],
  "expected_results": [{"col1": "val1", "col2": "val2"}],
  "model": "gpt-4o-mini"
}
```

Response:
```json
{
  "success": true,
  "score": {
    "total_score": 0.925,
    "is_successful": true,
    "threshold": 0.9,
    "components": {
      "schema_match": {
        "score": 1.0,
        "weighted_score": 0.25,
        "weight": 0.25,
        "details": {...}
      },
      ...
    }
  }
}
```

## Implementation Files

- **query_scorer.py**: Core scoring logic
- **web_app.py**: API endpoint `/api/score-query`
- **templates/index.html**: UI integration for batch testing
- Reports: Enhanced with score fields in JSON and Excel formats
