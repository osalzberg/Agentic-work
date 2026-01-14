# Query Scoring System

## Overview
The batch testing system now includes a comprehensive scoring mechanism that evaluates generated KQL queries against expected queries. A query is considered **successful** only if:
1. It executes without errors, AND
2. It achieves a **total score of 0.9 (90%) or higher**

<br>

## Scoring Components
The total score is calculated from 2 equally-weighted components, each contributing 50% to the final score:

### 1. Query Similarity Score (50%)
#### Purpose
Evaluate the generated query against the expected query. Evaludation is done by LLM

#### Process
1. Sends both queries and the user prompt to the LLM
2. LLM evaluates correctness, logic, and efficiency
3. Returns a score (0-1) with reasoning
**Evaluation Criteria:**
- Correct table selection
- Appropriate filters
- Proper aggregations/transformations
- Expected result equivalence
- Query logic and efficiency

<br>

### 2. Results Match Score (50%)
#### Purpose
Compares the actual data returned by both queries, including both schema (columns) and row content.

#### Calculation
- Schema (columns) are compared for overlap; extra columns do not penalize the score
- Row-by-row comparison (order-independent)
- Field order within rows doesn't matter
- Normalizes values (lowercase keys, string values) for comparison
- Counts how many expected rows are found in generated results

<br>

### Total Score Calculation

```
total_score = (query_similarity_score * 0.5) + 
              (results_match_score * 0.5)
```


**Success Threshold:** total_score >= 0.9 (90%)

<br><br>

## Example Scenarios
### Scenario 1: Exact Match
```
Generated: Traces | where timestamp > ago(1h) | summarize count() by level
Expected:  Traces | where timestamp > ago(1h) | summarize count() by level

* Query Similarity: 100% (exact match)
* Results Match: 100% (identical data and columns)
* Total: 100% → PASS
```

<br>

### Scenario 2: Minor Differences
```
Generated: Traces | where timestamp > ago(1h) | project timestamp, level, message
Expected:  Traces | where timestamp > ago(60m) | project timestamp, level, message

* Query Similarity: 95% (recognizes equivalence)
* Results Match: 100% (same data)
* Total: 97.5% → PASS
```

<br>

### Scenario 3: Wrong Query
```
Generated: AppTraces | where timestamp > ago(1h) | summarize count()
Expected:  Traces | where timestamp > ago(1h) | summarize count() by level

* Query Similarity: 60% (wrong approach)
* Results Match: 30% (different data structure)
* Total: 45% → FAIL
```
