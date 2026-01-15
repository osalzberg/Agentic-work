"""KQL parser for semantic query comparison."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace for easier parsing."""
    return re.sub(r"\s+", " ", text.strip())


class KqlStage:
    """Represents a stage in a KQL pipeline."""

    def __init__(self, raw: str):
        self.raw = raw.strip()
        self.stage_type = self._identify_type()
        self.details = self._extract_details()

    def _identify_type(self) -> str:
        """Identify the type of KQL operation."""
        lower = self.raw.lower()

        # Match keywords at the start
        if lower.startswith("where "):
            return "where"
        elif lower.startswith("project "):
            return "project"
        elif lower.startswith("summarize "):
            return "summarize"
        elif lower.startswith("extend "):
            return "extend"
        elif lower.startswith("join "):
            return "join"
        elif lower.startswith("sort by ") or lower.startswith("order by "):
            return "sort"
        elif lower.startswith("take ") or lower.startswith("limit "):
            return "take"
        elif lower.startswith("top "):
            return "top"
        elif lower.startswith("union "):
            return "union"
        elif lower.startswith("evaluate "):
            return "evaluate"
        elif lower.startswith("parse "):
            return "parse"
        elif lower.startswith("let "):
            return "let"
        elif lower.startswith("render "):
            return "render"
        else:
            # Likely a table name
            return "table"

    def _extract_details(self) -> Dict[str, Any]:
        """Extract semantic details from the stage."""
        if self.stage_type == "table":
            return {"table": self.raw.strip()}

        elif self.stage_type == "where":
            # Extract filter conditions
            conditions = self._extract_where_conditions(self.raw[6:])  # Remove "where "
            return {"conditions": conditions}

        elif self.stage_type == "project":
            # Extract projected columns
            columns = self._extract_columns(self.raw[8:])  # Remove "project "
            return {"columns": columns}

        elif self.stage_type == "summarize":
            # Extract aggregations and by clause
            content = self.raw[10:]  # Remove "summarize "
            by_match = re.search(r"\s+by\s+", content, re.IGNORECASE)
            if by_match:
                aggs = content[: by_match.start()].strip()
                by_clause = content[by_match.end() :].strip()
                return {
                    "aggregations": self._parse_aggregations(aggs),
                    "by": self._extract_columns(by_clause),
                }
            else:
                return {"aggregations": self._parse_aggregations(content)}

        elif self.stage_type in ("sort",):
            # Extract sort columns and directions
            content = self.raw
            if content.lower().startswith("sort by "):
                content = content[8:]
            elif content.lower().startswith("order by "):
                content = content[9:]
            return {"sort": self._parse_sort(content)}

        elif self.stage_type == "take":
            # Extract limit
            match = re.search(r"(\d+)", self.raw)
            return {"limit": int(match.group(1)) if match else None}

        elif self.stage_type == "top":
            # Extract top N and by column
            match = re.match(r"top\s+(\d+)\s+by\s+(.+)", self.raw, re.IGNORECASE)
            if match:
                return {"limit": int(match.group(1)), "by": match.group(2).strip()}
            return {}

        elif self.stage_type == "join":
            # Extract join details
            return {"join": self._parse_join(self.raw)}

        elif self.stage_type == "extend":
            # Extract extended columns
            return {"extends": self._parse_aggregations(self.raw[7:])}

        elif self.stage_type == "evaluate":
            # Extract evaluate function
            return {"function": self.raw[9:].strip()}

        return {}

    def _extract_where_conditions(self, content: str) -> List[str]:
        """Extract individual where conditions."""
        # Normalize for comparison
        content = normalize_whitespace(content)

        # Split by 'and' and 'or' (simple approach)
        # Note: This is simplified and may not handle all cases perfectly
        conditions = []
        parts = re.split(r"\s+(and|or)\s+", content, flags=re.IGNORECASE)

        for i, part in enumerate(parts):
            if i % 2 == 0:  # Skip the 'and'/'or' separators
                normalized = self._normalize_condition(part.strip())
                if normalized:
                    conditions.append(normalized)

        return conditions

    def _normalize_condition(self, cond: str) -> str:
        """Normalize a condition for comparison."""
        # Remove extra spaces
        cond = normalize_whitespace(cond)
        # Normalize operators
        cond = re.sub(r"\s*==\s*", "==", cond)
        cond = re.sub(r"\s*!=\s*", "!=", cond)
        cond = re.sub(r"\s*>=\s*", ">=", cond)
        cond = re.sub(r"\s*<=\s*", "<=", cond)
        cond = re.sub(r"\s*>\s*", ">", cond)
        cond = re.sub(r"\s*<\s*", "<", cond)
        cond = re.sub(r"\s*=\s*", "=", cond)

        # Normalize time range operators: treat >= as > and <= as < for comparison purposes
        # This is semantically reasonable for time ranges where granularity makes them equivalent
        if (
            "ago(" in cond.lower()
            or "datetime(" in cond.lower()
            or "now(" in cond.lower()
            or "timegenerated" in cond.lower()
        ):
            cond = cond.replace(">=", ">")
            cond = cond.replace("<=", "<")

        # Lowercase keywords
        cond = re.sub(
            r"\b(has|contains|in|between|startswith|endswith)\b",
            lambda m: m.group(0).lower(),
            cond,
            flags=re.IGNORECASE,
        )
        return cond

    def _extract_columns(self, content: str) -> List[str]:
        """Extract column names from a comma-separated list."""
        # Handle expressions like "a=b, c, d=e"
        parts = []
        depth = 0
        current = []

        for char in content:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue
            current.append(char)

        if current:
            parts.append("".join(current).strip())

        # Normalize column names (extract just the name, ignoring aliases)
        columns = []
        for part in parts:
            # If "alias=expr", take the alias
            if "=" in part:
                alias = part.split("=")[0].strip()
                columns.append(alias)
            else:
                columns.append(part.strip())

        return columns

    def _parse_aggregations(self, content: str) -> List[str]:
        """Parse aggregation expressions."""
        # Similar to columns but keep full expressions
        parts = []
        depth = 0
        current = []

        for char in content:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 0:
                expr = "".join(current).strip()
                if expr:
                    parts.append(normalize_whitespace(expr))
                current = []
                continue
            current.append(char)

        if current:
            expr = "".join(current).strip()
            if expr:
                parts.append(normalize_whitespace(expr))

        return parts

    def _parse_sort(self, content: str) -> List[Tuple[str, str]]:
        """Parse sort columns and directions."""
        columns = []
        parts = [p.strip() for p in content.split(",")]

        for part in parts:
            # Check for asc/desc
            if part.lower().endswith(" asc"):
                col = part[:-4].strip()
                columns.append((col, "asc"))
            elif part.lower().endswith(" desc"):
                col = part[:-5].strip()
                columns.append((col, "desc"))
            else:
                # Default is desc in KQL
                columns.append((part, "desc"))

        return columns

    def _parse_join(self, content: str) -> Dict[str, Any]:
        """Parse join details."""
        # Extract join kind
        kind_match = re.search(r"join\s+kind\s*=\s*(\w+)", content, re.IGNORECASE)
        kind = kind_match.group(1) if kind_match else "inner"

        # Extract table reference (simplified)
        table_match = re.search(r"\(\s*(\w+)", content)
        table = table_match.group(1) if table_match else None

        # Extract on conditions
        on_match = re.search(r"on\s+(.+)$", content, re.IGNORECASE)
        on_conditions = []
        if on_match:
            on_clause = on_match.group(1).strip()
            # Parse conditions (simplified)
            on_conditions = [
                normalize_whitespace(c.strip()) for c in on_clause.split(",")
            ]

        return {"kind": kind, "table": table, "on": on_conditions}


def parse_kql(query: str) -> List[KqlStage]:
    """Parse a KQL query into semantic stages."""
    # Normalize line breaks
    query = query.replace("\r\n", "\n").replace("\r", "\n")

    # Remove comments
    lines = [line for line in query.split("\n") if not line.strip().startswith("//")]
    query = "\n".join(lines)

    # Split by pipe, handling nested parentheses
    stages = []
    depth = 0
    current = []

    for char in query:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "|" and depth == 0:
            stage_text = "".join(current).strip()
            if stage_text:
                stages.append(KqlStage(stage_text))
            current = []
            continue
        current.append(char)

    if current:
        stage_text = "".join(current).strip()
        if stage_text:
            stages.append(KqlStage(stage_text))

    return stages


def compare_kql_semantic(
    expected: str, generated: str, prompt: str = ""
) -> Dict[str, Any]:
    """
    Compare two KQL queries semantically.

    Args:
        expected: Expected KQL query
        generated: Generated KQL query
        prompt: Natural language prompt (used to determine if specific fields were requested)

    Returns a dict with:
    - similarity: float 0-1
    - details: breakdown of what matched/differed
    """
    exp_stages = parse_kql(expected)
    gen_stages = parse_kql(generated)

    result = {
        "similarity": 0.0,
        "details": {
            "expected_stages": len(exp_stages),
            "generated_stages": len(gen_stages),
            "matches": {},
            "differences": [],
            "prompt": prompt,
        },
    }

    # 1. Compare table names (must match)
    exp_tables = [s for s in exp_stages if s.stage_type == "table"]
    gen_tables = [s for s in gen_stages if s.stage_type == "table"]

    table_match = False
    if exp_tables and gen_tables:
        exp_table = exp_tables[0].details.get("table", "").lower()
        gen_table = gen_tables[0].details.get("table", "").lower()
        table_match = exp_table == gen_table
        result["details"]["table_match"] = table_match

    if not table_match and exp_tables:
        result["details"]["differences"].append("Different source tables")
        return result  # Early exit if tables don't match

    # 2. Group stages by type
    exp_by_type = {}
    gen_by_type = {}

    for stage in exp_stages:
        if stage.stage_type != "table":
            exp_by_type.setdefault(stage.stage_type, []).append(stage)

    for stage in gen_stages:
        if stage.stage_type != "table":
            gen_by_type.setdefault(stage.stage_type, []).append(stage)

    # 2.5. Detect top <-> sort+take equivalence
    # If one query uses 'top N by X' and the other uses 'sort by X | take N', treat them as equivalent
    top_sort_take_equivalent = False

    # Case 1: Expected has 'top', Generated has 'sort' + 'take' (but no 'top')
    if "top" in exp_by_type and "top" not in gen_by_type:
        if "sort" in gen_by_type and "take" in gen_by_type:
            # Check if they're semantically equivalent
            exp_top = exp_by_type["top"][0].details
            gen_take = gen_by_type["take"][0].details
            gen_sorts = []
            for stage in gen_by_type["sort"]:
                gen_sorts.extend(stage.details.get("sort", []))

            top_limit = exp_top.get("limit")
            top_by = exp_top.get("by", "").lower().strip()
            take_limit = gen_take.get("limit")

            # Check if limits match and sort column matches top's by column
            if top_limit == take_limit and gen_sorts:
                # Check if top's 'by' column matches any sort column
                for sort_col, _ in gen_sorts:
                    if sort_col.lower().strip() == top_by:
                        top_sort_take_equivalent = True
                        break

    # Case 2: Generated has 'top', Expected has 'sort' + 'take' (but no 'top')
    if "top" in gen_by_type and "top" not in exp_by_type:
        if "sort" in exp_by_type and "take" in exp_by_type:
            # Check if they're semantically equivalent
            gen_top = gen_by_type["top"][0].details
            exp_take = exp_by_type["take"][0].details
            exp_sorts = []
            for stage in exp_by_type["sort"]:
                exp_sorts.extend(stage.details.get("sort", []))

            top_limit = gen_top.get("limit")
            top_by = gen_top.get("by", "").lower().strip()
            take_limit = exp_take.get("limit")

            # Check if limits match and sort column matches top's by column
            if top_limit == take_limit and exp_sorts:
                # Check if top's 'by' column matches any sort column
                for sort_col, _ in exp_sorts:
                    if sort_col.lower().strip() == top_by:
                        top_sort_take_equivalent = True
                        break

    # 3. Compare each stage type
    scores = []
    weights = {
        "where": 0.30,
        "project": 0.15,
        "summarize": 0.25,
        "sort": 0.10,
        "join": 0.15,
        "extend": 0.10,
        "take": 0.05,
        "top": 0.05,
        "evaluate": 0.10,
    }

    all_types = set(exp_by_type.keys()) | set(gen_by_type.keys())

    for stage_type in all_types:
        exp_list = exp_by_type.get(stage_type, [])
        gen_list = gen_by_type.get(stage_type, [])

        # Skip penalizing 'top', 'sort', or 'take' if they're equivalent patterns
        if top_sort_take_equivalent and stage_type in ("top", "sort", "take"):
            # Give full score for these stages when pattern is equivalent
            scores.append((weights.get(stage_type, 0.05), 1.0))
            continue

        if not exp_list and gen_list:
            # Extra stages in generated
            # Special case: if it's just "order by TimeGenerated desc", don't penalize
            if stage_type == "sort" and _is_default_time_sort(gen_list):
                # Don't penalize default time ordering
                scores.append((weights.get(stage_type, 0.05), 1.0))
            else:
                result["details"]["differences"].append(
                    f"Extra {stage_type} stage(s) in generated"
                )
                scores.append((weights.get(stage_type, 0.05), 0.0))
        elif exp_list and not gen_list:
            # Missing stages in generated
            result["details"]["differences"].append(
                f"Missing {stage_type} stage(s) in generated"
            )
            scores.append((weights.get(stage_type, 0.05), 0.0))
        else:
            # Compare the stages
            stage_score = _compare_stage_list(
                stage_type, exp_list, gen_list, result["details"]
            )
            scores.append((weights.get(stage_type, 0.05), stage_score))

    # 4. Calculate weighted similarity
    if scores:
        total_weight = sum(w for w, _ in scores)
        if total_weight > 0:
            weighted_sum = sum(w * s for w, s in scores)
            result["similarity"] = weighted_sum / total_weight
        else:
            result["similarity"] = 1.0  # No comparable stages
    else:
        result["similarity"] = 1.0  # Empty queries

    return result


def _compare_stage_list(
    stage_type: str, exp_list: List[KqlStage], gen_list: List[KqlStage], details: Dict
) -> float:
    """Compare lists of stages of the same type."""

    if stage_type == "where":
        return _compare_where_stages(exp_list, gen_list, details)
    elif stage_type == "project":
        prompt = details.get("prompt", "")
        return _compare_project_stages(exp_list, gen_list, details, prompt)
    elif stage_type == "summarize":
        return _compare_summarize_stages(exp_list, gen_list, details)
    elif stage_type == "sort":
        return _compare_sort_stages(exp_list, gen_list, details)
    elif stage_type == "join":
        return _compare_join_stages(exp_list, gen_list, details)
    else:
        # Generic comparison: just check if counts match
        return 1.0 if len(exp_list) == len(gen_list) else 0.5


def _compare_where_stages(
    exp_list: List[KqlStage], gen_list: List[KqlStage], details: Dict
) -> float:
    """Compare where clauses."""
    # Collect all conditions
    exp_conditions = []
    for stage in exp_list:
        exp_conditions.extend(stage.details.get("conditions", []))

    gen_conditions = []
    for stage in gen_list:
        gen_conditions.extend(stage.details.get("conditions", []))

    if not exp_conditions:
        return 1.0

    # Check condition matches (order-independent for now)
    exp_set = set(exp_conditions)
    gen_set = set(gen_conditions)

    matched = exp_set & gen_set
    missing = exp_set - gen_set
    extra = gen_set - exp_set

    if missing:
        details["differences"].append(f"Missing where conditions: {missing}")
    if extra:
        details["differences"].append(f"Extra where conditions: {extra}")

    # Score: ratio of matched conditions
    total = len(exp_set | gen_set)
    score = len(matched) / total if total > 0 else 1.0

    # Bonus for matching order
    if score > 0.8 and exp_conditions == gen_conditions:
        score = 1.0
    elif score > 0.8:
        # Slight penalty for different order
        score = min(1.0, score + 0.1)

    details["matches"]["where"] = {
        "expected": len(exp_conditions),
        "generated": len(gen_conditions),
        "matched": len(matched),
        "score": score,
    }

    return score


def _compare_project_stages(
    exp_list: List[KqlStage], gen_list: List[KqlStage], details: Dict, prompt: str = ""
) -> float:
    """Compare project clauses.

    Only penalizes missing fields if they were explicitly requested in the prompt.
    """
    # Order doesn't matter for projected columns
    exp_columns = set()
    for stage in exp_list:
        exp_columns.update(stage.details.get("columns", []))

    gen_columns = set()
    for stage in gen_list:
        gen_columns.update(stage.details.get("columns", []))

    if not exp_columns:
        return 1.0

    # Normalize column names for comparison
    exp_norm = {c.lower() for c in exp_columns}
    gen_norm = {c.lower() for c in gen_columns}

    matched = exp_norm & gen_norm
    missing = exp_norm - gen_norm
    extra = gen_norm - exp_norm

    # Check if prompt explicitly mentions specific fields/columns
    prompt_lower = prompt.lower()
    fields_explicitly_requested = any(
        keyword in prompt_lower
        for keyword in [
            "show",
            "display",
            "project",
            "select",
            "return",
            "get",
            "include",
        ]
    ) and any(
        field_keyword in prompt_lower
        for field_keyword in ["field", "column", "property", "attribute"]
    )

    # Only penalize missing columns if they were explicitly requested in the prompt
    if missing:
        if fields_explicitly_requested:
            # Check if any missing column is mentioned in the prompt
            missing_and_requested = {
                col
                for col in missing
                if col in prompt_lower or col.replace("_", " ") in prompt_lower
            }
            if missing_and_requested:
                details["differences"].append(
                    f"Missing explicitly requested columns: {missing_and_requested}"
                )
        else:
            # Prompt didn't explicitly request fields, don't penalize heavily
            details["differences"].append(
                f"Note: Missing projected columns (not explicitly requested): {missing}"
            )

    if extra:
        # Extra fields are generally fine - they provide more information
        details["differences"].append(f"Extra projected columns: {extra}")

    # Calculate score
    # If no fields explicitly requested, score based only on matched columns (don't penalize missing)
    if not fields_explicitly_requested and missing:
        # Don't penalize for missing columns if they weren't requested
        score = 1.0 if matched else 0.8  # Small penalty only if no overlap at all
    else:
        # Standard scoring when fields were explicitly requested
        total = len(exp_norm | gen_norm)
        score = len(matched) / total if total > 0 else 1.0

    details["matches"]["project"] = {
        "expected": len(exp_columns),
        "generated": len(gen_columns),
        "matched": len(matched),
        "score": score,
        "fields_explicitly_requested": fields_explicitly_requested,
    }

    return score


def _compare_summarize_stages(
    exp_list: List[KqlStage], gen_list: List[KqlStage], details: Dict
) -> float:
    """Compare summarize clauses."""
    if len(exp_list) != len(gen_list):
        details["differences"].append(f"Different number of summarize stages")
        return 0.5

    # Compare aggregations and by clauses
    exp = exp_list[0].details
    gen = gen_list[0].details

    exp_aggs = set(exp.get("aggregations", []))
    gen_aggs = set(gen.get("aggregations", []))

    exp_by = set(c.lower() for c in exp.get("by", []))
    gen_by = set(c.lower() for c in gen.get("by", []))

    # Score aggregations
    agg_matched = len(exp_aggs & gen_aggs)
    agg_total = len(exp_aggs | gen_aggs)
    agg_score = agg_matched / agg_total if agg_total > 0 else 1.0

    # Score by clause
    by_matched = len(exp_by & gen_by)
    by_total = len(exp_by | gen_by)
    by_score = by_matched / by_total if by_total > 0 else 1.0

    # Combined score (aggregations more important)
    score = 0.6 * agg_score + 0.4 * by_score

    details["matches"]["summarize"] = {
        "agg_score": agg_score,
        "by_score": by_score,
        "score": score,
    }

    return score


def _is_default_time_sort(gen_list: List[KqlStage]) -> bool:
    """Check if the generated sort is just 'order by TimeGenerated desc' (default time ordering)."""
    if not gen_list:
        return False

    # Collect all sort columns
    all_sorts = []
    for stage in gen_list:
        all_sorts.extend(stage.details.get("sort", []))

    # Check if it's exactly one sort: TimeGenerated desc (or desc is omitted since it's default)
    if len(all_sorts) == 1:
        col, direction = all_sorts[0]
        return col.lower() == "timegenerated" and direction.lower() in ("desc", "")

    return False


def _compare_sort_stages(
    exp_list: List[KqlStage], gen_list: List[KqlStage], details: Dict
) -> float:
    """Compare sort clauses."""
    # Order matters for sort
    exp_sorts = []
    for stage in exp_list:
        exp_sorts.extend(stage.details.get("sort", []))

    gen_sorts = []
    for stage in gen_list:
        gen_sorts.extend(stage.details.get("sort", []))

    if not exp_sorts:
        return 1.0

    # Exact match
    if exp_sorts == gen_sorts:
        return 1.0

    # Partial match: check if columns match regardless of direction
    exp_cols = [col.lower() for col, _ in exp_sorts]
    gen_cols = [col.lower() for col, _ in gen_sorts]

    if exp_cols == gen_cols:
        # Same columns, might differ in direction
        return 0.9

    # Check overlap
    matched = len(set(exp_cols) & set(gen_cols))
    total = max(len(exp_cols), len(gen_cols))
    score = matched / total if total > 0 else 1.0

    details["matches"]["sort"] = {"score": score}
    return score


def _compare_join_stages(
    exp_list: List[KqlStage], gen_list: List[KqlStage], details: Dict
) -> float:
    """Compare join clauses."""
    if len(exp_list) != len(gen_list):
        return 0.5

    exp_join = exp_list[0].details.get("join", {})
    gen_join = gen_list[0].details.get("join", {})

    # Compare join kind
    kind_match = exp_join.get("kind", "inner") == gen_join.get("kind", "inner")

    # Compare table
    table_match = exp_join.get("table", "").lower() == gen_join.get("table", "").lower()

    # Compare on conditions (order-independent)
    exp_on = set(exp_join.get("on", []))
    gen_on = set(gen_join.get("on", []))

    on_matched = len(exp_on & gen_on)
    on_total = len(exp_on | gen_on)
    on_score = on_matched / on_total if on_total > 0 else 1.0

    # Combined score
    score = (
        0.3 * (1.0 if kind_match else 0.5)
        + 0.3 * (1.0 if table_match else 0.0)
        + 0.4 * on_score
    )

    details["matches"]["join"] = {"score": score}
    return score
