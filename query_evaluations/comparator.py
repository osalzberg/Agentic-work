from __future__ import annotations

import csv
from typing import Any, Dict, List, Tuple, Optional


def load_csv_snapshot(path: str) -> Tuple[List[str], List[List[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], []
    header = rows[0]
    data = rows[1:]
    return header, data


def _norm_col(name: str) -> str:
    """Normalize column name: lowercase, remove spaces/hyphens/underscores, strip plural 's'."""
    s = (name or "")
    s = s.lower()
    # remove spaces, hyphens, and underscores
    s = s.replace(" ", "").replace("-", "").replace("_", "")
    # strip trailing 's' to handle plural/singular differences
    if s.endswith('s') and len(s) > 1:
        s = s[:-1]
    return s


def compare_schema(expected_cols: List[str], actual_cols: List[str], strict_order: bool = True) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "match": False,
        "expected": expected_cols,
        "actual": actual_cols,
        "details": {}
    }
    # Normalize column names for comparisons (ignore case, hyphens, underscores)
    exp_norm_list = [_norm_col(c) for c in expected_cols]
    act_norm_list = [_norm_col(c) for c in actual_cols]

    # Binary match on normalized columns
    if strict_order:
        result["match"] = exp_norm_list == act_norm_list
    else:
        result["match"] = set(exp_norm_list) == set(act_norm_list)

    # Calibrated overlap scoring (order-insensitive): fraction of expected columns present in actual
    exp_set = set(exp_norm_list)
    act_set = set(act_norm_list)
    overlap = exp_set.intersection(act_set)
    overlap_count = len(overlap)
    exp_count = len(exp_set) or 1
    overlap_ratio = overlap_count / float(exp_count)
    missing_norm = list(exp_set - act_set)
    extra_norm = list(act_set - exp_set)

    # Fuzzy match missing vs extra: if a missing column is a substring of an extra (or vice versa), consider them matched
    matched_pairs = []
    remaining_missing = []
    remaining_extra = list(extra_norm)
    
    for miss in missing_norm:
        matched = False
        for extra in remaining_extra:
            # Check if one is a substring of the other (e.g., "name" in "podname")
            if miss in extra or extra in miss:
                matched_pairs.append((miss, extra))
                remaining_extra.remove(extra)
                matched = True
                break
        if not matched:
            remaining_missing.append(miss)
    
    # Update overlap count to include fuzzy matches
    overlap_count += len(matched_pairs)
    overlap_ratio = overlap_count / float(exp_count)
    
    # Update match result: if all expected columns are accounted for (exact or fuzzy), it's a match
    if not strict_order:
        result["match"] = len(remaining_missing) == 0

    # Map missing/extra back to original names for readability (first occurrence wins)
    def _originals_for_norm(norm_keys: List[str], originals: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for nk in norm_keys:
            if nk in seen:
                continue
            for orig in originals:
                if _norm_col(orig) == nk:
                    out.append(orig)
                    seen.add(nk)
                    break
        return out

    missing = _originals_for_norm(remaining_missing, expected_cols)
    extra = _originals_for_norm(remaining_extra, actual_cols)

    result["details"] = {
        "overlap_count": overlap_count,
        "expected_count": exp_count,
        "overlap_ratio": overlap_ratio,
        "missing": missing,
        "extra": extra,
        "strict_order": strict_order,
    }
    return result


def compare_rows(expected_rows: List[List[str]], actual_rows: List[List[str]], strict_order: bool, tolerances: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    tolerances = tolerances or {}
    numeric_abs = float(tolerances.get("numeric_abs", 0.0))
    numeric_rel = float(tolerances.get("numeric_rel", 0.0))

    def _num_equal(a: str, b: str) -> bool:
        try:
            va = float(a)
            vb = float(b)
            if numeric_abs > 0 and abs(va - vb) <= numeric_abs:
                return True
            if numeric_rel > 0 and (abs(va - vb) / (abs(vb) + 1e-12)) <= numeric_rel:
                return True
            return va == vb
        except Exception:
            return a == b

    def _row_equal(r1: List[str], r2: List[str]) -> bool:
        if len(r1) != len(r2):
            return False
        for a, b in zip(r1, r2):
            if not _num_equal(a, b):
                return False
        return True

    exp_count = len(expected_rows)
    act_count = len(actual_rows)
    result: Dict[str, Any] = {
        "match": False,
        "expected_count": exp_count,
        "actual_count": act_count,
        "mismatches": 0,
        "details": {}
    }

    if strict_order:
        # Position-wise comparison; count matched rows
        matched = 0
        min_len = min(exp_count, act_count)
        mismatches = 0
        for r1, r2 in zip(expected_rows[:min_len], actual_rows[:min_len]):
            if _row_equal(r1, r2):
                matched += 1
            else:
                mismatches += 1
        # Any extra rows count as mismatches
        mismatches += abs(exp_count - act_count)
        result["mismatches"] = mismatches
        result["match"] = (mismatches == 0 and exp_count == act_count)
        overlap_ratio = (matched / float(exp_count or 1))
        result["details"] = {
            "matched_count": matched,
            "expected_count": exp_count,
            "actual_count": act_count,
            "overlap_ratio": overlap_ratio,
            "strict_order": True,
        }
        return result
    else:
        # Order-insensitive compare: treat rows as multisets with numeric tolerance aware equality
        from collections import Counter

        def _norm_row(r: List[str]) -> Tuple[str, ...]:
            # Normalize row by string values only; numeric tolerances are applied via _row_equal during matching
            return tuple(r)

        # Build counters for quick frequency comparison
        c_expected = Counter(_norm_row(r) for r in expected_rows)
        c_actual = Counter(_norm_row(r) for r in actual_rows)

        # Compute matched count as multiset intersection cardinality
        matched = 0
        for row, cnt in c_expected.items():
            matched += min(cnt, c_actual.get(row, 0))

        overlap_ratio = (matched / float(exp_count or 1))
        result["match"] = c_expected == c_actual
        result["mismatches"] = 0 if result["match"] else abs(sum(c_expected.values()) - sum(c_actual.values()))
        result["details"] = {
            "matched_count": matched,
            "expected_count": exp_count,
            "actual_count": act_count,
            "overlap_ratio": overlap_ratio,
            "strict_order": False,
        }
        return result
