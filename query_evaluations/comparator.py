
from __future__ import annotations

def fuzzy_column_alignment(expected_cols: List[str], actual_cols: List[str]) -> Dict[str, int]:
    """
    Returns a mapping from expected column index to actual column index using fuzzy matching.
    Only columns that can be mapped are included.
    """
    exp_norm = [_norm_col(c) for c in expected_cols]
    act_norm = [_norm_col(c) for c in actual_cols]
    col_map = {}  # expected index -> actual index
    used_actual = set()
    for i, exp_c in enumerate(exp_norm):
        # Try exact match first
        found = False
        for j, act_c in enumerate(act_norm):
            if j in used_actual:
                continue
            if exp_c == act_c:
                col_map[i] = j
                used_actual.add(j)
                found = True
                break
        if not found:
            # Fuzzy: substring match
            for j, act_c in enumerate(act_norm):
                if j in used_actual:
                    continue
                if exp_c in act_c or act_c in exp_c:
                    col_map[i] = j
                    used_actual.add(j)
                    found = True
                    break
    return col_map

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
    # Use shared fuzzy alignment
    col_map = fuzzy_column_alignment(expected_cols, actual_cols)
    mapped_expected_idxs = list(col_map.keys())
    mapped_actual_idxs = [col_map[i] for i in mapped_expected_idxs]
    # For schema, overlap = number of mapped columns
    overlap_count = len(mapped_expected_idxs)
    exp_count = len(exp_norm_list) or 1
    overlap_ratio = overlap_count / float(exp_count)
    # Missing: expected columns that could not be mapped
    missing = [expected_cols[i] for i in range(len(expected_cols)) if i not in mapped_expected_idxs]
    # Extra: actual columns that were not mapped
    extra = [actual_cols[j] for j in range(len(actual_cols)) if j not in mapped_actual_idxs]
    # Match: all expected columns are mapped
    if strict_order:
        result["match"] = exp_norm_list == [act_norm_list[col_map[i]] if i in col_map else None for i in range(len(expected_cols))]
    else:
        result["match"] = len(missing) == 0
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

    # --- NEW: Fuzzy column alignment and ignore extras ---
    # If available, get column headers from context (assume attached as attributes for this function)
    expected_cols = getattr(compare_rows, "expected_cols", None)
    actual_cols = getattr(compare_rows, "actual_cols", None)
    if expected_cols is not None and actual_cols is not None:
        col_map = fuzzy_column_alignment(expected_cols, actual_cols)
        mapped_expected_idxs = list(col_map.keys())
        mapped_actual_idxs = [col_map[i] for i in mapped_expected_idxs]
        reduced_expected = [[row[i] for i in mapped_expected_idxs] for row in expected_rows]
        reduced_actual = [[row[j] for j in mapped_actual_idxs] for row in actual_rows]
    else:
        reduced_expected = expected_rows
        reduced_actual = actual_rows

    if strict_order:
        # Position-wise comparison; count matched rows
        matched = 0
        min_len = min(len(reduced_expected), len(reduced_actual))
        mismatches = 0
        for r1, r2 in zip(reduced_expected[:min_len], reduced_actual[:min_len]):
            if _row_equal(r1, r2):
                matched += 1
            else:
                mismatches += 1
        mismatches += abs(len(reduced_expected) - len(reduced_actual))
        result["mismatches"] = mismatches
        result["match"] = (mismatches == 0 and len(reduced_expected) == len(reduced_actual))
        overlap_ratio = (matched / float(len(reduced_expected) or 1))
        result["details"] = {
            "matched_count": matched,
            "expected_count": len(reduced_expected),
            "actual_count": len(reduced_actual),
            "overlap_ratio": overlap_ratio,
            "strict_order": True,
        }
        return result
    else:
        from collections import Counter
        def _norm_row(r: List[str]) -> Tuple[str, ...]:
            return tuple(sorted(r))
        c_expected = Counter(_norm_row(r) for r in reduced_expected)
        c_actual = Counter(_norm_row(r) for r in reduced_actual)
        matched = 0
        for row, cnt in c_expected.items():
            matched += min(cnt, c_actual.get(row, 0))
        overlap_ratio = (matched / float(len(reduced_expected) or 1))
        result["match"] = c_expected == c_actual
        result["mismatches"] = 0 if result["match"] else abs(sum(c_expected.values()) - sum(c_actual.values()))
        result["details"] = {
            "matched_count": matched,
            "expected_count": len(reduced_expected),
            "actual_count": len(reduced_actual),
            "overlap_ratio": overlap_ratio,
            "strict_order": False,
        }
        return result
