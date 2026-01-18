"""Shared file parsing utilities for prompts/test cases.

Provides a single function `parse_prompts_from_file` that accepts either a file path
or a file-like object (werkzeug FileStorage) and returns a tuple:
  (prompts_list, test_cases_list, dataframe)

Heuristics:
- Flexible CSV parsing with fallbacks
- Headerless CSV detection for two-column prompt/expected files
- Column name normalization and mapping
"""

import io
from typing import Any, List, Optional, Tuple

import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    # Strip whitespace from column names
    df.columns = df.columns.astype(str).str.strip()
    # Map common variations to canonical names
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["prompt", "question", "nl", "natural language", "ask"]:
            col_map[col] = "Prompt"
        elif col_lower in [
            "query",
            "kql",
            "expected",
            "expected query",
            "expected_query",
        ]:
            col_map[col] = "Expected Query"
    if col_map:
        df = df.rename(columns=col_map)
    return df


def parse_prompts_from_file(
    source: Any, prompt_col: str = "Prompt", expected_cols: Optional[List[str]] = None
) -> Tuple[List[str], List[dict], pd.DataFrame]:
    """Parse prompts and optional expected queries from a CSV or Excel input.

    Args:
        source: Either a filesystem path (str/Path) or a file-like object with read()/seek().
        prompt_col: canonical prompt column name to return
        expected_cols: list of candidate expected columns to check; defaults to ['Expected Query','Query']

    Returns:
        (prompts_list, test_cases_list, dataframe)

    Raises:
        ValueError if parsing fails or required columns are missing.
    """
    if expected_cols is None:
        expected_cols = ["Expected Query", "Query"]

    # Determine whether source is a file path or file-like
    is_path = isinstance(source, (str,))

    try:
        if is_path:
            path = str(source)
            if path.lower().endswith(".csv"):
                # Flexible CSV parsing with fallbacks
                try:
                    df = pd.read_csv(path)
                except Exception:
                    try:
                        df = pd.read_csv(path, sep=None, engine="python")
                    except Exception:
                        df = pd.read_csv(
                            path, quotechar='"', escapechar="\\", on_bad_lines="skip"
                        )
            else:
                df = pd.read_excel(path)
        else:
            # file-like object (e.g., Flask FileStorage)
            file_obj = source
            # Try reading as CSV first using the file object
            try:
                # pandas can accept file-like for read_csv/read_excel
                df = pd.read_csv(file_obj)
            except Exception:
                try:
                    file_obj.seek(0)
                    df = pd.read_csv(file_obj, sep=None, engine="python")
                except Exception:
                    try:
                        file_obj.seek(0)
                        df = pd.read_csv(
                            file_obj,
                            quotechar='"',
                            escapechar="\\",
                            on_bad_lines="skip",
                        )
                    except Exception:
                        # Reset and try Excel parsing
                        file_obj.seek(0)
                        df = pd.read_excel(file_obj)

        df = _normalize_columns(df)

        # If prompt column missing, attempt smarter fallbacks: sniff delimiter, try common delimiters, or reload headerless
        if prompt_col not in df.columns:
            tried = []
            # 1) Try pandas auto-detect (sep=None) with python engine
            try:
                if is_path:
                    df2 = pd.read_csv(path, sep=None, engine="python")
                else:
                    source.seek(0)
                    df2 = pd.read_csv(source, sep=None, engine="python")
                df2 = _normalize_columns(df2)
                tried.append("sniff")
                if prompt_col in df2.columns:
                    df = df2
            except Exception:
                pass

        if prompt_col not in df.columns:
            # 2) Try common delimiters
            for d in [";", "\t", "|"]:
                try:
                    if is_path:
                        df2 = pd.read_csv(path, sep=d)
                    else:
                        source.seek(0)
                        df2 = pd.read_csv(source, sep=d)
                    df2 = _normalize_columns(df2)
                    tried.append(f"delim={d}")
                    if prompt_col in df2.columns:
                        df = df2
                        break
                except Exception:
                    continue

        if prompt_col not in df.columns:
            # 3) Try headerless two-column reload (common case)
            try:
                if is_path:
                    df2 = pd.read_csv(
                        path, header=None, names=[prompt_col, expected_cols[0]]
                    )
                else:
                    source.seek(0)
                    df2 = pd.read_csv(
                        source, header=None, names=[prompt_col, expected_cols[0]]
                    )
                df2 = _normalize_columns(df2)
                tried.append("headerless")
                if prompt_col in df2.columns:
                    df = df2
            except Exception:
                pass

        # Final check
        if prompt_col not in df.columns:
            raise ValueError(
                f"Missing required column: {prompt_col} (attempted: {', '.join(tried)})"
            )

        # Determine expected column if present
        expected_col = None
        for c in expected_cols:
            if c in df.columns:
                expected_col = c
                break

        prompts = df[prompt_col].fillna("").astype(str).tolist()
        test_cases = []
        for idx in range(len(df)):
            test_case = {
                "prompt": prompts[idx],
                "expected_query": (
                    df[expected_col].iloc[idx]
                    if expected_col and pd.notna(df[expected_col].iloc[idx])
                    else None
                ),
            }
            test_cases.append(test_case)

        return prompts, test_cases, df

    except Exception as e:
        raise
