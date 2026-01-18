import io
import json
from typing import Any, Dict, List, Tuple

import pandas as pd
import os


def _collect_exec_stat_keys(results: List[Dict[str, Any]]) -> Tuple[set, set]:
    gen_keys = set()
    exp_keys = set()
    for r in results:
        ge = r.get("gen_exec_stats") or {}
        re = r.get("returned_exec_stats") or {}
        gen_keys.update(ge.keys())
        exp_keys.update(re.keys())
    return gen_keys, exp_keys


def build_json(results: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
    # strip any large free-text reasoning fields and return a canonical JSON string
    sanitized = []
    for r in results:
        rr = dict(r)
        # remove any LLM reasoning/text fields if present
        for k in list(rr.keys()):
            if k.startswith("llm_") or k in ("reasoning", "explanation", "analysis"):
                rr.pop(k, None)
        sanitized.append(rr)

    out = {"metadata": metadata, "results": sanitized}
    return json.dumps(out, indent=2, default=str)


def build_excel(
    results: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    promote_exec_stats: bool = True,
    input_df=None,
    embed_input: bool = False,
) -> bytes:
    # Build a pandas DataFrame with promoted exec-stat columns
    gen_keys, exp_keys = _collect_exec_stat_keys(results)

    rows = []
    for r in results:
        base = {
            "Prompt ID": r.get("prompt_id"),
            "Prompt": r.get("prompt") or r.get("prompt_text"),
            "Generated Query": r.get("generated_query") or r.get("gen_query"),
            "Expected Query": r.get("expected_query"),
            "Expected Rows Count": r.get("expected_rows_count"),
            "Returned Rows Count": r.get("returned_rows_count"),
            "Zero Rows Warning": r.get("zero_rows_warning", False),
            "Score Query Similarity": (r.get("score") or {}).get("query_similarity"),
            "Score Results Match": (
                (r.get("score") or {}).get("results_match", {}).get("value")
                if isinstance((r.get("score") or {}).get("results_match"), dict)
                else (r.get("score") or {}).get("results_match")
            ),
            "Schema Match": (r.get("comparator") or {}).get("schema_match"),
            "Rows Match": (r.get("comparator") or {}).get("rows_match"),
        }

        if promote_exec_stats:
            for k in gen_keys:
                base[f"gen_{k}"] = (r.get("gen_exec_stats") or {}).get(k)
            for k in exp_keys:
                base[f"exp_{k}"] = (r.get("returned_exec_stats") or {}).get(k)

        rows.append(base)

    df = pd.DataFrame(rows)
    # Add metadata sheet as small key/value table
    meta_rows = [(k, str(v)) for k, v in (metadata or {}).items()]
    meta_df = (
        pd.DataFrame(meta_rows, columns=["key", "value"])
        if meta_rows
        else pd.DataFrame(columns=["key", "value"])
    )

    with io.BytesIO() as out:
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            # results sheet
            df.to_excel(writer, sheet_name="results", index=False)
            # optional embedded input sheet
            if embed_input and input_df is not None:
                try:
                    # if input_df is a pandas DataFrame, write directly
                    if hasattr(input_df, "to_excel"):
                        input_df.to_excel(writer, sheet_name="input", index=False)
                    else:
                        # attempt to coerce to DataFrame
                        pd.DataFrame(input_df).to_excel(
                            writer, sheet_name="input", index=False
                        )
                except Exception:
                    # fall back to writing a small representation
                    pd.DataFrame([{"input_preview": str(input_df)}]).to_excel(
                        writer, sheet_name="input", index=False
                    )

            meta_df.to_excel(writer, sheet_name="metadata", index=False)
        return out.getvalue()


def _escape_html(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _generate_warning_html(result: dict) -> str:
    try:
        score = result.get("score") or {}
        components = score.get("components", {}) if isinstance(score, dict) else {}
        rm_comp = components.get("results_match") if isinstance(components, dict) else None
        rm_score = None
        if isinstance(rm_comp, dict) and "score" in rm_comp:
            rm_score = float(rm_comp.get("score") or 0.0)
        elif isinstance(score, dict) and "results_match" in score:
            try:
                rm_score = float(score.get("results_match"))
            except Exception:
                rm_score = None

        rows_match = result.get("rows_match", {}) or {}
        rows_details = rows_match.get("details", {}) if isinstance(rows_match, dict) else {}
        actual_count = rows_details.get("actual_count") if isinstance(rows_details, dict) else None
        if actual_count is None:
            actual_count = result.get("returned_rows_count")

        warnings = []
        if (
            rm_score is not None
            and rm_score >= 1.0
            and (actual_count is None or actual_count == 0)
        ):
            warnings.append(
                '<span style="color:#9a5700; font-weight:600;">⚠️ Perfect results match reported but zero rows returned</span>'
            )

        return " ".join(warnings) if warnings else "-"
    except Exception:
        return "-"


def write_html_report(report: dict, output_path: str):
    """Generate an HTML report with a table view of evaluation results and write to `output_path`."""
    results = report.get("results", [])

    # Reuse the HTML template from the legacy runner; fill values from report
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>KQL Evaluation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        /* trimmed for brevity in code but kept same styles as runner */
        table {{ width: 100%; border-collapse: collapse; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
        th {{ background-color: #0078d4; color: white; padding: 12px; text-align: left; font-weight: 600; position: sticky; top: 0; }}
        td {{ padding: 12px; border-bottom: 1px solid #e0e0e0; vertical-align: top; }}
        .kql-cell {{ font-family: 'Consolas', 'Monaco', monospace; font-size:12px; white-space: pre-wrap; word-break: break-word; background-color: #f8f8f8; padding:8px; border-radius:4px; }}
        .score-cell {{ text-align: center; font-weight: 600; font-size: 16px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>KQL Evaluation Report</h1>
        <div class="metadata">
            <strong>Model:</strong> {deployment}<br>
            <strong>API Version:</strong> {api_version}<br>
            <strong>Dataset:</strong> {dataset}<br>
            <strong>Total Tests:</strong> {count}<br>
        </div>
        <div class="system-prompt">
            <strong>System Prompt:</strong><br>
            {system_prompt}
        </div>
    </div>
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Prompt</th>
                <th>Expected KQL</th>
                <th>Generated KQL</th>
                <th>Expected Rows</th>
                <th>Generated Rows</th>
                <th>Exec Status</th>
                <th>Critical Issue</th>
                <th>Overall Score</th>
                <th>Structural Similarity</th>
                <th>Schema Match</th>
                <th>Rows Match</th>
                <th>Warnings</th>
                <th>LLM Grade</th>
            </tr>
        </thead>
        <tbody>
""".format(
        deployment=report.get("model", {}).get("deployment", "unknown"),
        api_version=report.get("model", {}).get("api_version", "unknown"),
        system_prompt=_escape_html(report.get("model", {}).get("system_prompt", "")),
        dataset=report.get("dataset", ""),
        count=report.get("count", 0),
    )

    for result in results:
        test_id = result.get("id", "")
        prompt = result.get("prompt", "")
        expected_kql = result.get("expected_kql", "")
        generated_kql = result.get("generated_kql", "")

        metrics = result.get("metrics", {})
        exec_success = metrics.get("exec_success", "NA")
        overall_score = result.get("score", "NA")
        structural_sim = metrics.get("structural_similarity", "NA")
        schema_match = metrics.get("schema_match_score", "NA")
        rows_match = metrics.get("rows_match_score", "NA")
        llm_grade = metrics.get("llm_graded_similarity", "NA")

        if result.get("exec_error"):
            exec_status = f'<span class="status-failed">Failed</span><br><span class="error-cell">{_escape_html(result.get("exec_error", ""))}</span>'
        elif exec_success == 1.0:
            exec_status = '<span class="status-valid">Valid Query</span>'
        elif exec_success == "NA":
            exec_status = '<span class="status-na">N/A</span>'
        else:
            exec_status = f'<span class="status-failed">{exec_success}</span>'

        critical_issue = result.get("critical_issue")
        if critical_issue:
            critical_issue_html = f'<span class="error-cell">{_escape_html(critical_issue)}</span>'
        else:
            critical_issue_html = '<span class="status-valid">-</span>'

        def format_score(score):
            if score == "NA":
                return '<span class="score-na">N/A</span>'
            try:
                score_val = float(score)
                if score_val >= 0.95:
                    return f'<span class="score-perfect">{score_val:.3f}</span>'
                elif score_val >= 0.7:
                    return f'<span class="score-good">{score_val:.3f}</span>'
                else:
                    return f'<span class="score-poor">{score_val:.3f}</span>'
            except (ValueError, TypeError):
                return f'<span class="score-na">{score}</span>'

        rows_match_data = result.get("rows_match", {})
        rows_details = rows_match_data.get("details", {})
        expected_row_count = rows_details.get("expected_count", "N/A")
        actual_row_count = rows_details.get("actual_count", "N/A")

        html += f"""
            <tr>
                <td>{_escape_html(test_id)}</td>
                <td class="prompt-cell">{_escape_html(prompt)}</td>
                <td class="kql-cell">{_escape_html(expected_kql)}</td>
                <td class="kql-cell">{_escape_html(generated_kql)}</td>
                <td class="metric-cell">{expected_row_count}</td>
                <td class="metric-cell">{actual_row_count}</td>
                <td class="metric-cell">{exec_status}</td>
                <td class="metric-cell">{critical_issue_html}</td>
                <td class="score-cell">{format_score(overall_score)}</td>
                <td class="metric-cell">{format_score(structural_sim)}</td>
                <td class="metric-cell">{format_score(schema_match)}</td>
                <td class="metric-cell">{format_score(rows_match)}</td>
                <td class="metric-cell">{_escape_html(_generate_warning_html(result))}</td>
                <td class="metric-cell">{format_score(llm_grade)}</td>
            </tr>
"""

    html += """
        </tbody>
    </table>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
