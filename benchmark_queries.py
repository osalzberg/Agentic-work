#!/usr/bin/env python3
"""
Batch Test Queries from CSV/Excel
Reads prompts from CSV or Excel file, generates KQL queries, and writes results back

Use `python benchmark_queries.py` to run bulk benchmarks.
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")

from logs_agent import KQLAgent


async def process_file(
    input_file: str,
    output_file: str,
    workspace_id: str,
    prompt_column: str = "Prompt",
    expected_column: str = "Expected Query",
    output_column: str = "Generated Query",
    reason_column: str = "Reason",
):
    """Process CSV or Excel file with prompts and generate KQL queries.

    Args:
        input_file: Path to input CSV or Excel file
        output_file: Path to output CSV or Excel file
        workspace_id: Azure Log Analytics workspace ID
        prompt_column: Name of column containing prompts
        expected_column: Name of column containing expected queries (optional)
        output_column: Name of column to write generated queries
        reason_column: Name of column to write generation reason/status
    """

    # Use the shared parser to read prompts and expected queries
    from utils.file_parser import parse_prompts_from_file

    try:
        prompts, test_cases, df = parse_prompts_from_file(
            input_file, prompt_col=prompt_column, expected_cols=[expected_column]
        )
    except Exception as e:
        print(f"❌ Failed to read input file: {e}")
        return

    # Ensure output columns exist
    if output_column not in df.columns:
        df[output_column] = ""
    if reason_column not in df.columns:
        df[reason_column] = ""

    print(f"✅ Found {len(df)} rows to process")
    print(f"🔧 Using workspace ID: {workspace_id}")

    # Initialize agent
    agent = KQLAgent(workspace_id=workspace_id)

    # Use shared batch runner to process prompts (preserves agent behavior)
    from utils.batch_runner import run_batch
    from utils.report_builder import build_excel, build_json

    # Build prompt cases list expected by run_batch
    prompts_list = []
    for idx, row in df.iterrows():
        prompt = row[prompt_column]
        if pd.isna(prompt) or not str(prompt).strip():
            df.at[idx, reason_column] = "Empty prompt"
            continue
        prompts_list.append(
            {
                "id": idx,
                "prompt": str(prompt).strip(),
                "expected_query": row.get(expected_column),
            }
        )

    print(f"✅ Processing {len(prompts_list)} prompts using shared runner")
    results = run_batch(
        prompts_list, agent=agent, execute=False, stop_on_critical_error=False
    )

    # Attach results back to dataframe for saving
    for r in results:
        pid = r.get("prompt_id")
        if pid is None or pid >= len(df):
            continue
        df.at[pid, output_column] = r.get("gen_query", "")
        df.at[pid, reason_column] = (
            "Generated" if not r.get("error") else f"Error: {r.get('error')}"
        )
        # Optionally write returned_rows_count if present
        if "returned_rows_count" in r:
            df.at[pid, "Returned Rows Count"] = r.get("returned_rows_count")

    # Save results
    print(f"\n💾 Saving results to: {output_file}")
    output_is_csv = output_file.lower().endswith(".csv")
    try:
        # Avoid direct Excel writes from DataFrame; always write CSV for tabular annotated output
        if output_is_csv:
            df.to_csv(output_file, index=False)
        else:
            # If user requested an Excel output, save a CSV instead and keep canonical Excel report separate
            csv_file = output_file.replace(".xlsx", ".csv").replace(".xls", ".csv")
            df.to_csv(csv_file, index=False)
            print(
                f"⚠️ df.to_excel avoided; saved annotated results as CSV instead: {csv_file}"
            )
        print(f"✅ Results saved successfully (CSV)")
    except Exception as e:
        print(f"❌ Failed to save file: {e}")
        if not output_is_csv:
            # Try saving as CSV as fallback
            csv_file = output_file.replace(".xlsx", ".csv").replace(".xls", ".csv")
            df.to_csv(csv_file, index=False)
            print(f"💾 Saved as CSV instead: {csv_file}")

    # Also build and save report artifacts using report_builder
    metadata = {"model": os.environ.get("AZURE_OPENAI_DEPLOYMENT", "Unknown")}
    json_report = build_json(results, metadata)
    try:
        # If the user requested an Excel output file, embed the input dataframe into the workbook
        # so the canonical report contains the original prompts table.
        embed_input = not output_is_csv
        excel_bytes = build_excel(
            results, metadata, input_df=df, embed_input=embed_input
        )
    except Exception:
        excel_bytes = None
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(reports_dir, f"batch_results_{ts}.json")
    xlsx_path = os.path.join(reports_dir, f"batch_results_{ts}.xlsx")
    with open(json_path, "w", encoding="utf-8") as jf:
        jf.write(json_report)
    if excel_bytes:
        with open(xlsx_path, "wb") as xf:
            xf.write(excel_bytes)
    print(f"[INFO] JSON report saved to: {json_path}")
    if excel_bytes:
        print(f"[INFO] Excel report saved to: {xlsx_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    total_processed = len(df)
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "failed")
    skipped = total_processed - successful - failed
    print(f"Total rows processed: {total_processed}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Errors: {failed}")
    print(f"⏭️  Skipped: {skipped}")
    print("=" * 60)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch test KQL query generation from CSV or Excel file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process Excel file with default workspace
    python benchmark_queries.py test_prompts.xlsx
  
    # Process CSV file (with or without headers)
    python benchmark_queries.py test_prompts.csv
    python benchmark_queries.py benchmark_queries.csv
  
    # Specify workspace ID
    python benchmark_queries.py test_prompts.xlsx --workspace 81a662b5-8541-481b-977d-5d956616ac5e
  
    # Specify output file (auto-detects format)
    python benchmark_queries.py test_prompts.csv --output results.csv
  
    # Use custom column names
    python benchmark_queries.py test_prompts.xlsx --prompt-col "Question" --output-col "Answer"
  
Note: CSV files can have headers or be headerless (prompt, expected_query format).
      Headerless format is auto-detected.
        """,
    )

    parser.add_argument("input_file", help="Path to input file (.csv, .xlsx, or .xls)")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output file (default: same format as input with _results suffix)",
        default=None,
    )
    parser.add_argument(
        "-w",
        "--workspace",
        help="Azure Log Analytics workspace ID",
        default="81a662b5-8541-481b-977d-5d956616ac5e",
    )
    parser.add_argument(
        "--prompt-col",
        help="Name of column containing prompts (default: 'Prompt')",
        default="Prompt",
    )
    parser.add_argument(
        "--expected-col",
        help="Name of column containing expected queries (default: 'Expected Query')",
        default="Expected Query",
    )
    parser.add_argument(
        "--output-col",
        help="Name of column to write generated queries (default: 'Generated Query')",
        default="Generated Query",
    )
    parser.add_argument(
        "--reason-col",
        help="Name of column to write reason/status (default: 'Reason')",
        default="Reason",
    )

    args = parser.parse_args()

    # Determine output file
    if args.output is None:
        base_name = os.path.splitext(args.input_file)[0]
        extension = os.path.splitext(args.input_file)[1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"{base_name}_results_{timestamp}{extension}"

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        sys.exit(1)

    # Run async processing
    asyncio.run(
        process_file(
            input_file=args.input_file,
            output_file=args.output,
            workspace_id=args.workspace,
            prompt_column=args.prompt_col,
            expected_column=args.expected_col,
            output_column=args.output_col,
            reason_column=args.reason_col,
        )
    )


if __name__ == "__main__":
    main()
