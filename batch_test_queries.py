#!/usr/bin/env python3
"""
Batch Test Queries from CSV/Excel
Reads prompts from CSV or Excel file, generates KQL queries, and writes results back
"""

import sys
import os
import asyncio
import pandas as pd
from typing import Optional
from datetime import datetime

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
    reason_column: str = "Reason"
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
        prompts, test_cases, df = parse_prompts_from_file(input_file, prompt_col=prompt_column, expected_cols=[expected_column])
    except Exception as e:
        print(f"❌ Failed to read {file_type} file: {e}")
        return
            df = pd.read_excel(input_file)
            df = df.reset_index(drop=True)
    except Exception as e:
        print(f"❌ Failed to read {file_type} file: {e}")
        if not is_csv:
            print("💡 Make sure you have openpyxl installed: pip install openpyxl")
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
    
    # Process each row
    success_count = 0
    error_count = 0
    
    for idx, row in df.iterrows():
        prompt = row[prompt_column]
        
        # Skip empty prompts
        if pd.isna(prompt) or not str(prompt).strip():
            print(f"⏭️  Row {idx + 1}: Skipping empty prompt")
            df.at[idx, reason_column] = "Empty prompt"
            continue
        
        prompt_str = str(prompt).strip()
    # Process each row
    success_count = 0
    error_count = 0
    
    stop_on_critical_error = False
    critical_error_types = [
        'authentication', 'authorization', 'auth', '401', '403', 'permission', 'forbidden', 'server error', 'service unavailable', 'timeout', 'connection refused', 'invalid api key', 'not authorized', 'not authenticated', 'access denied', 'token expired', 'rate limit', '429', '500', '502', '503', '504'
    ]
    for row_num, (idx, row) in enumerate(df.iterrows()):
        if stop_on_critical_error:
            print(f"⏹️  Stopping batch due to critical error on previous row.")
            df.at[idx, reason_column] = "Skipped due to earlier critical error"
            continue
        prompt = row[prompt_column]
        # Skip empty prompts
        if pd.isna(prompt) or not str(prompt).strip():
            print(f"⏭️  Row {row_num + 1}: Skipping empty prompt")
            df.at[idx, reason_column] = "Empty prompt"
            continue
        prompt_str = str(prompt).strip()
        print(f"\n🔄 Row {row_num + 1}/{len(df)}: {prompt_str[:80]}{'...' if len(prompt_str) > 80 else ''}")
        try:
            # Translate natural language to KQL
            result = await agent.process_natural_language(prompt_str)
            # Handle different result formats
            if isinstance(result, dict) and result.get("type") == "query_success":
                kql_query = result.get("kql_query", "")
                df.at[idx, output_column] = kql_query
                df.at[idx, reason_column] = "Successfully generated"
                success_count += 1
                print(f"✅ Generated query ({len(kql_query)} chars)")
            elif isinstance(result, dict) and result.get("type") == "query_error":
                error = result.get("error", "Unknown error")
                kql_query = result.get("kql_query", "")
                df.at[idx, output_column] = kql_query if kql_query else ""
                df.at[idx, reason_column] = f"Error: {error}"
                error_count += 1
                print(f"❌ Failed: {error}")
                # Check for critical error on first row
                if row_num == 0 and any(word in str(error).lower() for word in critical_error_types):
                    print(f"🛑 Critical error detected on first prompt: {error}\nStopping further processing.")
                    stop_on_critical_error = True
            elif isinstance(result, str):
                # String result (error or other message)
                df.at[idx, output_column] = ""
                df.at[idx, reason_column] = f"Error: {result}"
                error_count += 1
                print(f"❌ Failed: {result}")
                if row_num == 0 and any(word in result.lower() for word in critical_error_types):
                    print(f"🛑 Critical error detected on first prompt: {result}\nStopping further processing.")
                    stop_on_critical_error = True
            else:
                df.at[idx, output_column] = ""
                df.at[idx, reason_column] = "Unknown result format"
                error_count += 1
                print(f"❌ Unknown result format")
        except Exception as e:
            df.at[idx, output_column] = ""
            df.at[idx, reason_column] = f"Exception: {str(e)}"
            error_count += 1
            print(f"❌ Exception: {e}")
            if row_num == 0 and any(word in str(e).lower() for word in critical_error_types):
                print(f"🛑 Critical exception detected on first prompt: {e}\nStopping further processing.")
                stop_on_critical_error = True
    
    # Save results
    print(f"\n💾 Saving results to: {output_file}")
    output_is_csv = output_file.lower().endswith('.csv')
    try:
        if output_is_csv:
            df.to_csv(output_file, index=False)
        else:
            df.to_excel(output_file, index=False)
        print(f"✅ Results saved successfully")
    except Exception as e:
        print(f"❌ Failed to save file: {e}")
        if not output_is_csv:
            # Try saving as CSV as fallback
            csv_file = output_file.replace('.xlsx', '.csv').replace('.xls', '.csv')
            df.to_csv(csv_file, index=False)
            print(f"💾 Saved as CSV instead: {csv_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"Total rows processed: {len(df)}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"⏭️  Skipped: {len(df) - success_count - error_count}")
    print("="*60)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch test KQL query generation from CSV or Excel file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process Excel file with default workspace
  python batch_test_queries.py test_prompts.xlsx
  
  # Process CSV file (with or without headers)
  python batch_test_queries.py test_prompts.csv
  python batch_test_queries.py benchmark_queries.csv
  
  # Specify workspace ID
  python batch_test_queries.py test_prompts.xlsx --workspace 81a662b5-8541-481b-977d-5d956616ac5e
  
  # Specify output file (auto-detects format)
  python batch_test_queries.py test_prompts.csv --output results.csv
  
  # Use custom column names
  python batch_test_queries.py test_prompts.xlsx --prompt-col "Question" --output-col "Answer"
  
Note: CSV files can have headers or be headerless (prompt, expected_query format).
      Headerless format is auto-detected.
        """
    )
    
    parser.add_argument("input_file", help="Path to input file (.csv, .xlsx, or .xls)")
    parser.add_argument(
        "-o", "--output",
        help="Path to output file (default: same format as input with _results suffix)",
        default=None
    )
    parser.add_argument(
        "-w", "--workspace",
        help="Azure Log Analytics workspace ID",
        default="81a662b5-8541-481b-977d-5d956616ac5e"
    )
    parser.add_argument(
        "--prompt-col",
        help="Name of column containing prompts (default: 'Prompt')",
        default="Prompt"
    )
    parser.add_argument(
        "--expected-col",
        help="Name of column containing expected queries (default: 'Expected Query')",
        default="Expected Query"
    )
    parser.add_argument(
        "--output-col",
        help="Name of column to write generated queries (default: 'Generated Query')",
        default="Generated Query"
    )
    parser.add_argument(
        "--reason-col",
        help="Name of column to write reason/status (default: 'Reason')",
        default="Reason"
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
    asyncio.run(process_file(
        input_file=args.input_file,
        output_file=args.output,
        workspace_id=args.workspace,
        prompt_column=args.prompt_col,
        expected_column=args.expected_col,
        output_column=args.output_col,
        reason_column=args.reason_col
    ))


if __name__ == "__main__":
    main()
