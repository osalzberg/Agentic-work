#!/usr/bin/env python3
"""
Batch Test Queries from Excel
Reads prompts from Excel file, generates KQL queries, and writes results back
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


async def process_excel_file(
    input_file: str,
    output_file: str,
    workspace_id: str,
    prompt_column: str = "Prompt",
    expected_column: str = "Expected Query",
    output_column: str = "Generated Query",
    reason_column: str = "Reason"
):
    """Process Excel file with prompts and generate KQL queries.
    
    Args:
        input_file: Path to input Excel file
        output_file: Path to output Excel file
        workspace_id: Azure Log Analytics workspace ID
        prompt_column: Name of column containing prompts
        expected_column: Name of column containing expected queries (optional)
        output_column: Name of column to write generated queries
        reason_column: Name of column to write generation reason/status
    """
    
    print(f"📂 Reading Excel file: {input_file}")
    
    # Read Excel file
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"❌ Failed to read Excel file: {e}")
        print("💡 Make sure you have openpyxl installed: pip install openpyxl")
        return
    
    # Validate columns
    if prompt_column not in df.columns:
        print(f"❌ Column '{prompt_column}' not found in Excel file")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    # Add output columns if they don't exist
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
        print(f"\n🔄 Row {idx + 1}/{len(df)}: {prompt_str[:80]}{'...' if len(prompt_str) > 80 else ''}")
        
        try:
            # Translate natural language to KQL
            result = await agent.process_query(prompt_str)
            
            if result.get("success"):
                kql_query = result.get("kql", "")
                df.at[idx, output_column] = kql_query
                df.at[idx, reason_column] = "Successfully generated"
                success_count += 1
                print(f"✅ Generated query ({len(kql_query)} chars)")
            else:
                error = result.get("error", "Unknown error")
                df.at[idx, output_column] = ""
                df.at[idx, reason_column] = f"Error: {error}"
                error_count += 1
                print(f"❌ Failed: {error}")
                
        except Exception as e:
            df.at[idx, output_column] = ""
            df.at[idx, reason_column] = f"Exception: {str(e)}"
            error_count += 1
            print(f"❌ Exception: {e}")
    
    # Save results
    print(f"\n💾 Saving results to: {output_file}")
    try:
        df.to_excel(output_file, index=False)
        print(f"✅ Results saved successfully")
    except Exception as e:
        print(f"❌ Failed to save Excel file: {e}")
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
        description="Batch test KQL query generation from Excel file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process test_prompts.xlsx with default workspace
  python batch_test_queries.py test_prompts.xlsx
  
  # Specify workspace ID
  python batch_test_queries.py test_prompts.xlsx --workspace 81a662b5-8541-481b-977d-5d956616ac5e
  
  # Specify output file
  python batch_test_queries.py test_prompts.xlsx --output results.xlsx
  
  # Use custom column names
  python batch_test_queries.py test_prompts.xlsx --prompt-col "Question" --output-col "Answer"
        """
    )
    
    parser.add_argument("input_file", help="Path to input Excel file (.xlsx or .xls)")
    parser.add_argument(
        "-o", "--output",
        help="Path to output Excel file (default: input_file with _results suffix)",
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"{base_name}_results_{timestamp}.xlsx"
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Run async processing
    asyncio.run(process_excel_file(
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
