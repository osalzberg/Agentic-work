#!/usr/bin/env python3
"""Create a sample Excel template for batch testing"""

import pandas as pd
import sys

# Sample prompts for testing
sample_data = {
    "Prompt": [
        "Show me failed requests from the last hour",
        "Top 3 browser exceptions",
        "Failing dependencies",
        "Show me slow requests over 2 seconds",
        "Count exceptions by type",
    ],
    "Expected Query": [
        "",  # Leave empty or add expected queries if you have them
        "",
        "",
        "",
        "",
    ],
    "Generated Query": [
        "",  # Will be filled by batch_test_queries.py
        "",
        "",
        "",
        "",
    ],
    "Reason": [
        "",  # Will be filled by batch_test_queries.py
        "",
        "",
        "",
        "",
    ]
}

def create_template(output_file="test_prompts_template.xlsx"):
    """Create a template Excel file with sample prompts"""
    df = pd.DataFrame(sample_data)
    
    try:
        df.to_excel(output_file, index=False)
        print(f"✅ Template created: {output_file}")
        print(f"📝 Contains {len(df)} sample prompts")
        print("\nTo use this template:")
        print(f"1. Edit '{output_file}' and add your test prompts")
        print(f"2. Run: python batch_test_queries.py {output_file}")
    except Exception as e:
        print(f"❌ Failed to create template: {e}")
        print("💡 Make sure you have openpyxl installed: pip install openpyxl")
        sys.exit(1)

if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "test_prompts_template.xlsx"
    create_template(output)
