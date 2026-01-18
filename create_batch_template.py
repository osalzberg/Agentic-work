#!/usr/bin/env python3
"""
Create a properly formatted Excel template for batch testing
"""
import sys

import pandas as pd

# Sample data
data = {
    "Prompt": [
        "Show me all failed requests in the last hour",
        "What are the top 5 exceptions by count?",
        "Show request duration over time",
    ]
}

df = pd.DataFrame(data)

# Save as xlsx
output_file = "batch_test_template.csv"
df.to_csv(output_file, index=False)

print(f"✅ Created template file: {output_file}")
print(f"   Rows: {len(df)}")
print(f"   Columns: {list(df.columns)}")
