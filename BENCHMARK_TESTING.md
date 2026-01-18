# Benchmark Testing KQL Queries from CSV/Excel

This tool allows you to test multiple natural language prompts in batch by reading them from CSV or Excel files, generating KQL queries, and writing the results back.

## Setup

1. Install required dependencies:
```bash
pip install pandas openpyxl
```

2. Make sure your Azure OpenAI credentials are configured in your `.env` file

## Quick Start

### Option 1: Use the Template

1. Create a template file:
```bash
python create_test_template.py
```

2. Edit `test_prompts_template.xlsx` and add your prompts

3. Run the benchmark:
```bash
python benchmark_queries.py test_prompts_template.xlsx
```

### Option 2: Use Your Own CSV/Excel File

Your CSV or Excel file should have these columns:
- **Prompt** (required): Natural language questions
- **Expected Query** (optional): Expected KQL queries for comparison
- **Generated Query** (will be added): The actual generated KQL
- **Reason** (will be added): Success/error status

Example:
```bash
python benchmark_queries.py my_prompts.xlsx --workspace 81a662b5-8541-481b-977d-5d956616ac5e
```

## Usage

### Basic Usage (Excel)
```bash
python benchmark_queries.py input.xlsx
```

### Basic Usage (CSV)
```bash
python benchmark_queries.py input.csv
```

### Specify Workspace ID
```bash
python benchmark_queries.py input.xlsx --workspace YOUR_WORKSPACE_ID
```

### Specify Output File
```bash
# Output format matches input format by default
python benchmark_queries.py input.csv --output results.csv
python benchmark_queries.py input.xlsx --output results.xlsx
```

### Use Custom Column Names
```bash
python benchmark_queries.py input.xlsx \
  --prompt-col "Question" \
  --output-col "Answer" \
  --reason-col "Status"
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `input_file` | Path to input Excel file | (required) |
| `-o, --output` | Path to output Excel file | `input_results_TIMESTAMP.xlsx` |
| `-w, --workspace` | Azure workspace ID | `81a662b5-8541-481b-977d-5d956616ac5e` |
| `--prompt-col` | Name of prompt column | `Prompt` |
| `--expected-col` | Name of expected query column | `Expected Query` |
| `--output-col` | Name of output query column | `Generated Query` |
| `--reason-col` | Name of reason/status column | `Reason` |

## File Format (CSV/Excel)

### Input Format
Works with both CSV and Excel formats with the same column structure:

**Option 1: CSV/Excel with Headers**
```
| Prompt                                    | Expected Query | Generated Query | Reason |
|-------------------------------------------|----------------|-----------------|--------|
| Show me failed requests from last hour    |                |                 |        |
| Top 3 browser exceptions                  |                |                 |        |
```

**Option 2: Headerless CSV** (auto-detected)
```csv
Show me failed requests from last hour,"AppRequests | where TimeGenerated > ago(1h)..."
Top 3 browser exceptions,"AppExceptions | top 3..."
```

The tool automatically detects headerless CSV files and assigns standard column names.

### Output Format
```
| Prompt                                    | Expected Query | Generated Query                              | Reason                 |
|-------------------------------------------|----------------|----------------------------------------------|------------------------|
| Show me failed requests from last hour    |                | AppRequests | where TimeGenerated > ago(1h)... | Successfully generated |
| Top 3 browser exceptions                  |                | AppExceptions | where TimeGenerated > ago(1h)... | Successfully generated |
| Failing dependencies                      |                | AppDependencies | where Success == false...      | Successfully generated |
```

## Examples
Excel file with default settings
```bash
python benchmark_queries.py test_prompts.xlsx
```

### Process CSV file with default settings
```bash
python benchmark_queries.py test_prompts.csv
```

### Process headerless CSV (like benchmark_queries.csv)
```bash
python benchmark_queries.py app_insights_capsule/benchmark_queries.csv
```

### Process with specific workspace
```bash
python benchmark_queries.py test_prompts.xlsx --workspace 81a662b5-8541-481b-977d-5d956616ac5e
```

### Process and save to specific output file
```bash
python benchmark_queries.py test_prompts.xlsx --output my_results.xlsx
python benchmark_queries.py teCSV or st_prompts.csv --output my_results.csv
python benchmark_queries.py test_prompts.xlsx --output my_results.xlsx
```

## Output

The script will:
1. Read prompts from the input Excel file
2. Generate KQL queries for each prompt using the agent
3. Write generated queries into the "Generated Query" column
4. Write success/error status to the "Reason" column
5. Save results to the output file

Example output:
```
📂 Reading Excel file: test_prompts.xlsx
✅ Found 5 rows to process
🔧 Using workspace ID: 81a662b5-8541-481b-977d-5d956616ac5e

🔄 Row 1/5: Show me failed requests from the last hour
✅ Generated query (245 chars)

🔄 Row 2/5: Top 3 browser exceptions
✅ Generated query (312 chars)

💾 Saving results to: test_prompts_results_20260111_123456.csv
✅ Results saved successfully

============================================================
📊 SUMMARY
============================================================
Total rows processed: 5
✅ Successful: 5
❌ Errors: 0
⏭️  Skipped: 0
============================================================
```

## Troubleshooting

### "Failed to read Excel file"
Install openpyxl for Excel support:
```bash
pip install openpyxl
```

### "Column 'Prompt' not found"
Make sure your CSV/Excel file has a column named "Prompt", or use `--prompt-col` to specify a different column name.

### Authentication errors
Make sure your `.env` file has the correct Azure OpenAI credentials:
```
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=your-deployment
```

## Notes

- Empty prompts will be skipped
- The script processes rows sequentially to avoid rate limiting
- Results are saved even if some queries fail
- Output format matches input format (CSV→CSV, Excel→Excel)
- If Excel save fails, a CSV file will be created as a fallback
- CSV files don't require openpyxl installation
- **Headerless CSV files are automatically detected** - common format with 2 columns (prompt, expected_query)
