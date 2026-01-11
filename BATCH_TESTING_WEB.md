# Batch Testing in Web App

The web app now includes built-in batch testing functionality for testing multiple prompts at once through Excel files.

## 🚀 How to Use

### Step 1: Prepare Your Excel File

Create an Excel file (`.xlsx` or `.xls`) with the following columns:

| Column | Required | Description |
|--------|----------|-------------|
| **Prompt** | ✅ Yes | Your natural language questions |
| **Expected Query** | ❌ No | Optional reference queries (ignored by system) |
| **Generated Query** | Auto-filled | System writes generated KQL here |
| **Reason** | Auto-filled | System writes success/error status here |

**Example Excel Content:**
```
Prompt                                    | Expected Query | Generated Query | Reason
Show me failed requests from last hour    |                |                 |
Top 3 browser exceptions                  |                |                 |
Failing dependencies                      |                |                 |
Slow requests over 2 seconds              |                |                 |
```

### Step 2: Setup Workspace

1. Open the web app at http://localhost:8080
2. Enter your workspace ID
3. Click "🚀 Connect Workspace"
4. Wait for connection to establish

### Step 3: Upload and Process

1. Scroll to the **"📊 Batch Test Queries"** section
2. Click **"📁 Choose Excel File"** 
3. Select your Excel file
4. Click **"🚀 Process Batch Test"**
5. Watch the progress bar as queries are generated
6. Review the summary (Total, Successful, Errors, Skipped)

### Step 4: Download Results

1. Once complete, click **"💾 Download Results Excel File"**
2. Open the downloaded file
3. Review generated queries in the "Generated Query" column
4. Check status in the "Reason" column

## 📊 Results Format

The downloaded Excel file will have all your original data plus:

**Generated Query Column:**
- Contains the full KQL query generated from your prompt
- Empty if generation failed

**Reason Column:**
- `"Successfully generated"` - Query generated successfully
- `"Error: <message>"` - Generation failed with specific error
- `"Empty prompt (skipped)"` - Row had no prompt text
- `"Exception: <message>"` - Unexpected error occurred

## 💡 Tips

### For Best Results
- Write clear, specific prompts
- Include time ranges when relevant (e.g., "last hour", "past 24 hours")
- Mention specific metrics or fields you want to see
- Use domain-specific keywords (requests, exceptions, dependencies, etc.)

### Common Prompts
```
✅ Good:
- "Show me failed requests from the last hour"
- "Top 10 slowest dependencies in the past day"
- "Count exceptions by type in the last 3 hours"
- "Chart request duration over time for last 6 hours"

❌ Too Vague:
- "Show data"
- "Get logs"
- "Find errors"
```

### Handling Large Files
- The system processes prompts sequentially to avoid rate limiting
- Each prompt may take 2-5 seconds to process
- For 100 prompts, expect ~5-10 minutes processing time
- Progress bar shows real-time status

## 🔧 Troubleshooting

### "No file selected"
- Make sure you clicked "Choose Excel File" and selected a valid `.xlsx` or `.xls` file

### "Excel file must have a 'Prompt' column"
- Your Excel file must have a column named exactly "Prompt" (case-sensitive)
- Check for typos or extra spaces in the column name

### "Please setup workspace first"
- You must connect to a workspace before batch testing
- Go back to the top and setup your workspace ID

### "pandas or openpyxl not installed"
- Run: `pip install pandas openpyxl`
- Restart the web server

### Download fails
- Check browser's download settings
- Try a different browser
- Check browser console for errors (F12)

## 📁 Sample Template

Use the provided template to get started:
```bash
python create_test_template.py
```

This creates `test_prompts_template.xlsx` with sample prompts you can modify.

## 🎯 Use Cases

### Testing Query Patterns
Upload a file with variations of similar prompts to test consistency:
```
- "Show failed requests"
- "Display failed requests"  
- "Get failed requests"
- "List failed requests"
```

### Documentation Generation
Generate a library of example queries for your team:
```
- Create prompts for common scenarios
- Generate queries via batch test
- Share the results file as documentation
```

### Quality Assurance
Compare generated queries against expected queries:
```
1. Add expected queries in "Expected Query" column
2. Run batch test to generate queries
3. Manually compare "Expected Query" vs "Generated Query" columns
4. Report discrepancies for improvement
```

### Training Data Collection
Build a dataset of prompt→query pairs for model training or evaluation.

## 🔐 Security Notes

- Files are processed server-side and not stored permanently
- Results are returned directly to your browser
- No data is logged or persisted beyond the session
- All processing uses your configured Azure credentials

## 📊 Performance

| File Size | Est. Processing Time |
|-----------|---------------------|
| 10 prompts | ~30 seconds |
| 50 prompts | ~3 minutes |
| 100 prompts | ~6 minutes |
| 500 prompts | ~30 minutes |

*Times are approximate and depend on prompt complexity and API response times.*
