# Batch Testing in Web App

The web app now includes built-in batch testing functionality for testing multiple prompts at once through Excel files.

## 🚀 How to Use

### Step 1: Prepare Your Excel File

Create an Excel file (`.xlsx` or `.xls`) or CSV (.csv) file with the following columns:

| Column | Required | Description |
|--------|----------|-------------|
| **Prompt** | ✅ Yes | Your natural language questions |
| **Expected Query** | ❌ No | Optional reference queries (ignored by system) |
| **Generated Query** | Auto-filled | System writes generated KQL here |


**Example Excel Content:**
```
Prompt                                    | Expected Query | Generated Query 
Show me failed requests from last hour    |                |                
Top 3 browser exceptions                  |                |                
Failing dependencies                      |                |                
Slow requests over 2 seconds              |                |                
```

### Step 2: Setup Workspace and LLM

1. Open the web app at http://localhost:8080
2. Enter your workspace ID
3. Select a LLM deployment
4. Review/edit the system prompt

### Step 3: Upload and Process

1. Scroll to the **"📊 Batch Test Queries"** section
2. Click **"📁 Input File"** 
3. Select your Excel/CSV file
4. Click **"🚀 Run Benchmark"**
5. Watch the progress bar as queries are generated and scored
6. Review the summary (Total, Successful, Failed, Wrong Query)

### Step 4: Download Results

1. Once complete, click one of the **"💾 Download Results"** buttons
2. Open the downloaded file
3. Review the detailed report and scoring


### For Best Results
- Write clear, specific prompts
- Include time ranges when relevant (e.g., "last hour", "past 24 hours")
- Mention specific metrics or fields you want to see
- Use domain-specific keywords (requests, exceptions, container, etc.)

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
