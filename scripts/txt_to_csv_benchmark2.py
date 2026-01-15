import re
import csv
from pathlib import Path

input_path = Path('query_evaluations/benchmark_queries/AppInsightsQueries_from_excel.txt')
output_path = input_path.with_suffix('.csv')
text = input_path.read_text(encoding='utf-8')

# Normalize CRLF to LF
text = text.replace('\r\n', '\n')

# Regex: prompt (anything not tab/newline) then tab then double-quoted query (supports "" inside)
pattern = re.compile(r'([^\t\n]+)\t"((?:[^"]|"")*?)"', re.S)
matches = pattern.findall(text)

rows = []
for m in matches:
    prompt = m[0].strip()
    q = m[1]
    # Replace doubled double-quotes with single
    q = q.replace('""', '"')
    # Collapse whitespace to single space
    q_norm = re.sub(r"\s+", ' ', q).strip()
    rows.append((prompt, q_norm))

with output_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Prompt', 'Expected Query'])
    for p, q in rows:
        writer.writerow([p, q])

print(f'Found {len(rows)} entries, wrote to {output_path}')
