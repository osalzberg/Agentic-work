import pandas as pd

try:
    df = pd.read_csv('app_insights_capsule/benchmark_queries.csv', sep=None, engine='python', nrows=5)
    print('Columns:', list(df.columns))
    print('\nFirst 3 rows:')
    print(df.head(3))
    print('\nFirst row values:')
    for col in df.columns:
        print(f'  {col}: "{df[col].iloc[0]}"')
except Exception as e:
    print(f'Error: {e}')
    # Try with quotechar
    try:
        df = pd.read_csv('app_insights_capsule/benchmark_queries.csv', quotechar='"', escapechar='\\', on_bad_lines='skip', nrows=5)
        print('\n\nWith quotechar/escapechar:')
        print('Columns:', list(df.columns))
        print('\nFirst 3 rows:')
        print(df.head(3))
    except Exception as e2:
        print(f'Also failed: {e2}')
