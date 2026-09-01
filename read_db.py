import sqlite3
import pandas as pd 

conn = sqlite3.connect("rag_metrics.db")
# Read the SQL query straight into a clean terminal view
df = pd.read_sql_query("SELECT * FROM evaluation_logs", conn)
print(df.to_string(index=False))
conn.close()
