import sqlite3
import os

db_path = 'user_data/tradesv3.sqlite'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print('Tables:', c.fetchall())
    c.execute("PRAGMA table_info(trades);")
    for col in c.fetchall():
        print(f"{col[1]} ({col[2]})")
else:
    print('No db found')
