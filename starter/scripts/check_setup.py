import os
from urllib.parse import urlparse
from dotenv import load_dotenv
import pg8000.native

load_dotenv()

database_url = os.getenv("DATABASE_URL")
if not database_url:
    raise SystemExit("DATABASE_URL is not set")

parsed = urlparse(database_url)
user = parsed.username or "day6"
password = parsed.password or "day6_local_only"
host = parsed.hostname or "127.0.0.1"
port = parsed.port or 55433
database = parsed.path.lstrip("/") or "day6_lab"

con = pg8000.native.Connection(user=user, host=host, port=port, database=database, password=password)

res = con.run("SELECT version();")
pg_version = res[0][0].split(",")[0]
print(f"Database : {database}")
print(f"PostgreSQL: {pg_version}")

res = con.run("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
if not res:
    raise SystemExit("Extension 'vector' is not installed in the database")
vector_version = res[0][0]
print(f"pgvector  : {vector_version}")

print("Setup OK")
con.close()
