import os

DATA_DIR = "/var/lib/pgadmin"
LOG_FILE = os.path.join(DATA_DIR, "pgadmin4.log")
SQLITE_PATH = os.path.join(DATA_DIR, "pgadmin4.db")
SESSION_DB_PATH = '/tmp/pgadmin_sessions'