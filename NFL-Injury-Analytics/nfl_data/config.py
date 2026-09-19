"""Environment-driven MySQL connection settings.

Set these variables in the shell or in a local, ignored ``.env`` file before
running the API, notebooks, or data pipeline. No credentials belong in source
control.
"""

import os


host = os.getenv("NFL_DB_HOST", "127.0.0.1")
port = int(os.getenv("NFL_DB_PORT", "3306"))
user = os.getenv("NFL_DB_USER", "root")
passwd = os.getenv("NFL_DB_PASSWORD", "")
db = os.getenv("NFL_DB_NAME", "nfl_injuries")
