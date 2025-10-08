"""Package initialization for src.

Loads environment variables from a `.env` file if present.
"""

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=False)
