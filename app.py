# Vercel entrypoint: Vercel loads the WSGI `app` from this file and sends every
# request to it. The app itself lives in src/web_app.py (also run locally).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from web_app import app  # noqa: E402,F401
