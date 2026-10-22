# Vercel Python function for POST /api/predict. The logic lives in
# src/web_app.py so the local server and the deployed site behave the same.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from web_app import Handler as handler  # noqa: E402  (Vercel looks for `handler`)
