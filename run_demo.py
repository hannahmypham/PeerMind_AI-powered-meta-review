#!/usr/bin/env python3
"""
Launcher for Streamlit demo. Use this as Main file path when deploying:
  Main file path: run_demo.py
"""
import sys
from pathlib import Path

# Run the actual demo via Streamlit CLI
demo_path = Path(__file__).resolve().parent / "src" / "demo_csv.py"
sys.argv = ["streamlit", "run", str(demo_path), "--server.headless", "true"]

from streamlit.web import cli as stcli
stcli.main()
