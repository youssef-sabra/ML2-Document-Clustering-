import os
import sys
import runpy

# Ensure package src is on path so imports like `from nlp_clustering import ...` work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Execute the existing Streamlit script
runpy.run_path(os.path.join(os.path.dirname(__file__), "app", "streamlit_app.py"), run_name="__main__")
