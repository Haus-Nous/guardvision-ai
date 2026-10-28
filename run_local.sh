#!/usr/bin/env bash
# =============================================================================
# GuardVision AI: Local Launch Script
# Starts Streamlit using the dedicated virtual environment (.venv)
# =============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ -f "$DIR/.venv/bin/streamlit" ]; then
    echo "Starting GuardVision AI with .venv/bin/streamlit..."
    "$DIR/.venv/bin/streamlit" run app.py --server.port 8501
elif command -v streamlit >/dev/null 2>&1; then
    echo "Starting GuardVision AI with system streamlit..."
    streamlit run app.py --server.port 8501
else
    echo "Streamlit not found. Please activate your environment or install dependencies:"
    echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi
