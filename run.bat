@echo off
cd /d "%~dp0"
echo Starting Hallucination Detection demo...
streamlit run app.py --server.headless true
pause
