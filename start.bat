@echo off
REM Quick start script for Windows

echo ============================================
echo Starting Pre-test/Post-test Analysis App
echo ============================================
echo.

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment and install dependencies
echo Installing dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt --quiet

REM Run the Streamlit app
echo.
echo Starting Streamlit application...
echo The app will open in your default browser.
echo Press Ctrl+C to stop the server.
echo.
streamlit run app\main.py

pause
