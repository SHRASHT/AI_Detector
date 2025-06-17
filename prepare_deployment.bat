@echo off
echo ====================================
echo   AI Detector - Deployment Helper
echo ====================================
echo.

echo This script will help you prepare your AI Detector for deployment.
echo.

echo Step 1: Checking requirements...
if exist requirements.txt (
    echo ✅ requirements.txt found
) else (
    echo ❌ requirements.txt not found
    pause
    exit /b 1
)

if exist app.py (
    echo ✅ app.py found
) else (
    echo ❌ app.py not found
    pause
    exit /b 1
)

if exist .streamlit\config.toml (
    echo ✅ Streamlit config found
) else (
    echo ❌ Streamlit config not found
    pause
    exit /b 1
)

echo.
echo Step 2: Testing local installation...
echo Installing/updating packages...
call .venv\Scripts\activate 2>nul || (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate
)

pip install -r requirements.txt

echo.
echo Step 3: Testing app startup...
echo Starting Streamlit app (will close automatically)...
timeout /t 3 /nobreak >nul
start "" streamlit run app.py --server.headless true --server.port 8501
timeout /t 10 /nobreak >nul
taskkill /f /im streamlit.exe >nul 2>&1

echo.
echo ====================================
echo   Deployment Ready! ✅
echo ====================================
echo.
echo Your AI Detector is ready for deployment!
echo.
echo Next Steps:
echo 1. Create a GitHub repository
echo 2. Upload your code to GitHub
echo 3. Deploy to Streamlit Community Cloud
echo.
echo See DEPLOYMENT_GUIDE.md for detailed instructions.
echo.
pause
