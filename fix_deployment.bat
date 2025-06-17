@echo off
echo ==========================================
echo   FIXING DEPLOYMENT ISSUES - AI Detector
echo ==========================================
echo.

echo 🔧 Fixed issues identified from Streamlit Cloud logs:
echo   - Pillow version conflict (changed to ^10.0.0)
echo   - PyTorch CPU version format (removed +cpu suffix)
echo   - Added opencv-python-headless for cloud compatibility
echo   - Updated numpy/pandas version constraints
echo.

REM Check if git is available
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Git is not installed or not in PATH
    echo Please install Git from: https://git-scm.com/download/windows
    pause
    exit /b 1
)
echo ✅ Git is available

REM Check if we're in the right directory
if not exist "app.py" (
    echo ❌ app.py not found. Please run this script from the AI_Detector directory
    pause
    exit /b 1
)
echo ✅ Found app.py - we're in the right directory

echo.
echo 📝 Staging fixed requirements.txt...
git add requirements.txt .streamlit/config.toml

echo.
echo 📊 Changes to be committed:
git status --short

echo.
set /p commitMessage="Enter commit message (press Enter for default): "
if "%commitMessage%"=="" (
    set commitMessage=🔧 Fix deployment issues: Update dependencies for Streamlit Cloud compatibility
)

git commit -m "%commitMessage%"

echo.
echo 🚀 Pushing fixes to GitHub...
git push

if %errorlevel% equ 0 (
    echo ✅ Successfully pushed fixes to GitHub!
    echo.
    echo 🎉 Your Streamlit Cloud app should now deploy successfully!
    echo.
    echo Next steps:
    echo 1. 🌐 Go to your Streamlit Cloud dashboard
    echo 2. 🔄 The app should automatically redeploy with the fixes
    echo 3. ⏱️ Wait 2-3 minutes for deployment to complete
    echo 4. ✅ Your app should now work without dependency conflicts
) else (
    echo ❌ Push failed. Please check your Git setup.
)

echo.
echo 📋 Fixed Dependencies:
echo   - streamlit==1.39.0
echo   - torch==2.7.1 (no +cpu suffix)
echo   - pillow^=7.1.0,^11 (compatible with Streamlit)
echo   - opencv-python-headless (cloud-compatible)
echo.

pause
