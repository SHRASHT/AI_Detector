@echo off
echo ==========================================
echo   AI Deepfake Detector - GitHub Deploy
echo ==========================================
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

REM Initialize git if needed
if not exist ".git" (
    echo 📦 Initializing Git repository...
    git init
    git branch -M main
)
echo ✅ Git repository ready

echo.
echo 🔗 GitHub Repository Setup
echo Please create a new repository on GitHub first:
echo 1. Go to https://github.com/new
echo 2. Repository name: ai-deepfake-detector (or your choice)
echo 3. Make it PUBLIC (required for free Streamlit deployment)
echo 4. Don't initialize with README
echo.

set /p repoUrl="Enter your GitHub repository URL: "

if "%repoUrl%"=="" (
    echo ❌ Repository URL is required
    pause
    exit /b 1
)

REM Add remote origin
git remote remove origin >nul 2>&1
git remote add origin %repoUrl%
echo ✅ Added remote origin: %repoUrl%

REM Add all files
echo.
echo 📁 Adding files to Git...
git add .

REM Show status
echo.
echo 📊 Git Status:
git status --short

REM Commit
echo.
set /p commitMessage="Enter commit message (press Enter for default): "
if "%commitMessage%"=="" (
    set commitMessage=🚀 Initial deployment: AI Deepfake Detector with Continuous Learning System
)

git commit -m "%commitMessage%"

REM Push to GitHub
echo.
echo 🚀 Pushing to GitHub...
git push -u origin main

if %errorlevel% equ 0 (
    echo ✅ Successfully pushed to GitHub!
) else (
    echo ❌ Push failed. Please check your GitHub credentials and repository URL
    echo You may need to authenticate with GitHub
)

echo.
echo 🎉 GitHub Deployment Complete!
echo ==========================================
echo.
echo Next Steps:
echo 1. 🌐 Go to https://share.streamlit.io
echo 2. 🔑 Sign in with your GitHub account
echo 3. 📱 Click 'New app'
echo 4. 📂 Select your repository
echo 5. 📄 Main file path: app.py
echo 6. 🚀 Click 'Deploy!'
echo 7. ⏱️ Wait 2-5 minutes for deployment
echo.
echo Your app will be available at:
echo https://your-app-name.streamlit.app
echo.

pause
