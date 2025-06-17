# GitHub Deployment Script
# Run this in PowerShell to push your AI Detector to GitHub

Write-Host "🚀 AI Deepfake Detector - GitHub Deployment" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is available
try {
    git --version | Out-Null
    Write-Host "✅ Git is available" -ForegroundColor Green
} catch {
    Write-Host "❌ Git is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/download/windows"
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if we're in the right directory
if (!(Test-Path "app.py")) {
    Write-Host "❌ app.py not found. Please run this script from the AI_Detector directory" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✅ Found app.py - we're in the right directory" -ForegroundColor Green

# Check if git is initialized
if (!(Test-Path ".git")) {
    Write-Host "📦 Initializing Git repository..." -ForegroundColor Yellow
    git init
    git branch -M main
}

Write-Host "✅ Git repository ready" -ForegroundColor Green

# Get GitHub repository URL from user
Write-Host ""
Write-Host "🔗 GitHub Repository Setup" -ForegroundColor Cyan
Write-Host "Please create a new repository on GitHub first:" -ForegroundColor Yellow
Write-Host "1. Go to https://github.com/new" -ForegroundColor White
Write-Host "2. Repository name: ai-deepfake-detector (or your choice)" -ForegroundColor White
Write-Host "3. Make it PUBLIC (required for free Streamlit deployment)" -ForegroundColor White
Write-Host "4. Don't initialize with README" -ForegroundColor White
Write-Host ""

$repoUrl = Read-Host "Enter your GitHub repository URL (e.g., https://github.com/username/ai-deepfake-detector.git)"

if ([string]::IsNullOrWhiteSpace($repoUrl)) {
    Write-Host "❌ Repository URL is required" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Add remote origin
try {
    git remote remove origin 2>$null
    git remote add origin $repoUrl
    Write-Host "✅ Added remote origin: $repoUrl" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Remote origin may already exist" -ForegroundColor Yellow
}

# Add all files
Write-Host ""
Write-Host "📁 Adding files to Git..." -ForegroundColor Cyan
git add .

# Check git status
Write-Host ""
Write-Host "📊 Git Status:" -ForegroundColor Cyan
git status --short

# Commit
$commitMessage = Read-Host "Enter commit message (press Enter for default)"
if ([string]::IsNullOrWhiteSpace($commitMessage)) {
    $commitMessage = "🚀 Initial deployment: AI Deepfake Detector with Continuous Learning System"
}

git commit -m $commitMessage

# Push to GitHub
Write-Host ""
Write-Host "🚀 Pushing to GitHub..." -ForegroundColor Cyan
try {
    git push -u origin main
    Write-Host "✅ Successfully pushed to GitHub!" -ForegroundColor Green
} catch {
    Write-Host "❌ Push failed. Please check your GitHub credentials and repository URL" -ForegroundColor Red
    Write-Host "You may need to authenticate with GitHub" -ForegroundColor Yellow
    Read-Host "Press Enter to continue"
}

Write-Host ""
Write-Host "🎉 GitHub Deployment Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "1. 🌐 Go to https://share.streamlit.io" -ForegroundColor White
Write-Host "2. 🔑 Sign in with your GitHub account" -ForegroundColor White
Write-Host "3. 📱 Click 'New app'" -ForegroundColor White
Write-Host "4. 📂 Select your repository: $repoUrl" -ForegroundColor White
Write-Host "5. 📄 Main file path: app.py" -ForegroundColor White
Write-Host "6. 🚀 Click 'Deploy!'" -ForegroundColor White
Write-Host "7. ⏱️  Wait 2-5 minutes for deployment" -ForegroundColor White
Write-Host ""
Write-Host "Your app will be available at:" -ForegroundColor Yellow
Write-Host "https://your-app-name.streamlit.app" -ForegroundColor Green
Write-Host ""

Read-Host "Press Enter to exit"
