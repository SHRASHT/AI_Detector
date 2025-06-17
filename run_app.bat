@echo off
echo Starting AI Deepfake Detector...
echo.
echo Activating virtual environment...
cd /d "d:\Shreshth WebApps"
call .venv\Scripts\activate.bat
cd AI_Detector
echo.
echo Checking for PyTorch weights...
python create_pytorch_weights.py
echo.
echo Open your browser and go to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.
python -m streamlit run app.py
pause
