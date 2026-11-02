@echo off
cd /d "%~dp0"
echo Installing dependencies into your system Python...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Setup complete! Run run.bat to start the app.
pause