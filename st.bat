@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
if defined FRAM_PYTHON goto custom_python
if exist ".venv\Scripts\python.exe" goto venv_python
python "%~dp0main.py" %*
goto finished
:custom_python
"%FRAM_PYTHON%" "%~dp0main.py" %*
goto finished
:venv_python
"%~dp0.venv\Scripts\python.exe" "%~dp0main.py" %*
:finished
set "FRAM_EXIT=%ERRORLEVEL%"
if not "%FRAM_EXIT%"=="0" pause
exit /b %FRAM_EXIT%
