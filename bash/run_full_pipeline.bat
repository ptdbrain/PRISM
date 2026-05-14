@echo off
setlocal

rem Thin Windows launcher for the shared Python PRISM full-pipeline runner.
cd /d "%~dp0.."
python scripts\run_full_pipeline.py %*
exit /b %ERRORLEVEL%
