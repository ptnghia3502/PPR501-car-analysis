@echo off
REM =============================================================================
REM run_app.bat - Chạy app Streamlit trên Windows
REM =============================================================================
REM Cách dùng: double-click file này hoặc gõ "run_app.bat" trong cmd
REM =============================================================================

cd /d "%~dp0"
echo Đang chạy app tai: %cd%
echo Mo trinh duyet tai: http://localhost:8501
echo Nhan Ctrl+C de dung app.
echo.
py -m streamlit run app.py
pause
