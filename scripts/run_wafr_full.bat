@echo off
REM Complete WAFR Pipeline Runner - Windows Batch Script
REM Includes all latest features: HRI validation, strict quality control, AG-UI, lens detection

echo ======================================================================
echo WAFR Complete Pipeline Runner
echo ======================================================================
echo.

REM Check if client name is provided
if "%1"=="" (
    echo Usage: run_wafr_full.bat "Client Name"
    echo.
    echo Example: run_wafr_full.bat "My Company"
    echo.
    echo This will run the complete WAFR pipeline with:
    echo   - AG-UI event streaming
    echo   - Strict quality control (confidence ^>= 0.7)
    echo   - HRI validation (filters non-tangible HRIs)
    echo   - Automatic lens detection
    echo   - WA Tool integration and PDF generation
    echo.
    pause
    exit /b 1
)

set CLIENT_NAME=%1

echo Running WAFR pipeline for: %CLIENT_NAME%
echo.
echo Features enabled:
echo   - AG-UI Integration
echo   - Strict Quality Control (confidence ^>= 0.7)
echo   - HRI Validation (Claude-based)
echo   - Automatic Lens Detection
echo   - WA Tool Integration
echo   - PDF Report Generation
echo.
echo Starting pipeline...
echo.

python run_wafr_full.py --wa-tool --client-name "%CLIENT_NAME%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================================================
    echo Pipeline completed successfully!
    echo ======================================================================
) else (
    echo.
    echo ======================================================================
    echo Pipeline completed with errors. Check logs for details.
    echo ======================================================================
)

pause
