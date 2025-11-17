@echo off
echo ========================================
echo SuperAI Extension Packaging Script v3.0
echo ========================================
echo.

echo Checking for VS Code Extension Manager (vsce)...
where vsce >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing VS Code Extension Manager...
    npm install -g @vscode/vsce
    if %errorlevel% neq 0 (
        echo Failed to install vsce. Please install it manually:
        echo npm install -g @vscode/vsce
        pause
        exit /b 1
    )
)

echo.
echo Packaging SuperAI Extension v3.0...
echo Features included:
echo - Three-Stage Gemini System (Understand → Fix → Complete)
echo - Multi-API Key Support with Load Balancing
echo - Stage-Specific API Keys
echo - API Key Status Monitoring
echo - Enhanced VS Code Integration
echo.

vsce package --out superai-3.0.0.vsix
if %errorlevel% eq 0 (
    echo.
    echo ✅ SUCCESS! Extension packaged successfully!
    echo 📦 File created: superai-3.0.0.vsix
    echo.
    echo To install the extension:
    echo 1. Open VS Code
    echo 2. Go to Extensions (Ctrl+Shift+X)
    echo 3. Click the "..." menu → "Install from VSIX..."
    echo 4. Select the superai-3.0.0.vsix file
    echo.
    echo New features in v3.0:
    echo • Three-Stage Gemini Fix & Complete (Ctrl+Shift+3)
    echo • API Key Status Monitor (Ctrl+Shift+K)
    echo • Multi-API Key Load Balancing
    echo • Enhanced Code Understanding
    echo.
) else (
    echo ❌ ERROR: Failed to package extension
    echo Please check for any errors above
)

echo.
pause
