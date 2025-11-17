@echo off
echo ========================================
echo SuperAI Extension Installation Script
echo ========================================
echo.

echo 1. Installing VS Code Extension...
echo.

REM Get VS Code extensions directory
set "VSCODE_EXTENSIONS=%USERPROFILE%\.vscode\extensions"
set "EXTENSION_DIR=%VSCODE_EXTENSIONS%\superai-1.0.0"

echo Installing to: %EXTENSION_DIR%
echo.

REM Create extensions directory if it doesn't exist
if not exist "%VSCODE_EXTENSIONS%" (
    mkdir "%VSCODE_EXTENSIONS%"
)

REM Remove existing extension if present
if exist "%EXTENSION_DIR%" (
    echo Removing existing SuperAI extension...
    rmdir /s /q "%EXTENSION_DIR%"
)

REM Copy extension files
echo Copying extension files...
xcopy /s /e /i "superai-extension" "%EXTENSION_DIR%"

echo.
echo 2. Installing Node.js dependencies...
cd "%EXTENSION_DIR%"
npm install

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Start the Advanced MCP Server:
echo    python advanced_mcp_server.py
echo.
echo 2. Restart VS Code
echo.
echo 3. Look for "SuperAI" in the status bar
echo.
echo 4. Try the new Two-Stage Gemini Fix:
echo    Ctrl+Shift+T
echo.
echo ========================================
pause
