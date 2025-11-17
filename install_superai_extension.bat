@echo off
echo ========================================
echo SuperAI Extension v2.0.0 Installation
echo ========================================
echo.
echo 🚀 New Features in v2.0.0:
echo   • Two-Stage Gemini Fix & Complete (Ctrl+Shift+T)
echo   • Complete TODO Implementation
echo   • Visual Results Panel
echo   • Zero TODO Policy
echo.

echo Installing SuperAI Extension v2.0.0...
echo.

REM Install the VSIX extension
code --install-extension "superai-extension\superai-2.0.0.vsix"

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next Steps:
echo 1. Start the Advanced MCP Server:
echo    python advanced_mcp_server.py
echo.
echo 2. Restart VS Code (if already open)
echo.
echo 3. Look for "SuperAI" in the status bar
echo.
echo 4. Try the NEW Two-Stage Gemini Fix:
echo    • Open a file with TODO comments
echo    • Press Ctrl+Shift+T
echo    • Watch it fix syntax AND implement TODOs!
echo.
echo 5. Available Commands:
echo    • Ctrl+Shift+T - Two-Stage Gemini Fix
echo    • Ctrl+Shift+Space - Complete Patterns
echo    • Ctrl+Shift+F - Auto-Fix & Complete
echo    • Ctrl+Shift+A - Analyze File
echo.
echo ========================================
echo Ready to eliminate ALL TODO comments! 🎉
echo ========================================
pause
