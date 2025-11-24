@echo off
echo Starting OpenCode Web Version...
echo.
echo Starting API server on port 4096...
start /B bun run packages/opencode/src/index.ts serve --port 4096
timeout /t 3 /nobreak > nul
echo Starting web app...
start /B bun run --cwd packages/app dev
echo.
echo Both services started! Open http://localhost:3000 in your browser.
echo Press Ctrl+C to stop both services.
pause
