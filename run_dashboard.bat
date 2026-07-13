@echo off
setlocal
REM ============================================================================
REM  Equity BTST — one command to start the board.
REM
REM   1) checks the Fyers token (it expires ~06:00 IST every day)
REM   2) re-auths in Tradebot ONLY if it is not usable
REM   3) starts the dashboard on :8055
REM
REM  Without a usable token the LIVE tab is blank, so the check comes first.
REM  The BTST and Replay tabs read the EOD archive and work regardless.
REM ============================================================================
cd /d "%~dp0"

echo.
echo [1/3] Checking Fyers token...
python -c "from eqbtst import live; s=live.token_status(); print('      '+s['describe']); raise SystemExit(0 if s['usable'] else 1)"

if errorlevel 1 (
    echo.
    echo [2/3] Token NOT usable -- re-authenticating in Tradebot...
    pushd "d:\Python Projects\Tradebot"
    python fyers_auth.py
    popd

    echo.
    echo       Re-checking...
    python -c "from eqbtst import live; s=live.token_status(); print('      '+s['describe']); raise SystemExit(0 if s['usable'] else 1)"
    if errorlevel 1 (
        echo.
        echo   *** TOKEN STILL NOT USABLE -- the LIVE tab will be blank. ***
        echo   BTST and Replay still work ^(they read the EOD archive^).
        echo.
        pause
    )
) else (
    echo [2/3] Token OK -- no re-auth needed.
)

echo.
echo [3/3] Starting the board on http://127.0.0.1:8055
echo.
python -m streamlit run eqbtst/dashboard.py --server.port 8055

endlocal
