@echo off
REM Register the daily paper loop in Windows Task Scheduler (runs 20:00 IST).
REM Run this ONCE (right-click -> Run as administrator is safest). To change the
REM time, edit /ST. To remove: schtasks /Delete /TN "EqBTST Paper Loop" /F
schtasks /Create /TN "EqBTST Paper Loop" ^
  /TR "\"d:\Python Projects\Equity_Intra_BTST\paper_loop.bat\"" ^
  /SC DAILY /ST 20:00 /F
echo.
echo Registered. It runs daily at 20:00 (only while the PC is on).
echo Check the record any time:  python -m eqbtst.cli scorecard
echo Or open the dashboard BTST tab (the paper ledger panel).
pause
