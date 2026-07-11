@echo off
REM ============================================================================
REM  Self-running BTST paper loop. Run once daily ~20:00 IST (after the DCM EOD
REM  sync, so today's close is in the archive). No Fyers token needed.
REM    1) reconcile  -> fill yesterday's positions at today's open (now in DCM)
REM    2) emit       -> log tonight's leak-free BTST-CARRY candidates (entry=close)
REM    3) scorecard  -> update P&L + edge-health + stale-open integrity guard
REM  Nothing auto-executes real orders — this is a PAPER track record.
REM ============================================================================
cd /d "d:\Python Projects\Equity_Intra_BTST"
echo. >> paper_loop.log
echo ==================== %date% %time% ==================== >> paper_loop.log
python -m eqbtst.cli reconcile >> paper_loop.log 2>&1
python -m eqbtst.cli emit       >> paper_loop.log 2>&1
python -m eqbtst.cli scorecard  >> paper_loop.log 2>&1
python -m eqbtst.cli calibrate  >> paper_loop.log 2>&1
echo (paper loop done) >> paper_loop.log
