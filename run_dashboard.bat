@echo off
REM Equity BTST + live intraday board. Fresh Fyers token needed for the live tab
REM (re-auth in Tradebot each morning: python fyers_auth.py).
python -m streamlit run eqbtst/dashboard.py --server.port 8055
