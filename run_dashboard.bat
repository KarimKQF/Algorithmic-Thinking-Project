@echo off
echo Lancement du Dashboard Credit Risk...
cd /d "%~dp0"
python -m streamlit run dashboard.py
pause
