@echo off
echo Running Credit Risk Analysis in R...
cd /d "%~dp0"
"c:\Program Files\R\R-4.5.2\bin\x64\Rscript.exe" analysis.R
echo.
echo Analysis Complete. Check output/plots for results.
pause
