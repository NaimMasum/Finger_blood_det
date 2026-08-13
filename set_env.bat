@echo off

REM Install Miniconda using winget
winget install --id Anaconda.Miniconda3 -e

REM Initialize conda
call "%USERPROFILE%\miniconda3\Scripts\activate.bat"

REM Create environment from YAML
conda env create -f environment.yml

REM Activate environment
call conda activate fingerprint

echo.
echo =====================================
echo Environment setup complete!
echo =====================================

pause