@echo off
set "VENV_DIR=build_venv"

echo Creating clean virtual environment...
python -m venv %VENV_DIR%

echo Activating environment...
call %VENV_DIR%\Scripts\activate

echo Installing minimal dependencies...
python -m pip install --upgrade pip
pip install -r requirements_clean.txt

echo Generating Icon...
python create_icon.py

echo Building Executive...
pyinstaller --noconfirm LAZ_Grid_Generator_Clean_v0.7.spec

echo Deactivating...
deactivate

echo Build Complete!
pause
