@echo off
echo Generating Icon...
python create_icon.py

echo Building Executive...
pyinstaller --noconfirm --onefile --windowed --icon "app_icon.ico" --name "LAZ_Grid_Generator" "laz_grid_generator_gui.py"

echo Build Complete!
pause
