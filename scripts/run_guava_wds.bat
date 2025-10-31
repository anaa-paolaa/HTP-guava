@echo off

echo Step 1: Downloading images from Drive/Fotos Scanner

REM Get the current directory
set CURRENT_DIR=%~dp0

python "%CURRENT_DIR%scripts\download_images.py"

echo Step 2: Running PlantCV Workflow

REM Run the workflow (relative to current folder)
python "%USERPROFILE%\anaconda3\Scripts\plantcv-run-workflow" --config "%CURRENT_DIR%scripts\my_config_guava.json"

echo Step 3: Converting JSON to CSV
python "%USERPROFILE%\anaconda3\Scripts\plantcv-utils" json2csv -j "%CURRENT_DIR%Outputs\output.json" -c "%CURRENT_DIR%outputs\raw_results"

echo Step 4: Cleaning data
python "%CURRENT_DIR%scripts\clean_results.py"

echo All steps completed!
pause
