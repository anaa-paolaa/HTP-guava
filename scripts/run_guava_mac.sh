#!/bin/bash

echo "Step 1: Download images from Drive/Fotos Scanner"
# Get the directory of the current script
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#python "$CURRENT_DIR/scripts/download_images.py"

echo "Step 2: Running PlantCV Workflow"

# Run PlantCV workflow using relative path
python "$HOME/anaconda3/bin/plantcv-run-workflow" --config "$CURRENT_DIR/scripts/my_config_guava.json"

echo "Step 3: Converting JSON to CSV"
python "$HOME/anaconda3/bin/plantcv-utils" json2csv -j "$CURRENT_DIR/outputs/output.json" -c "$CURRENT_DIR/outputs/raw_results"

echo "Step 4: Cleaning data"
python "$CURRENT_DIR/scripts/clean_results.py"

echo "All steps completed!"

