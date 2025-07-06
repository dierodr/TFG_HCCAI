#!/bin/bash

# Get the current directory name
CURRENT_DIR=$(basename "$PWD")
TARGET_DIR="Convert" 

# Check if the script is running inside a directory named "Convert"
if [ "$CURRENT_DIR" != $TARGET_DIR ]; then
    echo "Error: This script must be run inside a directory named $TARGET_DIR."
    echo "Current directory: $CURRENT_DIR"
    exit 1
fi

# Get the current working directory
ROOT_DIR=$(pwd)

# Echo info before executing
echo "We are in $ROOT_DIR"
echo -n "Number of files before executing: "
ls $ROOT_DIR | wc -l

# Move all files from subdirectories to the root directory
find "$ROOT_DIR" -mindepth 2 -type f -exec mv -t "$ROOT_DIR" {} +

echo "All files moved to $ROOT_DIR." 
# Remove empty directories
find "$ROOT_DIR" -mindepth 1 -type d -empty -delete
echo "Empty directories removed."

#Remove videos
rm *.wmv
echo "Video files removed."


mkdir Bazo Cálculos_y_pólipos_en_la_vesicula Higado_con_cirrosis Higado_con_esteatosis Higado_con_hepatocarcinoma Higado_sano Lesiones_hepaticas_benignas Pancreas_normal Riñón

#Echo info after the executiong
echo -n "The followind directories were created: "
ls -d */

echo -n "Number of files after executing: "
ls $ROOT_DIR | wc -l
