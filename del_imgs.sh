#!/bin/bash

# Directory containing the images
folder_path="./imgs"

# Move to the directory
cd "$folder_path" || exit

# Loop through files in the directory
for file in *; do
    # Check if the filename contains '2000' or 'epoch'
    if [[ $file == *"2000"* ]] || [[ $file == *"epoch"* ]]; then
        echo "Keeping: $file"
    else
        # Remove files that don't meet the criteria
        echo "Removing: $file"
        rm "$file"
    fi
done
