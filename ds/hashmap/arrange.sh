#!/bin/bash

# Find the highest existing numeric prefix among folders
max_index=0
for d in */ ; do
    if [[ "$d" =~ ^([0-9]+)_ ]]; then
        idx="${BASH_REMATCH[1]}"
        if (( idx > max_index )); then
            max_index=$idx
        fi
    fi
done

# Start counter from next available index
counter=$((max_index + 1))

# Loop through all .py and .cc files in the current directory
for file in *.py *.cc; do
    # Check if the file exists and is not already in a numbered folder
    if [[ -f "$file" ]]; then
        filename="${file%.*}"
        new_folder="${counter}_${filename}"
        mkdir -p "$new_folder"
        mv "$file" "$new_folder/"
        echo "# $filename" > "$new_folder/README.md"
        echo "This folder contains the processed file: $file" >> "$new_folder/README.md"
        ((counter++))
    fi
done