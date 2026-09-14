#!/bin/bash

# Define variables
url="https://drive.google.com/uc?id=1wKrSQHif96ucColaRkebKTQUZ84g1weY"
main_dir="/home/GAN_Writing"
dest_dir="$main_dir/datasets/TC_Words"
dataset_file_path="$dest_dir/TC_Words.rar"
extracted_dir="$dest_dir/Words"
new_dir_name="$dest_dir/words"

# Create destination directory
mkdir -p "$dest_dir"

# Download dataset
echo "Downloading dataset..."
gdown --output "$dataset_file_path" "$url"
echo "Dataset downloaded successfully."

# Extract dataset
echo "Extracting dataset..."
unrar x "$dataset_file_path" -d "$dest_dir"
echo "Dataset extracted successfully."

# Rename extracted folder
echo "Renaming extracted folder..."
mv "$extracted_dir" "$new_dir_name"
echo "Folder renamed successfully."

# After that run this on terminal
# rm -rf TC_Words.rar 