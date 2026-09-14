#!/bin/sh

# Define variables
url="https://git.io/J0fjL"
main_dir="/home/GAN_Writing"
dest_dir="$main_dir/datasets/iam"
dataset_file_path="$dest_dir/IAM_Words.zip"
file_path="$dest_dir/IAM_Words/words.tgz"
extract_path="$dest_dir/words"

# Create destination directory
mkdir -p "$dest_dir"

# Download dataset
echo "Downloading dataset..."
wget -O "$dataset_file_path" "$url"
echo "Dataset downloaded successfully."

# Create extraction directory
mkdir -p "$extract_path"

# Extract dataset
echo "Extracting dataset..."
unzip "$dataset_file_path" -d "$dest_dir"
tar -xzf "$file_path" -C "$extract_path"
echo "Dataset extracted successfully."
