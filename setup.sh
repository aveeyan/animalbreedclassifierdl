#!/bin/bash

# Set repository URL
REPO_URL="https://github.com/ml4py/dataset-iiit-pet.git"
DATASET_DIR="dataset-iiit-pet"

# Clone the repository if it doesn't exist
if [ ! -d "$DATASET_DIR" ]; then
    echo "Cloning Oxford-IIIT Pet Dataset..."
    git clone "$REPO_URL"
else
    echo "Dataset repository already exists. Pulling latest changes..."
    cd "$DATASET_DIR" && git pull && cd ..
fi

# Check if requirements.txt exists and install dependencies
if [ -f "$DATASET_DIR/requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r "$DATASET_DIR/requirements.txt"
else
    echo "requirements.txt not found in the repository."
fi

echo "Dataset download and setup complete."
