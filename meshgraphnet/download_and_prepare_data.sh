#!/bin/bash
# Download DeepMind MeshGraphNet datasets directly from Google Cloud Storage
# Datasets are hosted at: https://storage.googleapis.com/dm-meshgraphnets/

set -e

BASE_URL="https://storage.googleapis.com/dm-meshgraphnets"
DATA_DIR="data"

echo "=========================================="
echo "Downloading DeepMind MeshGraphNet Datasets"
echo "=========================================="
echo ""

# Create data directory
mkdir -p "${DATA_DIR}"

download_dataset() {
    local dataset_name=$1
    # DeepMind datasets have nested structure: dataset_name/dataset_name/
    local output_dir="${DATA_DIR}/${dataset_name}/${dataset_name}"
    
    echo "Downloading ${dataset_name} dataset..."
    echo "  Output: ${output_dir}"
    
    mkdir -p "${output_dir}"
    
    # Download required files
    for file in meta.json train.tfrecord valid.tfrecord test.tfrecord; do
        local url="${BASE_URL}/${dataset_name}/${file}"
        local output_file="${output_dir}/${file}"
        
        echo "  Downloading ${file}..."
        if command -v wget &> /dev/null; then
            wget -q --show-progress -O "${output_file}" "${url}"
        elif command -v curl &> /dev/null; then
            curl -L --progress-bar -o "${output_file}" "${url}"
        else
            echo "Error: Neither wget nor curl found. Please install one."
            exit 1
        fi
        
        if [ ! -f "${output_file}" ]; then
            echo "  Error: Failed to download ${file}"
            exit 1
        fi
    done
    
    echo "  ${dataset_name} downloaded successfully"
    echo ""
}

# Download CylinderFlow dataset (~16 GB)
download_dataset "cylinder_flow"

# Download Flag dataset (~8 GB)
download_dataset "flag_simple"

echo "=========================================="
echo "All datasets downloaded successfully"
echo "=========================================="
echo ""
echo "Datasets are located at:"
echo "  - ${DATA_DIR}/cylinder_flow/cylinder_flow/"
echo "  - ${DATA_DIR}/flag_simple/flag_simple/"
echo ""
echo "Next steps:"
echo "  1. Create splits:"
echo "     python create_splits.py --dataset cylinder --output data/splits/cylinder_splits.json"
echo "     python create_splits.py --dataset flag --output data/splits/flag_splits.json"
echo ""
echo "  2. Filter datasets:"
echo "     python filter_trajectories_tf1.py --splits_file data/splits/cylinder_splits.json \\"
echo "       --input_dir ${DATA_DIR}/cylinder_flow/cylinder_flow \\"
echo "       --output_dir data/cylinder_flow_filtered --dataset cylinder"
echo ""
echo "     python filter_trajectories_tf1.py --splits_file data/splits/flag_splits.json \\"
echo "       --input_dir ${DATA_DIR}/flag_simple/flag_simple \\"
echo "       --output_dir data/flag_simple_filtered --dataset flag"
echo ""
