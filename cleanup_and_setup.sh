#!/bin/bash

# Script to clean up base environment and setup new conda environment
# Author: Generated for DL Project cleanup

echo "========================================="
echo "Conda Environment Cleanup and Setup"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Step 1: Getting list of packages to uninstall from base environment${NC}"
echo ""

# Extract package names from requirements.txt (without version specifiers)
packages_to_remove=(
    "transformers"
    "accelerate"
    "datasets"
    "peft"
    "torch"
    "torchvision"
    "sentencepiece"
    "scikit-learn"
    "faiss-cpu"
    "numpy"
    "pydantic"
    "protobuf"
    "rich"
    "typer"
    "pytest"
)

echo -e "${YELLOW}The following packages will be uninstalled from base environment:${NC}"
for pkg in "${packages_to_remove[@]}"; do
    echo "  - $pkg"
done
echo ""

read -p "Do you want to proceed with uninstalling these packages from base? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo -e "${GREEN}Uninstalling packages from base environment...${NC}"
    
    # Deactivate any active conda environment to ensure we're in base
    eval "$(conda shell.bash hook)"
    conda activate base
    
    # Uninstall pip packages from base
    for pkg in "${packages_to_remove[@]}"; do
        echo "Uninstalling $pkg..."
        pip uninstall -y "$pkg" 2>/dev/null || echo "  (not found or already removed)"
    done
    
    echo -e "${GREEN}Cleanup of base environment complete!${NC}"
    echo ""
else
    echo -e "${YELLOW}Skipping base environment cleanup.${NC}"
    echo ""
fi

echo -e "${YELLOW}Step 2: Setting up dl_project environment${NC}"
echo ""

# Activate the new environment
eval "$(conda shell.bash hook)"
conda activate dl_project

echo -e "${GREEN}Activated dl_project environment${NC}"
echo ""

read -p "Do you want to install requirements in dl_project environment? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo -e "${GREEN}Installing packages in dl_project environment...${NC}"
    pip install -r requirements.txt
    echo ""
    echo -e "${GREEN}Installation complete!${NC}"
else
    echo -e "${YELLOW}Skipping package installation.${NC}"
    echo "You can install them later with: pip install -r requirements.txt"
fi

echo ""
echo "========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================="
echo ""
echo "To use your new environment:"
echo "  conda activate dl_project"
echo ""
echo "To deactivate:"
echo "  conda deactivate"
echo ""
