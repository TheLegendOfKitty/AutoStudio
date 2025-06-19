#!/bin/bash

# AutoStudio Model Setup Script
# Downloads all required models for Flux text generation

set -e  # Exit on any error

echo "🚀 AutoStudio Model Setup"
echo "========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if HuggingFace CLI is installed
print_status "Checking HuggingFace CLI..."
if ! command -v huggingface-cli &> /dev/null; then
    print_error "huggingface-cli not found. Please install it first:"
    echo "pip install huggingface_hub[cli]"
    exit 1
fi

# Check if wget is available
if ! command -v wget &> /dev/null; then
    print_error "wget not found. Please install wget first"
    exit 1
fi

# Create directories if they don't exist
print_status "Creating required directories..."
mkdir -p DETECT_SAM/pretrain/
mkdir -p DETECT_SAM/Grounding-DINO/

# 1. Download Flux.1-dev model
print_status "Downloading Flux.1-dev model (this may take a while, ~24GB)..."
echo "This will download the main Flux model for text-to-image generation"
huggingface-cli download black-forest-labs/FLUX.1-dev --max-workers 1

if [ $? -eq 0 ]; then
    print_success "Flux.1-dev model downloaded successfully"
else
    print_error "Failed to download Flux model"
    exit 1
fi

# 2. Download EfficientSAM model
print_status "Downloading EfficientSAM model..."
echo "This model is used for object segmentation and detection"

if [ -f "DETECT_SAM/pretrain/efficient_sam_s_gpu.jit" ]; then
    print_warning "EfficientSAM model already exists, skipping download"
else
    wget https://huggingface.co/spaces/SkalskiP/YOLO-World/resolve/main/efficient_sam_s_gpu.jit
    
    if [ $? -eq 0 ]; then
        mv efficient_sam_s_gpu.jit DETECT_SAM/pretrain/
        print_success "EfficientSAM model downloaded and moved to correct location"
    else
        print_error "Failed to download EfficientSAM model"
        exit 1
    fi
fi

# 3. Download GroundingDINO model
print_status "Downloading GroundingDINO model..."
echo "This model is used for grounded object detection"

if [ -f "DETECT_SAM/Grounding-DINO/groundingdino_swint_ogc.pth" ]; then
    print_warning "GroundingDINO model already exists, skipping download"
else
    wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
    
    if [ $? -eq 0 ]; then
        mv groundingdino_swint_ogc.pth DETECT_SAM/Grounding-DINO/
        print_success "GroundingDINO model downloaded and moved to correct location"
    else
        print_error "Failed to download GroundingDINO model"
        exit 1
    fi
fi

# 4. Verify all files are in place
print_status "Verifying model files..."

models_ok=true

# Check Flux model (it's in HuggingFace cache, so we check if the download succeeded)
if huggingface-cli list black-forest-labs/FLUX.1-dev --cache-dir ~/.cache/huggingface &> /dev/null; then
    print_success "✓ Flux.1-dev model verified"
else
    print_error "✗ Flux.1-dev model not found in cache"
    models_ok=false
fi

# Check EfficientSAM
if [ -f "DETECT_SAM/pretrain/efficient_sam_s_gpu.jit" ]; then
    print_success "✓ EfficientSAM model verified"
else
    print_error "✗ EfficientSAM model not found"
    models_ok=false
fi

# Check GroundingDINO
if [ -f "DETECT_SAM/Grounding-DINO/groundingdino_swint_ogc.pth" ]; then
    print_success "✓ GroundingDINO model verified"
else
    print_error "✗ GroundingDINO model not found"
    models_ok=false
fi

# Final status
echo ""
echo "📊 Setup Summary"
echo "================"

if [ "$models_ok" = true ]; then
    print_success "All models downloaded successfully!"
    echo ""
    echo "🎉 You can now run AutoStudio with Flux:"
    echo "   python run.py --sd_version flux --device auto --data_path text_examples.json"
    echo ""
    echo "💡 For text generation, try:"
    echo "   python run.py --sd_version flux --device auto --data_path minimal_text.json"
    echo ""
    echo "📂 Models installed:"
    echo "   • Flux.1-dev: ~/.cache/huggingface/hub/"
    echo "   • EfficientSAM: DETECT_SAM/pretrain/efficient_sam_s_gpu.jit"
    echo "   • GroundingDINO: DETECT_SAM/Grounding-DINO/groundingdino_swint_ogc.pth"
else
    print_error "Some models failed to download. Please check the errors above."
    exit 1
fi

echo ""
print_status "Setup complete! 🚀"