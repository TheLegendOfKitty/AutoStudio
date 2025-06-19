#!/bin/bash

# GGUF Quantization Setup for AutoStudio
# Installs required dependencies for GGUF support

set -e

echo "🎯 Setting up GGUF Quantization Support"
echo "======================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "Python version: $python_version"

# Install GGUF support
print_status "Installing GGUF dependencies..."

# Install core packages
pip install --upgrade diffusers transformers accelerate torch

# Install GGUF support specifically
print_status "Installing GGUF format support..."
pip install gguf

# Verify installation
print_status "Verifying GGUF installation..."
python3 -c "
try:
    from diffusers import GGUFQuantizationConfig
    print('✅ GGUFQuantizationConfig available')
except ImportError as e:
    print('❌ GGUF support not available:', e)
    exit(1)
"

if [ $? -eq 0 ]; then
    print_success "GGUF support installed successfully!"
else
    print_error "GGUF installation failed"
    exit 1
fi

# Install additional dependencies for memory monitoring
print_status "Installing additional dependencies..."
pip install psutil huggingface_hub

print_success "Setup complete! 🎉"

echo ""
echo "📊 Available GGUF Quantization Levels:"
echo "======================================="
echo "Q8_0:   50% memory reduction, 99% quality (recommended for high-end)"
echo "Q6_K:   60% memory reduction, 98% quality (excellent balance)"
echo "Q5_K_M: 65% memory reduction, 97% quality (recommended for most users)"
echo "Q4_K_M: 70% memory reduction, 94% quality (good for low VRAM)"
echo "Q3_K_M: 75% memory reduction, 90% quality (very limited VRAM)"

echo ""
echo "🚀 Usage Examples:"
echo "=================="
echo "# Auto-select quantization based on your device:"
echo "python run.py --sd_version flux --quantization Q5_K_M"
echo ""
echo "# Use specific model variant:"
echo "python run.py --sd_version flux --model_variant dev --quantization Q8_0"
echo ""
echo "# Use pre-downloaded GGUF file:"
echo "python run.py --sd_version flux --gguf_path /path/to/model.gguf"
echo ""
echo "# Check available quantizations:"
echo "python flux_quantization.py"

echo ""
print_success "Ready to use GGUF quantization! 🎯"