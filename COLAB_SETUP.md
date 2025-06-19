# Google Colab Setup Guide for AutoStudio

## Quick Fix for Dependency Errors

If you're getting the `torchvision::nms does not exist` or `AutoImageProcessor` import errors, follow these steps:

### Method 1: Automated Fix

```python
# In a Colab cell, run:
!wget https://raw.githubusercontent.com/TheLegendOfKitty/AutoStudio/main/fix_colab_deps.py
!python fix_colab_deps.py
```

Then **restart your runtime**: `Runtime → Restart Runtime`

### Method 2: Manual Fix

Run these commands in separate Colab cells:

```python
# Step 1: Remove conflicting packages
!pip uninstall -y torchvision transformers diffusers accelerate

# Step 2: Install compatible versions
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install transformers>=4.44.0
!pip install diffusers>=0.31.0
!pip install accelerate>=0.33.0

# Step 3: Install GGUF and other dependencies
!pip install gguf safetensors pillow numpy matplotlib opencv-python huggingface_hub psutil inflect
```

**Restart runtime**: `Runtime → Restart Runtime`

### Method 3: Requirements File

```python
# Download and install requirements
!wget https://raw.githubusercontent.com/TheLegendOfKitty/AutoStudio/main/requirements_colab.txt
!pip install -r requirements_colab.txt
```

## Complete Colab Setup

### 1. Clone Repository
```python
!git clone https://github.com/TheLegendOfKitty/AutoStudio.git
%cd AutoStudio
```

### 2. Fix Dependencies (choose one method above)
```python
!python fix_colab_deps.py
# Then restart runtime
```

### 3. Setup Models
```python
!chmod +x setup_models.sh
!./setup_models.sh
```

### 4. Setup GGUF (for memory efficiency)
```python
!chmod +x setup_gguf.sh  
!./setup_gguf.sh
```

### 5. Test Installation
```python
# Verify everything works
import torch, diffusers, transformers
from diffusers import FluxPipeline, GGUFQuantizationConfig
print("✅ All dependencies working!")
```

## Running AutoStudio in Colab

### Basic Usage
```python
# Generate with text
!python run.py --sd_version flux --data_path text_examples.json --device cuda

# Memory-efficient with GGUF quantization
!python run.py --sd_version flux --quantization Q5_K_S --data_path balanced_manga.json --device cuda

# Create manga pages
!python run.py --sd_version flux --quantization Q4_K_S --data_path visual_manga.json --device cuda
```

### Memory Optimization for Colab

Colab has limited VRAM (~15GB), so use quantization:

```python
# For T4 GPU (15GB) - use Q4_K_S or Q5_K_S
!python run.py --sd_version flux --quantization Q4_K_S --data_path my_prompt.json

# For higher-tier Colab (A100) - use Q6_K or Q8_0  
!python run.py --sd_version flux --quantization Q6_K --data_path my_prompt.json
```

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError` or import errors:
1. Run the dependency fix script
2. **Restart runtime** (crucial step!)
3. Try importing again

### CUDA Out of Memory
```python
# Use smaller quantization
!python run.py --sd_version flux --quantization Q3_K_S --data_path my_prompt.json

# Or use CPU (slower but works)
!python run.py --sd_version flux --quantization Q5_K_S --device cpu --data_path my_prompt.json
```

### Download Errors
```python
# Check available quantizations
!python flux_quantization.py

# Use specific working quantization
!python run.py --sd_version flux --quantization Q5_K_S --data_path my_prompt.json
```

### Version Conflicts
If you still get conflicts after the fix:

```python
# Nuclear option - reinstall everything
!pip freeze | grep -E "(torch|diffusers|transformers)" | xargs pip uninstall -y
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install transformers==4.44.0 diffusers==0.30.0 accelerate==0.33.0 gguf
```

## Colab-Specific Tips

### 1. Save Your Work
Colab sessions are temporary. Save important outputs:
```python
# Save generated images to Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy outputs to Drive
!cp -r output /content/drive/MyDrive/AutoStudio_Output
```

### 2. Monitor Resources
```python
# Check GPU memory
!nvidia-smi

# Check disk space  
!df -h
```

### 3. Optimize for Colab
```python
# Use smaller batch sizes
!python run.py --sd_version flux --quantization Q4_K_S --repeats 1

# Use faster model variant
!python run.py --sd_version flux --model_variant schnell --quantization Q5_K_S
```

## Working Example for Colab

Here's a complete working example:

```python
# 1. Setup (run once)
!git clone https://github.com/TheLegendOfKitty/AutoStudio.git
%cd AutoStudio
!python fix_colab_deps.py
# Restart runtime here!

# 2. Generate (after restart)
!python run.py --sd_version flux --quantization Q4_K_S --data_path text_examples.json --device cuda

# 3. View results
from IPython.display import Image, display
import os

output_dir = "output"
for root, dirs, files in os.walk(output_dir):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):
            display(Image(os.path.join(root, file)))
```

## Compatible Version Matrix

| Package | Version | Notes |
|---------|---------|-------|
| torch | ≥2.0.0 | CUDA 12.1 version recommended |
| torchvision | ≥0.15.0 | Must match torch version |
| transformers | 4.44.0 | Specific version for compatibility |
| diffusers | 0.30.0 | Supports GGUF, compatible with transformers |
| accelerate | 0.33.0 | Memory optimization |
| gguf | ≥0.6.0 | For quantized models |

This setup ensures AutoStudio works reliably in Google Colab! 🚀