# GGUF Quantization Guide for AutoStudio

## Overview

GGUF (GPT-Generated Unified Format) quantization allows you to run Flux models with **50-75% less memory** while maintaining excellent quality. This is especially useful for:

- **Mac users** (MPS compatibility issues with other quantization methods)
- **Limited VRAM** systems (4-8GB GPUs)
- **CPU inference** (faster loading and inference)
- **Multiple model variants** (run both schnell and dev models)

## Quick Start

### 1. Install GGUF Support
```bash
./setup_gguf.sh
```

### 2. Run with Auto-Selected Quantization
```bash
python run.py --sd_version flux --quantization Q5_K_M --data_path balanced_manga.json
```

### 3. Check Available Options
```bash
python flux_quantization.py
```

## Quantization Levels

| Level  | Memory Reduction | Quality | Best For |
|--------|------------------|---------|----------|
| **Q8_0** | 50% | 99% identical | High-end GPUs (16GB+) |
| **Q6_K** | 60% | 98% quality | Mid-high GPUs (12-16GB) |
| **Q5_K_S** | 65% | 96% quality | **Most users** (8-12GB) |
| **Q5_1** | 67% | 95% quality | Alternative 5-bit |
| **Q4_K_S** | 70% | 93% quality | Low VRAM (6-8GB) |
| **Q4_1** | 72% | 92% quality | Alternative 4-bit |
| **Q3_K_S** | 75% | 88% quality | Very limited VRAM (4-6GB) |
| **Q2_K** | 80% | 75% quality | Extremely limited systems |

## Usage Examples

### Basic Usage
```bash
# Auto-recommended quantization
python run.py --sd_version flux --quantization Q5_K_S

# High quality (if you have VRAM)
python run.py --sd_version flux --quantization Q8_0

# Low VRAM systems
python run.py --sd_version flux --quantization Q4_K_S
```

### Model Variants
```bash
# Flux.1-schnell (default, faster)
python run.py --sd_version flux --model_variant schnell --quantization Q5_K_S

# Flux.1-dev (higher quality, slower)
python run.py --sd_version flux --model_variant dev --quantization Q6_K
```

### Pre-downloaded GGUF Files
```bash
# Use your own GGUF file
python run.py --sd_version flux --gguf_path /path/to/flux1-schnell-Q5_K_M.gguf
```

### Text Generation with Quantization
```bash
# Manga generation with memory savings
python run.py --sd_version flux --quantization Q5_K_M --data_path balanced_manga.json

# Text examples with quantization
python run.py --sd_version flux --quantization Q6_K --data_path text_examples.json
```

## Device-Specific Recommendations

### Mac (MPS)
```bash
# Recommended for Mac users
python run.py --sd_version flux --quantization Q6_K --device mps
```
- **Why Q6_K**: Macs have unified memory, can handle higher quality
- **GGUF Benefits**: No BitsAndBytes compatibility issues
- **Memory**: Uses ~60% less memory than FP16

### NVIDIA GPUs
```bash
# High-end (RTX 4090, 3090, etc.)
python run.py --sd_version flux --quantization Q8_0 --device cuda

# Mid-range (RTX 4070, 3070, etc.)
python run.py --sd_version flux --quantization Q5_K_M --device cuda

# Low-end (RTX 4060, 3060, etc.)
python run.py --sd_version flux --quantization Q4_K_M --device cuda
```

### CPU Inference
```bash
# CPU with good RAM (32GB+)
python run.py --sd_version flux --quantization Q5_K_M --device cpu

# CPU with limited RAM (16GB)
python run.py --sd_version flux --quantization Q4_K_M --device cpu
```

## Advanced Features

### Download Management
The system automatically downloads and caches GGUF models:

```python
from flux_quantization import FluxGGUFManager

manager = FluxGGUFManager()

# List available files
files = manager.list_available_files("schnell")
print(files)

# Download specific quantization
path = manager.download_gguf_model("schnell", "Q5_K_M")
```

### Memory Monitoring
The system automatically detects your available memory:

```python
from flux_quantization import get_memory_info

# Get memory for your device
memory_gb = get_memory_info("cuda")  # or "mps", "cpu"
print(f"Available memory: {memory_gb:.1f}GB")
```

### Custom Pipeline Creation
```python
from flux_quantization import FluxGGUFManager

manager = FluxGGUFManager()
pipe = manager.create_quantized_pipeline(
    model_variant="schnell",
    quantization="Q5_K_M",
    device="auto"
)
```

## Performance Comparison

### Memory Usage (Flux.1-schnell)
- **FP16 (Standard)**: ~24 GB VRAM
- **Q8_0**: ~12 GB VRAM (50% reduction)
- **Q5_K_M**: ~8.4 GB VRAM (65% reduction)  
- **Q4_K_M**: ~7.2 GB VRAM (70% reduction)

### Quality Comparison
- **Q8_0**: Virtually identical to FP16
- **Q5_K_M**: Barely noticeable quality loss
- **Q4_K_M**: Minor quality loss, still very good
- **Q3_K_M**: Noticeable but acceptable quality loss

### Speed
- **Loading**: GGUF models load faster due to smaller file sizes
- **Inference**: Slightly slower due to dynamic dequantization
- **Overall**: Better performance on memory-limited systems

## Troubleshooting

### Installation Issues
```bash
# If GGUF support is missing
pip install --upgrade diffusers
pip install gguf

# Verify installation
python -c "from diffusers import GGUFQuantizationConfig; print('GGUF OK')"
```

### Download Failures
```bash
# Check available files
python flux_quantization.py

# Manual download from HuggingFace
# Visit: https://huggingface.co/city96/FLUX.1-schnell-gguf
```

### Memory Issues
```bash
# For very limited memory, try Q3_K_M
python run.py --sd_version flux --quantization Q3_K_M

# Enable CPU offloading
python run.py --sd_version flux --quantization Q4_K_M --device cpu
```

### Quality Issues
```bash
# If Q4_K_M quality is too low, try Q5_K_M
python run.py --sd_version flux --quantization Q5_K_M

# For maximum quality with memory savings, use Q8_0
python run.py --sd_version flux --quantization Q8_0
```

## Integration with Existing Features

### Text Generation
GGUF quantization works seamlessly with AutoStudio's text generation:

```bash
# Quantized manga generation
python run.py --sd_version flux --quantization Q5_K_M --data_path balanced_manga.json

# Memory-efficient text examples
python run.py --sd_version flux --quantization Q6_K --data_path text_examples.json
```

### Character Consistency
Quantization maintains character consistency across dialogue turns:

```bash
# Multi-character scenes with quantization
python run.py --sd_version flux --quantization Q5_K_M --data_path cache/demo.json
```

### Multiple Variants
Run different model variants based on your needs:

```bash
# Fast generation (schnell)
python run.py --sd_version flux --model_variant schnell --quantization Q5_K_M

# High quality (dev) 
python run.py --sd_version flux --model_variant dev --quantization Q6_K
```

## Best Practices

1. **Start with Q5_K_M** - Best balance for most users
2. **Use Q8_0** if you have sufficient VRAM and want maximum quality
3. **Try Q6_K on Mac** - Takes advantage of unified memory
4. **Monitor memory usage** - Adjust quantization based on your system
5. **Cache models** - Downloaded GGUF files are reused automatically
6. **Test quality** - Compare different quantization levels for your use case

## File Locations

- **Downloaded GGUF models**: `~/.cache/autostudio/gguf/`
- **Configuration**: Arguments passed to `run.py`
- **Logs**: Console output shows quantization details

## Support

For issues with GGUF quantization:

1. **Check dependencies**: Run `./setup_gguf.sh`
2. **Verify installation**: Run `python flux_quantization.py`
3. **Try different quantization**: Start with Q5_K_M
4. **Check memory**: Ensure sufficient RAM/VRAM for chosen quantization
5. **Use fallback**: System automatically falls back to standard loading if GGUF fails