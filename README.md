# HRM-Models: Hierarchical Reasoning Model for Text Generation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c4exU-zMt4SuT1kRlwQQXlLPaiazEDCf?usp=sharing)

A family of transformer models with Hierarchical Reasoning Module (HRM) architecture trained on multiple high-quality text datasets. These models feature adaptive computation with pondering mechanisms for improved text generation quality, available in multiple sizes from 10M to 1B parameters.

## 📊 Model Family

**HRM-Models** comes in 6 different sizes optimized for various use cases:

| Model Size | Parameters | Architecture | Memory Usage | Best Use Case |
|------------|------------|--------------|--------------|---------------|
| **Micro-10M** | 10.3M | HRM + Pondering | ~500MB | Research, prototyping |
| **Nano-25M** | 25.6M | HRM + Pondering | ~1GB | Mobile, edge devices |
| **Small-50M** | 53.2M | HRM + Pondering | ~2GB | General purpose |
| **Medium-100M** | 106.4M | HRM + Pondering | ~4GB | Production inference |
| **Medium-350M** | 353.8M | HRM + Pondering | ~12GB | High-quality generation |
| **Large-1B** | 1.06B | HRM + Pondering | ~32GB | State-of-the-art results |

### ⚡ Performance Optimizations

All models include:
- **Flash Attention** support (when available)
- **Multi-GPU** distributed training  
- **Mixed Precision** (BF16/FP16) training
- **Optimized DataLoaders** with intelligent worker management
- **TensorBoard** integration for monitoring
- **Fast HF Hub transfers** with hf_transfer

## 🏗️ Model Architecture

**HRM-Models** implements a novel hierarchical reasoning architecture with the following key components:

### Core Architecture
- **Hierarchical Reasoning Module** with dual-stream processing
- **Adaptive Computation**: Pondering mechanism with halt probabilities  
- **SwiGLU Activation**: Enhanced non-linear transformations
- **RMSNorm**: Improved normalization for stable training
- **Context Length**: 512 tokens
- **Vocabulary**: 32,100 tokens (T5 tokenizer)

### Model Specifications by Size

#### Micro-10M (Development & Testing)
- **Embeddings**: 256 dimensions
- **Layers**: 6 
- **Attention Heads**: 4
- **Feed-Forward**: 1024 dimensions
- **Max Pondering Steps**: 4

#### Nano-25M (Mobile & Edge)
- **Embeddings**: 384 dimensions  
- **Layers**: 8
- **Attention Heads**: 6
- **Feed-Forward**: 1536 dimensions
- **Max Pondering Steps**: 6

#### Small-50M (General Purpose)
- **Embeddings**: 512 dimensions
- **Layers**: 8
- **Attention Heads**: 8
- **Feed-Forward**: 2048 dimensions  
- **Max Pondering Steps**: 6

#### Medium-100M & 350M (Production)
- **Embeddings**: 768 dimensions
- **Layers**: 12
- **Attention Heads**: 12
- **Feed-Forward**: 3072 dimensions
- **Max Pondering Steps**: 8

#### Large-1B (State-of-the-art)
- **Embeddings**: 1024 dimensions
- **Layers**: 16
- **Attention Heads**: 16  
- **Feed-Forward**: 4096 dimensions
- **Max Pondering Steps**: 10

## 📚 Dataset Support

### Primary Dataset: C4-English
- **Source**: Common Crawl filtered text (allenai/c4)
- **Size**: 365M training samples, 3.65M validation samples
- **Language**: English (high-quality web content)
- **Streaming**: Memory-efficient streaming for large-scale training

### Additional Supported Datasets

| Dataset | Description | Type | Language |
|---------|-------------|------|----------|
| **c4** | Common Crawl multilingüe | HF Streaming | Multi |
| **openwebtext** | OpenWebText dataset | HF | English |
| **pile** | EleutherAI's Pile dataset | HF | English |
| **fineweb** | High-quality web text | HF | Multi |
| **slimpajama** | SlimPajama 627B tokens | HF Streaming | Multi |
| **human_conversations** | Kaggle conversations | Kaggle | English |

### Dataset Mixing Strategies

For **50M+ models**, custom dataset mixes are available:
- **high_quality_small**: Curated high-quality subset
- **balanced_small**: Balanced multi-domain mix  
- **dev_small**: Development and testing mix
- **conversation_small**: Dialog-focused training

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
```txt
torch>=2.6.0
transformers>=4.55.2
datasets>=4.0.0
huggingface_hub>=0.34.4
sentencepiece>=0.1.99
protobuf>=3.20.0
tqdm>=4.67.1
```

**Optional (Recommended):**
```txt
hf_transfer>=0.1.0          # Faster HF Hub downloads
tensorboard>=2.14.0         # Training visualization
langdetect>=1.0.9          # Language filtering
kagglehub>=0.2.0           # Kaggle dataset support
flash-attn>=2.0.0          # GPU attention optimization
```

### Usage Examples

#### Basic Text Generation
```python
import os
os.environ['HRM_IMPORT_ONLY'] = '1'  # Fast import mode

from hrm_training_small_50m import HRMText1, HRMText1Config
from transformers import T5Tokenizer

# Load model and tokenizer
config = HRMText1Config()
model = HRMText1(config)
tokenizer = T5Tokenizer.from_pretrained("t5-small", use_fast=False, legacy=False)

# Generate text
prompt = "The future of artificial intelligence"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
```

#### Training from Scratch

**Environment Setup:**
```bash
# Optimal performance settings
export HF_HUB_ENABLE_HF_TRANSFER=1  # Fast HF downloads
export HF_TOKEN="your_huggingface_token"  # For model uploads
export HRM_OUTPUT_BASE="/path/to/output"  # Custom output path

# Disable import mode for training
unset HRM_IMPORT_ONLY
```

**Single GPU Training:**
```bash
python hrm_training_small_50m.py
```

**Multi-GPU Training:**
```bash
torchrun --nproc_per_node=2 hrm_training_medium_350m.py
```

**Google Colab Training:**
```python
# Colab automatically detects and optimizes for the environment
!python hrm_training_nano_25m.py
```

## ⚙️ Advanced Configuration

### Training Parameters by Model Size

#### Development Models (Micro-10M, Nano-25M)
```python
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
LEARNING_RATE = 5e-4
EPOCHS = 2
```

#### Production Models (Small-50M to Large-1B)  
```python
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 3e-4
EPOCHS = 4
EARLY_STOPPING_PATIENCE = 3
```

### Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `HRM_IMPORT_ONLY` | Fast import mode (no training setup) | - | `1` |
| `HF_HUB_ENABLE_HF_TRANSFER` | Fast HF Hub transfers | `1` | `1` |
| `HF_TOKEN` | Hugging Face API token | - | `hf_xxx...` |
| `HRM_OUTPUT_BASE` | Custom output directory | `./HRM_Models` | `/data/models` |

### Hardware Optimization

#### GPU Support
- **NVIDIA H200/H100**: Optimal performance with TF32 precision
- **NVIDIA A100/A6000**: Full BF16 mixed precision support  
- **RTX 4090/3090**: FP16 mixed precision recommended
- **RTX 3080/2080**: Reduce batch size, use gradient checkpointing

#### Multi-GPU Setup
```bash
# Distributed training with proper environment
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=localhost
export MASTER_PORT=29500

torchrun --nproc_per_node=2 hrm_training_large_1b.py
```

#### Memory Management
- **Micro-10M**: 4GB VRAM minimum
- **Nano-25M**: 8GB VRAM minimum  
- **Small-50M**: 12GB VRAM minimum
- **Medium-100M**: 16GB VRAM minimum
- **Medium-350M**: 32GB VRAM minimum
- **Large-1B**: 64GB VRAM minimum

## 📈 Training Features

### Advanced Optimizations
- **Adaptive Learning Rate**: Cosine annealing with warmup
- **Gradient Clipping**: Automatic norm clipping at 1.0
- **Early Stopping**: Validation-based with configurable patience
- **Checkpointing**: Automatic save/resume with best model tracking  
- **Mixed Precision**: BF16/FP16 support for memory efficiency
- **Flash Attention**: When available (2x faster attention)

### Monitoring & Visualization
- **TensorBoard**: Real-time training metrics
- **Progress Bars**: tqdm integration with ETA
- **Memory Tracking**: GPU utilization monitoring
- **Performance Metrics**: Tokens/second, loss curves, learning rate

### Dataset Features
- **Streaming Support**: Memory-efficient large dataset processing
- **Language Detection**: Automatic language filtering (optional)
- **Text Quality Filtering**: Length and content quality checks
- **Dynamic Batching**: Intelligent batch size optimization
- **Multi-Dataset Training**: Sequential training across different datasets

## 🛠️ Troubleshooting

### Common Issues & Solutions

#### Memory Issues
```bash
# Reduce batch size
BATCH_SIZE = 4  # Instead of 8

# Enable gradient checkpointing  
GRADIENT_CHECKPOINTING = True

# Use smaller model
python hrm_training_nano_25m.py  # Instead of larger models
```

#### Import Errors
```bash
# Missing dependencies
pip install -r requirements.txt

# SentencePiece error (T5 tokenizer)
pip install sentencepiece protobuf

# Flash Attention (optional)
pip install flash-attn --no-build-isolation
```

#### Performance Issues
```bash
# Enable fast transfers
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install hf_transfer

# Disable import mode for training
unset HRM_IMPORT_ONLY

# Check GPU utilization
nvidia-smi
```

#### Training Stuck Issues
```bash
# Verify not in import-only mode
echo $HRM_IMPORT_ONLY  # Should be empty

# Check worker configuration
# Should see: "workers=4" not "workers=0"

# Restart with clean environment
unset HRM_IMPORT_ONLY
python hrm_training_small_50m.py
```

## 📊 Model Performance

### Benchmarks by Size

| Model | Training Speed | Memory Usage | Generation Quality | Best Use Case |
|-------|----------------|--------------|-------------------|---------------|
| Micro-10M | ~1000 tok/sec | 4GB | Basic | Research, debugging |
| Nano-25M | ~800 tok/sec | 8GB | Good | Mobile deployment |
| Small-50M | ~600 tok/sec | 12GB | Very Good | General purpose |
| Medium-100M | ~400 tok/sec | 16GB | Excellent | Production |
| Medium-350M | ~200 tok/sec | 32GB | Superior | High-quality |
| Large-1B | ~100 tok/sec | 64GB | State-of-art | Research, best results |

*Benchmarks on NVIDIA H200 with optimized settings*

### Quality Metrics
- **Coherence**: Hierarchical reasoning maintains context consistency
- **Adaptivity**: Pondering mechanism adjusts computation per complexity  
- **Efficiency**: SwiGLU and RMSNorm provide better parameter utilization
- **Stability**: Advanced mixed precision training for reliable convergence

## 📁 Output Structure

```
HRM_Models/
├── hrm_text1_c4_micro_10m_output/
│   ├── config.json              # Model configuration
│   ├── pytorch_model.bin        # Final trained model
│   ├── best_model.bin          # Best checkpoint by validation loss
│   ├── checkpoint.pth          # Training state for resuming
│   └── tensorboard_logs/       # TensorBoard training logs
├── hrm_text1_c4_nano_25m_output/
├── hrm_text1_c4_small_50m_output/
├── hrm_text1_c4_medium_100m_output/  
├── hrm_text1_c4_medium_350m_output/
└── hrm_text1_c4_large_1b_output/
```

## 🏷️ Model Releases

### Hugging Face Models
- **dreamwar/HRM-Models-Micro-10M** - Research and prototyping
- **dreamwar/HRM-Models-Nano-25M** - Mobile and edge deployment
- **dreamwar/HRM-Models-Small-50M** - General purpose applications  
- **dreamwar/HRM-Models-Medium-100M** - Production inference
- **dreamwar/HRM-Models-Medium-350M** - High-quality generation
- **dreamwar/HRM-Models-Large-1B** - State-of-the-art results

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Credits

This project is based on the original HRM-Text implementation by qingy1337:
- **Original Repository**: [https://github.com/qingy1337/HRM-Text](https://github.com/qingy1337/HRM-Text)
- **Extensions**: Multi-scale model family (10M-1B parameters), optimized training, and production improvements

## 🔬 Citation

```bibtex
@misc{hrm-models-2024,
  title={HRM-Models: Hierarchical Reasoning Model Family for Text Generation},
  author={DreamWar},
  year={2024},
  url={https://github.com/julianjjo/HRM_Models},
  note={Multi-scale transformer models with adaptive computation, based on HRM-Text by qingy1337}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### Development Setup
```bash
git clone https://github.com/julianjjo/HRM_Models.git
cd HRM_Models  
pip install -r requirements.txt
export HRM_IMPORT_ONLY=1  # For development
```

---

*The HRM-Models model family represents a significant advancement in hierarchical reasoning for text generation, offering scalable solutions from mobile edge devices to high-performance research applications.*