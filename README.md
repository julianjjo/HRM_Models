# HRM-Text1: Hierarchical Reasoning Model for Text Generation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c4exU-zMt4SuT1kRlwQQXlLPaiazEDCf?usp=sharing)

A large-scale transformer model with Hierarchical Reasoning Module (HRM) architecture trained on multiple high-quality text datasets. This model features adaptive computation with pondering mechanisms for improved text generation quality.

## Model Architecture

**HRM-Text1** implements a novel hierarchical reasoning architecture with the following key components:

- **Model Size**: 99M parameters
- **Architecture**: Hierarchical Reasoning Module with dual-stream processing
- **Embeddings**: 512 dimensions
- **Attention Heads**: 8 heads
- **Feed-Forward**: 2048 dimensions
- **Context Length**: 512 tokens
- **Vocabulary**: 32,128 tokens (T5 tokenizer)

### Key Features

- **Adaptive Computation**: Pondering mechanism with halt probabilities
- **Dual-Stream Processing**: High-level (H) and Low-level (L) reasoning modules
- **SwiGLU Activation**: Enhanced non-linear transformations
- **RMSNorm**: Improved normalization for stable training
- **Mixed Precision**: BF16 training support for NVIDIA Ampere+ GPUs

## Training Configuration

### Dataset

The model is trained specifically on:

- **C4 Multilingual**: Common Crawl web text (multilingual dataset)
  - Dataset Size: 364M training samples, 364K validation samples
  - Language: Multilingual support with automatic language detection
  - Quality: High-quality web content filtered and processed

### Training Hyperparameters

- **Learning Rate**: 3e-4 (max) → 1e-5 (min) with cosine annealing
- **Batch Size**: 8 (with gradient accumulation steps: 2)
- **Weight Decay**: 0.05
- **Optimizer**: AdamW with β₁=0.9, β₂=0.95
- **Epochs**: 2
- **Mixed Precision**: Enabled for compatible hardware

## Model Components

### HRMBlock Architecture

```python
class HRMBlock(nn.Module):
    def __init__(self, n_embd, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = SwiGLUMuchPelu(n_embd, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
```

### Pondering Mechanism

The model implements adaptive computation through a halt probability mechanism:

- **Max Steps**: 8 reasoning steps
- **Halt Bias**: -2.2 (initial)
- **Ponder Loss Weight**: 1e-2

## Usage

### Quick Start

```python
from transformers import T5Tokenizer
from modeling_hrm_text1 import HRMText1

# Load model and tokenizer
model = HRMText1.from_pretrained("dreamwar/HRM-Text1-C4-large")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Generate text
prompt = "The future of artificial intelligence"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Training from Scratch

**Option 1: Google Colab (Recommended)**
```bash
# Open the Colab notebook
https://colab.research.google.com/drive/1c4exU-zMt4SuT1kRlwQQXlLPaiazEDCf?usp=sharing
```

**Option 2: Local Training**
```bash
# Set environment variables
export HRM_OUTPUT_BASE="/path/to/output"
export HF_TOKEN="your_huggingface_token"

# Run training
python hrm_training_small_100m.py
```

### Configuration Options

The training script supports extensive configuration:

```python
# Dataset configuration
DATASET_NAME = "allenai/c4"
DATASET_CONFIG = "multilingual"

# Dataset subset percentage (default: 0.01% for testing)
DATASET_SUBSET_PERCENT = 0.01  # 0.01-100%

# Model parameters
MODEL_PARAMS = {
    "n_embd": 512,
    "n_head": 8,
    "d_ff": 2048,
    "dropout": 0.1,
    "halt_max_steps": 8,
    "ponder_loss_weight": 1e-2,
    "halt_bias_init": -2.2
}

# Training configuration
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
```

## Features

### Dataset Optimization

- **C4 Multilingual**: Optimized for the C4 Common Crawl dataset
- **Multi-Dataset Support**: Support for Hugging Face and Kaggle datasets
- **Sequential Training**: Maintain checkpoint continuity across different datasets
- **Streaming Support**: Memory-efficient streaming for large datasets
- **Configurable Sampling**: Adjust dataset subset percentage for testing/production
- **Multilingual**: Native support for multiple languages from C4

### Training Optimizations

- **Checkpointing**: Automatic checkpoint saving and resuming
- **Sequential Training Mode**: Maintain checkpoints across dataset changes
- **Early Stopping**: Validation-based early stopping (patience: 2)
- **Gradient Clipping**: Norm clipping at 1.0
- **Mixed Precision**: BF16 for memory efficiency
- **Model Compilation**: PyTorch 2.0 compilation support

### Hardware Support

- **CUDA**: GPU acceleration with TF32 precision on Ampere+
- **Multi-Platform**: Linux, macOS, Windows support
- **Google Colab**: Full compatibility with free and pro tiers
- **Memory Management**: Automatic DataLoader worker detection

## Output Structure

```
/content/drive/MyDrive/HRM_T4/
├── hrm_text1_c4_output-large/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── best_model.bin
│   └── checkpoint.pth
```

## Environment Setup

### Quick Start with Google Colab

Click the Colab badge above to get started immediately with a pre-configured environment including all dependencies.

### Local Installation

```bash
pip install torch transformers datasets tqdm huggingface_hub
pip install langdetect  # Optional: for language filtering
```

### Environment Variables

```bash
# Required for model upload
export HF_TOKEN="your_huggingface_token"

# Optional: custom output path
export HRM_OUTPUT_BASE="/your/custom/path"
```

## Model Variant

This repository contains:

- **HRM-Text1-C4-large**: 99M parameter model trained on C4 multilingual dataset
  - Repository: `dreamwar/HRM-Text1-C4-large`
  - Architecture: Hierarchical Reasoning Module
  - Training: Optimized for Google Colab environment

## Performance

### Model Specifications

- **Parameters**: 99M trainable parameters
- **Memory Usage**: ~2-3GB VRAM for inference
- **Training Time**: Optimized for Google Colab (free tier compatible)
- **Context Length**: 512 tokens
- **Dataset**: C4 Multilingual (0.01% subset by default for testing)

### Generation Quality

The model implements sophisticated reasoning through:

- Hierarchical processing of information
- Adaptive computation based on input complexity
- Pondering mechanism for quality-vs-speed trade-offs

## License

This model and training code are released under the Apache 2.0 License.

## Citation

```bibtex
@misc{hrm-text1-2024,
  title={HRM-Text1: Hierarchical Reasoning Model for Text Generation},
  author={DreamWar},
  year={2024},
  url={https://huggingface.co/dreamwar/HRM-Text1}
}
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or enable gradient checkpointing
2. **Dataset Loading**: Ensure stable internet connection for streaming
3. **CUDA Errors**: Update PyTorch and CUDA drivers
4. **Language Detection**: Install `langdetect` for language filtering

### Support

For issues and questions:
- Check the training script comments for detailed configuration
- Review error messages for specific guidance
- Ensure proper environment setup and dependencies

## Sequential Training

The model supports sequential training across different datasets while maintaining checkpoint continuity:

### Quick Setup for Sequential Training

1. **Enable Sequential Mode** (IMPORTANT: Set this BEFORE starting training):
   ```python
   SEQUENTIAL_TRAINING = True
   BASE_MODEL_NAME = "hrm_text1_c4_output-large"
   ```

2. **Train on First Dataset**:
   ```python
   ACTIVE_DATASET = "c4"
   # Run training normally
   ```

3. **Continue with Second Dataset**:
   ```python
   ACTIVE_DATASET = "human_conversations"
   # Automatically loads checkpoint from previous training
   ```

### Configuration Example

```python
# Sequential training configuration
SEQUENTIAL_TRAINING = True  # Keep same directory across datasets
BASE_MODEL_NAME = "hrm_text1_c4_output-large"

# Dataset switching
ACTIVE_DATASET = "human_conversations"  # or "c4"

# Available datasets
DATASET_OPTIONS = {
    "c4": {
        "name": "allenai/c4",
        "config": "multilingual", 
        "type": "huggingface"
    },
    "human_conversations": {
        "name": "projjal1/human-conversation-training-data",
        "type": "kaggle"
    }
}
```

### How It Works

- **Normal Mode**: Each dataset creates its own output directory
- **Sequential Mode**: Uses a fixed base directory to preserve checkpoints
- **Automatic Detection**: Script detects dataset changes and adjusts learning rate scheduler
- **Checkpoint Continuity**: All training state is preserved between dataset switches

---

*This model was trained using the HRM (Hierarchical Reasoning Module) architecture with adaptive computation for improved text generation capabilities.*
