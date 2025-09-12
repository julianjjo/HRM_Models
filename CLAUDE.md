# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is HRM-Models, a hierarchical reasoning model family for text generation with 6 different model sizes (10M to 1B parameters). The codebase implements transformer models with Hierarchical Reasoning Module (HRM) architecture featuring adaptive computation and pondering mechanisms, all based on the proven 10M architecture.

## Core Architecture

The repository contains training scripts for each model size:

- **Training Scripts**: `hrm_training_{size}_standalone_hf.py` (e.g., `hrm_training_micro_10m_standalone_hf.py`)
- **Chat Interface**: `chat_hrm_hf.py` (HuggingFace integration) and `chat_hrm_standalone.py` (standalone)
- **Tokenizer Wrapper**: `hf_tokenizer_wrapper_simple.py` for HuggingFace tokenizer integration
- **Test Script**: `test_hrm_micro_10m.py` for comprehensive model testing

### Model Sizes Available
- **Micro-10M**: ~10M parameters, research and prototyping (256 emb, 4 layers, 128 ctx)
- **Nano-25M**: ~25M parameters, mobile and edge devices (384 emb, 6 layers, 256 ctx)
- **Small-50M**: ~50M parameters, general purpose (512 emb, 8 layers, 512 ctx)
- **Medium-100M**: ~100M parameters, production inference (768 emb, 12 layers, 1024 ctx)
- **Medium-350M**: ~350M parameters, high-quality generation (1024 emb, 16 layers, 1024 ctx)
- **Large-1B**: ~1B parameters, state-of-the-art results (1536 emb, 24 layers, 1024 ctx)

## Training Commands

### Single GPU Training
```bash
# Micro model (10M) - Fast prototyping
python hrm_training_micro_10m_standalone_hf.py

# Nano model (25M) - Edge devices  
python hrm_training_nano_25m_standalone_hf.py

# Small model (50M) - General purpose
python hrm_training_small_50m_standalone_hf.py

# Medium models (100M/350M) - Production
python hrm_training_medium_100m_standalone_hf.py
python hrm_training_medium_350m_standalone_hf.py
```

### Multi-GPU Training (Recommended for larger models)
```bash
# Distributed training with torchrun for better performance
torchrun --nproc_per_node=2 hrm_training_medium_350m_standalone_hf.py
torchrun --nproc_per_node=4 hrm_training_large_1b_standalone_hf.py

# Large model (1B) - State-of-the-art (requires significant GPU memory)
torchrun --nproc_per_node=8 hrm_training_large_1b_standalone_hf.py
```

### Environment Variables
```bash
# Fast import mode (for development, no training setup)
export HRM_IMPORT_ONLY=1

# Fast HuggingFace Hub transfers
export HF_HUB_ENABLE_HF_TRANSFER=1

# Custom output directory
export HRM_OUTPUT_BASE="/path/to/output"
```

## Development Commands

### Testing
```bash
# Test trained models with comprehensive analysis
python test_hrm_micro_10m.py --model_path ./hrm-micro-10m-hf/final_model

# Full test suite (generation, HRM analysis, perplexity, benchmark)
python test_hrm_micro_10m.py --model_path ./model_path --test_generation --test_hrm --test_perplexity --benchmark
```

### Code Quality
```bash
# Install development dependencies (includes black, flake8, pytest)
pip install -r requirements-dev.txt

# Format code
black *.py

# Lint code
flake8 *.py
```

### Dependencies
```bash
# Minimal installation (standalone models only)
pip install -r requirements-minimal.txt

# Full installation (all features)  
pip install -r requirements.txt

# Development installation (includes testing and linting tools)
pip install -r requirements-dev.txt
```

## Key Technical Details

### HRM Architecture Components
- Hierarchical Reasoning Module with dual-stream processing (H-module and L-module)
- Adaptive computation via pondering mechanism with halt probabilities
- Q-learning based halting decisions for optimal computation
- SwiGLU activation and RMSNorm for stable training
- Rotary Position Embeddings (RoPE) for better positional encoding
- Deep Supervision with intermediate prediction heads
- Context length: 128-1024 tokens (model dependent), Vocabulary: 50,257 tokens (GPT-2 tokenizer)

### Training Features  
- Multi-GPU distributed training support
- Mixed precision (BF16/FP16) training
- TensorBoard integration for monitoring
- Automatic checkpointing with best model tracking

### Dataset Support
- Primary: C4-English (365M samples)
- Additional: OpenWebText, Pile, SlimPajama, FinewWeb
- Streaming support for memory-efficient large dataset processing

## Output Structure
```
HRM_Models/
├── hrm-{size}-hf/              # Model output directory
│   ├── final_model/            # Final trained model
│   │   ├── config.json         # Model configuration
│   │   ├── pytorch_model.bin   # Model weights
│   │   └── tokenizer files     # HuggingFace tokenizer
│   ├── best_model/             # Best checkpoint during training
│   └── checkpoint-{step}/      # Intermediate checkpoints
│
├── Available model outputs:
│   ├── hrm-micro-10m-hf/       # ~10M parameters
│   ├── hrm-nano-25m-hf/        # ~25M parameters
│   ├── hrm-small-50m-hf/       # ~50M parameters
│   ├── hrm-medium-100m-hf/     # ~100M parameters
│   ├── hrm-medium-350m-hf/     # ~350M parameters
│   └── hrm-large-1b-hf/        # ~1B parameters
```

## Important Notes

- Always disable `HRM_IMPORT_ONLY` for training: `unset HRM_IMPORT_ONLY`
- Use torchrun for multi-GPU training to avoid DataParallel bottlenecks
- Standalone variants have zero external dependencies beyond PyTorch
- GPU memory requirements: 
  - Micro-10M: ~4GB VRAM
  - Nano-25M: ~6GB VRAM  
  - Small-50M: ~8GB VRAM
  - Medium-100M: ~16GB VRAM
  - Medium-350M: ~32GB VRAM
  - Large-1B: ~64GB VRAM (multi-GPU recommended)
- Models are uploaded to HuggingFace Hub as `julianmican/hrm-text-{size}-hf`
- All models based on the proven Micro-10M architecture with scaled parameters