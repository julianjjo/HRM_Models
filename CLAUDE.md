# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is HRM-Models, a hierarchical reasoning model family for text generation with 6 different model sizes (10M to 1B parameters). The codebase implements transformer models with Hierarchical Reasoning Module (HRM) architecture featuring adaptive computation and pondering mechanisms.

## Core Architecture

The repository contains training scripts for each model size with both standard (HuggingFace ecosystem) and standalone (PyTorch-only) variants:

- **Training Scripts**: `hrm_training_{size}_{type}_hf.py` (e.g., `hrm_training_micro_10m_standalone_hf.py`)
- **Chat Interface**: `chat_hrm_hf.py` (HuggingFace integration) and `chat_hrm_standalone.py` (standalone)
- **Tokenizer Wrapper**: `hf_tokenizer_wrapper_simple.py` for HuggingFace tokenizer integration

### Model Sizes Available
- **Micro-10M**: 10.3M parameters, research and prototyping
- **Nano-25M**: 25.6M parameters, mobile and edge devices  
- **Small-50M**: 53.2M parameters, general purpose
- **Medium-100M**: 106.4M parameters, production inference
- **Medium-350M**: 353.8M parameters, high-quality generation
- **Large-1B**: 1.06B parameters, state-of-the-art results

## Training Commands

### Single GPU Training
```bash
# Standard models (with HuggingFace ecosystem)
python hrm_training_small_50m.py

# Standalone models (PyTorch only, no external dependencies)
python hrm_training_small_50m_standalone.py
```

### Multi-GPU Training (Recommended for better performance)
```bash
# Distributed training with torchrun
torchrun --nproc_per_node=2 hrm_training_medium_350m.py
torchrun --nproc_per_node=4 hrm_training_large_1b.py
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
The codebase includes basic testing functions within training scripts. No external test framework is used.

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
- Hierarchical Reasoning Module with dual-stream processing
- Adaptive computation via pondering mechanism with halt probabilities
- SwiGLU activation and RMSNorm for stable training
- Context length: 512 tokens, Vocabulary: 32,100 tokens (T5 tokenizer)

### Training Features  
- Multi-GPU distributed training support
- Mixed precision (BF16/FP16) training
- TensorBoard integration for monitoring
- Automatic checkpointing with best model tracking
- Early stopping with configurable patience

### Dataset Support
- Primary: C4-English (365M samples)
- Additional: OpenWebText, Pile, SlimPajama, FinewWeb
- Streaming support for memory-efficient large dataset processing

## Output Structure
```
HRM_Models/
├── hrm_models_{size}_output/
│   ├── config.json              # Model configuration
│   ├── pytorch_model.bin        # Final trained model  
│   ├── best_model.bin          # Best checkpoint
│   ├── checkpoint.pth          # Training state
│   └── tensorboard_logs/       # Training logs
```

## Important Notes

- Always disable `HRM_IMPORT_ONLY` for training: `unset HRM_IMPORT_ONLY`
- Use torchrun for multi-GPU training to avoid DataParallel bottlenecks
- Standalone variants have zero external dependencies beyond PyTorch
- Check GPU memory requirements: 4GB (Micro) to 64GB (Large-1B)
- Models are uploaded to HuggingFace Hub as `julianmican/hrm-text-{size}-{variant}`